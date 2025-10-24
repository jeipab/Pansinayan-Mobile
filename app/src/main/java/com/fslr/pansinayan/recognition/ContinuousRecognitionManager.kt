package com.fslr.pansinayan.recognition

import android.content.Context
import android.util.Log
import androidx.camera.core.ImageProxy
import androidx.lifecycle.LifecycleOwner
import com.fslr.pansinayan.camera.CameraManager
import com.fslr.pansinayan.inference.CTCModelInterpreter
import com.fslr.pansinayan.inference.CTCDecoder
import com.fslr.pansinayan.inference.CTCSequenceBufferManager
import com.fslr.pansinayan.mediapipe.MediaPipeProcessor
import com.fslr.pansinayan.utils.LabelMapper
import kotlinx.coroutines.*
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/**
 * Orchestrates continuous CTC-based sign language recognition pipeline.
 * 
 * Pipeline flow:
 * Camera → MediaPipe → CTC Buffer → CTC Model → CTC Decoder → Continuous Transcript
 * 
 * Responsibilities:
 * - Coordinate all CTC components
 * - Manage continuous recognition without resets
 * - Handle threading and async operations
 * - Implement continuous inference scheduling
 * - Monitor pipeline health and recover from freezes
 * - Handle recording state changes
 * 
 * Usage:
 *   val pipeline = ContinuousRecognitionManager(context, lifecycleOwner, previewView) { transcript ->
 *       // Update UI with continuous transcript
 *       updateTranscriptDisplay(transcript)
 *   }
 *   pipeline.start()
 *   // ... later ...
 *   pipeline.stop()
 */
class ContinuousRecognitionManager(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: androidx.camera.view.PreviewView,
    private val onTranscriptUpdate: (ContinuousTranscript) -> Unit,
    private val onFrameUpdate: ((keypoints: FloatArray?, imageWidth: Int, imageHeight: Int, isOccluded: Boolean) -> Unit)? = null
) : MediaPipeProcessor.KeypointListener {
    companion object {
        private const val TAG = "ContinuousRecognitionManager"
        private const val INFERENCE_INTERVAL = 15  // Run inference every N frames (every ~500ms at 30 FPS)
        private const val HEALTH_CHECK_INTERVAL_MS = 2000L  // Check pipeline health every 2 seconds
        private const val FRAME_TIMEOUT_MS = 3000L  // Consider pipeline frozen after 3 seconds
        private const val MOTION_SKIP_THRESHOLD = 10  // Skip inference if no motion for N frames
    }

    // Core components
    private lateinit var cameraManager: CameraManager
    private lateinit var mediaPipeProcessor: MediaPipeProcessor
    private lateinit var bufferManager: CTCSequenceBufferManager
    private lateinit var modelInterpreter: CTCModelInterpreter
    private lateinit var ctcDecoder: CTCDecoder
    private lateinit var labelMapper: LabelMapper

    // Coroutine scope for async operations
    private val pipelineScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    
    // Frame counter
    private val frameCounter = AtomicInteger(0)
    
    // State
    private val isRunning = AtomicBoolean(false)
    private val isPaused = AtomicBoolean(false)
    private val isRecording = AtomicBoolean(false)
    private val isRestartingMediaPipe = AtomicBoolean(false)
    
    // Health monitoring
    private val lastFrameTime = AtomicLong(0)
    private val lastKeypointTime = AtomicLong(0)
    private var healthMonitorJob: Job? = null

    // Continuous transcript management
    private val transcriptHistory = mutableListOf<String>()
    private val maxTranscriptLength = 20  // Keep last 20 recognized phrases
    private var lastRecognizedPhrase = ""
    private var lastInferenceTime = 0L

    /**
     * Initialize all CTC components.
     * Call this before start().
     */
    fun initialize() {
        try {
            Log.i(TAG, "Initializing continuous CTC recognition pipeline...")

            cameraManager = CameraManager(context, lifecycleOwner, previewView, targetFps = 30)
            mediaPipeProcessor = MediaPipeProcessor(context, this)
            bufferManager = CTCSequenceBufferManager(maxWindowSize = 300, maxGap = 5)
            
            // Initialize CTC model interpreter
            modelInterpreter = CTCModelInterpreter(context, modelPath = "ctc/sign_transformer_ctc_fp16.tflite")
            
            // Initialize CTC decoder
            labelMapper = LabelMapper(context)
            ctcDecoder = CTCDecoder(labelMapper)

            Log.i(TAG, "All CTC components initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize CTC pipeline", e)
            throw e
        }
    }

    /**
     * Start the continuous recognition pipeline.
     */
    fun start() {
        if (isRunning.get()) {
            Log.w(TAG, "CTC pipeline already running")
            return
        }

        isRunning.set(true)
        isPaused.set(false)
        frameCounter.set(0)
        lastFrameTime.set(System.currentTimeMillis())
        lastKeypointTime.set(System.currentTimeMillis())
        
        Log.i(TAG, "Starting continuous CTC recognition pipeline...")

        // Start camera with frame callback
        cameraManager.startCamera { frame ->
            if (!isPaused.get()) {
                processFrame(frame)
            } else {
                frame.close()
            }
        }
        
        // Start health monitoring
        startHealthMonitor()
    }

    private fun processFrame(imageProxy: ImageProxy) {
        lastFrameTime.set(System.currentTimeMillis())
        
        // Skip frames if MediaPipe is restarting (prevents crash)
        if (isRestartingMediaPipe.get()) {
            imageProxy.close()
            return
        }
        
        mediaPipeProcessor.detectLiveStream(imageProxy, isFrontCamera = cameraManager.isFrontCamera())
    }

    override fun onKeypointsExtracted(keypoints: FloatArray?, imageWidth: Int, imageHeight: Int) {
        if (isPaused.get()) return
        
        lastKeypointTime.set(System.currentTimeMillis())
        
        pipelineScope.launch {
            try {
                val currentFrame = frameCounter.incrementAndGet()
                
                // Use hand-face occlusion detection
                val isOccluded = keypoints?.let { mediaPipeProcessor.detectHandFaceOcclusion(it) } ?: false
                
                onFrameUpdate?.let { callback ->
                    withContext(Dispatchers.Main) {
                        callback(keypoints, imageWidth, imageHeight, isOccluded)
                    }
                }
                
                // Add frame to CTC buffer
                bufferManager.addFrame(keypoints)

                // Run CTC inference periodically
                if (currentFrame % INFERENCE_INTERVAL == 0) {
                    runContinuousInference()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Frame processing failed", e)
            }
        }
    }

    override fun onError(error: String) {
        Log.e(TAG, "MediaPipe error: $error")
    }

    /**
     * Run continuous CTC inference.
     */
    private suspend fun runContinuousInference() {
        // Get current sequence from buffer
        val sequence = bufferManager.getCurrentSequence() ?: run {
            Log.d(TAG, "Buffer not ready (size: ${bufferManager.getBufferSize()})")
            return
        }

        // Run CTC inference
        val logits = modelInterpreter.runInference(sequence) ?: run {
            Log.w(TAG, "CTC inference failed")
            return
        }

        // Decode CTC output
        val decodedPhrase = ctcDecoder.decode(logits, sequence.size)
        
        // Process continuous transcript
        processContinuousTranscript(decodedPhrase)
    }

    /**
     * Process continuous transcript updates.
     */
    private suspend fun processContinuousTranscript(decodedPhrase: String) {
        if (decodedPhrase.isEmpty()) return

        val currentTime = System.currentTimeMillis()
        
        // Check if this is a new phrase or continuation
        val isNewPhrase = decodedPhrase != lastRecognizedPhrase
        val timeSinceLastInference = currentTime - lastInferenceTime
        
        if (isNewPhrase || timeSinceLastInference > 2000) { // New phrase or 2+ seconds gap
            // Add to transcript history
            transcriptHistory.add(decodedPhrase)
            
            // Maintain max history length
            while (transcriptHistory.size > maxTranscriptLength) {
                transcriptHistory.removeAt(0)
            }
            
            // Create continuous transcript
            val fullTranscript = transcriptHistory.joinToString(" ")
            
            val continuousTranscript = ContinuousTranscript(
                currentPhrase = decodedPhrase,
                fullTranscript = fullTranscript,
                timestamp = currentTime,
                phraseCount = transcriptHistory.size
            )

            // Update UI on main thread
            withContext(Dispatchers.Main) {
                onTranscriptUpdate(continuousTranscript)
            }

            lastRecognizedPhrase = decodedPhrase
            lastInferenceTime = currentTime
            
            Log.i(TAG, "Continuous transcript: '$decodedPhrase' (total: ${transcriptHistory.size} phrases)")
        }
    }

    /**
     * Start health monitoring to detect frozen pipeline.
     */
    private fun startHealthMonitor() {
        healthMonitorJob?.cancel()
        healthMonitorJob = pipelineScope.launch {
            while (isActive && isRunning.get()) {
                delay(HEALTH_CHECK_INTERVAL_MS)
                
                if (!isPaused.get()) {
                    checkPipelineHealth()
                }
            }
        }
    }

    /**
     * Check pipeline health and recover if necessary.
     */
    private fun checkPipelineHealth() {
        val currentTime = System.currentTimeMillis()
        val timeSinceLastFrame = currentTime - lastFrameTime.get()
        val timeSinceLastKeypoint = currentTime - lastKeypointTime.get()
        
        // Check if pipeline is frozen
        if (timeSinceLastFrame > FRAME_TIMEOUT_MS) {
            Log.w(TAG, "Camera frame timeout detected! Time: ${timeSinceLastFrame}ms")
            recoverPipeline("Camera frame timeout")
        } else if (timeSinceLastKeypoint > FRAME_TIMEOUT_MS) {
            Log.w(TAG, "Keypoint extraction timeout detected! Time: ${timeSinceLastKeypoint}ms")
            recoverPipeline("Keypoint extraction timeout")
        }
    }

    /**
     * Recover pipeline when frozen.
     */
    private fun recoverPipeline(reason: String) {
        Log.i(TAG, "Starting CTC pipeline recovery. Reason: $reason")
        
        pipelineScope.launch {
            try {
                // Keypoint timeout = MediaPipe is frozen, needs restart
                if (reason.contains("Keypoint") || reason.contains("keypoint")) {
                    Log.i(TAG, "MediaPipe frozen detected, restarting MediaPipe processor...")
                    withContext(Dispatchers.IO) {
                        safeRestartMediaPipe()
                    }
                }
                // Camera timeout = camera issue, restart camera
                else if (reason.contains("Camera") || reason.contains("frame")) {
                    Log.i(TAG, "Camera frame timeout, restarting camera...")
                    cameraManager.restartCamera()
                    lastFrameTime.set(System.currentTimeMillis())
                }
                // Fallback: full restart
                else {
                    Log.i(TAG, "Unknown issue, full restart...")
                    withContext(Dispatchers.Main) {
                        pause()
                        delay(500)
                        resume()
                    }
                }
                
                Log.i(TAG, "CTC pipeline recovery completed")
            } catch (e: Exception) {
                Log.e(TAG, "CTC pipeline recovery failed", e)
            }
        }
    }

    /**
     * Notify pipeline that recording has started.
     */
    fun onRecordingStarted() {
        Log.i(TAG, "Recording started, protecting CTC pipeline...")
        isRecording.set(true)
        
        // Proactively restart MediaPipe to prevent freeze
        pipelineScope.launch {
            delay(300)
            
            if (isRunning.get() && !isPaused.get()) {
                Log.i(TAG, "Proactively restarting MediaPipe to prevent recording conflict...")
                withContext(Dispatchers.IO) {
                    try {
                        safeRestartMediaPipe()
                        Log.i(TAG, "MediaPipe restarted successfully for recording")
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to restart MediaPipe for recording", e)
                    }
                }
            }
        }
    }

    /**
     * Notify pipeline that recording has stopped.
     */
    fun onRecordingStopped() {
        Log.i(TAG, "Recording stopped, resuming normal CTC operation...")
        isRecording.set(false)
        
        // Restart MediaPipe to ensure clean state after recording
        pipelineScope.launch {
            delay(300)
            
            if (isRunning.get() && !isPaused.get()) {
                Log.i(TAG, "Restarting MediaPipe after recording stopped...")
                withContext(Dispatchers.IO) {
                    try {
                        safeRestartMediaPipe()
                        Log.i(TAG, "MediaPipe restarted successfully after recording")
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to restart MediaPipe after recording", e)
                    }
                }
            }
        }
    }

    /**
     * Stop the continuous recognition pipeline.
     */
    fun stop() {
        if (!isRunning.get()) {
            return
        }

        isRunning.set(false)
        
        Log.i(TAG, "Stopping continuous CTC recognition pipeline...")

        // Stop health monitor
        healthMonitorJob?.cancel()

        // Stop camera
        cameraManager.stopCamera()

        // Clear buffer
        bufferManager.clear()

        // Reset CTC decoder
        ctcDecoder.reset()

        Log.i(TAG, "CTC pipeline stopped")
    }

    /**
     * Release all resources.
     * Call this in onDestroy().
     */
    fun release() {
        stop()
        
        // Now cancel the scope (activity is being destroyed)
        pipelineScope.cancel()
        
        cameraManager.release()
        mediaPipeProcessor.release()
        modelInterpreter.release()
        
        Log.i(TAG, "All CTC resources released")
    }

    /**
     * Get pipeline statistics for debugging.
     */
    fun getStats(): ContinuousPipelineStats {
        val (cameraProcessed, cameraTotal) = cameraManager.getStats()
        val (mediapipeSuccess, mediapipeFailure) = mediaPipeProcessor.getStats()
        val avgInferenceTime = modelInterpreter.getAverageInferenceTime()
        val ctcStats = ctcDecoder.getDetailedStats()
        val bufferStats = bufferManager.getBufferStats()
        
        val timeSinceLastFrame = System.currentTimeMillis() - lastFrameTime.get()
        val timeSinceLastKeypoint = System.currentTimeMillis() - lastKeypointTime.get()
        
        return ContinuousPipelineStats(
            framesProcessed = cameraProcessed,
            framesTotal = cameraTotal,
            keypointSuccess = mediapipeSuccess,
            keypointFailure = mediapipeFailure,
            avgInferenceTimeMs = avgInferenceTime,
            ctcStats = ctcStats,
            bufferStats = bufferStats,
            timeSinceLastFrame = timeSinceLastFrame,
            timeSinceLastKeypoint = timeSinceLastKeypoint,
            isRecording = isRecording.get(),
            transcriptPhrases = transcriptHistory.size
        )
    }

    /**
     * Pause recognition (stop processing but keep resources).
     */
    fun pause() {
        isPaused.set(true)
        Log.i(TAG, "CTC pipeline paused")
    }

    /**
     * Resume recognition after pause.
     */
    fun resume() {
        isPaused.set(false)
        lastFrameTime.set(System.currentTimeMillis())
        lastKeypointTime.set(System.currentTimeMillis())
        Log.i(TAG, "CTC pipeline resumed")
    }

    /**
     * Switch between front and back camera.
     */
    fun switchCamera() {
        cameraManager.switchCamera()
        Log.i(TAG, "Camera switched to ${if (isFrontCamera()) "front" else "back"}")
    }

    /**
     * Check if currently using front camera.
     */
    fun isFrontCamera(): Boolean {
        return cameraManager.isFrontCamera()
    }
    
    /**
     * Manually restart MediaPipe processors.
     * Useful for testing or recovering from frozen state.
     */
    fun restartMediaPipe() {
        Log.i(TAG, "Manual MediaPipe restart called")
        safeRestartMediaPipe()
    }
    
    /**
     * Safely restart MediaPipe by blocking frames during restart.
     * Prevents race condition crashes.
     */
    private fun safeRestartMediaPipe() {
        if (isRestartingMediaPipe.getAndSet(true)) {
            Log.w(TAG, "MediaPipe restart already in progress, skipping...")
            return
        }
        
        try {
            Log.i(TAG, "Blocking frames for MediaPipe restart...")
            
            // Wait for in-flight frames to clear (max 200ms)
            Thread.sleep(200)
            
            // Restart MediaPipe
            mediaPipeProcessor.restart()
            
            // Reset timestamps
            lastKeypointTime.set(System.currentTimeMillis())
            lastFrameTime.set(System.currentTimeMillis())
            
            Log.i(TAG, "MediaPipe restart completed, resuming frames")
            
        } catch (e: Exception) {
            Log.e(TAG, "MediaPipe restart failed", e)
        } finally {
            // Always unblock frames
            isRestartingMediaPipe.set(false)
        }
    }

    /**
     * Clear transcript history.
     */
    fun clearTranscript() {
        transcriptHistory.clear()
        lastRecognizedPhrase = ""
        Log.i(TAG, "Transcript history cleared")
    }
}

/**
 * Represents a continuous transcript update.
 */
data class ContinuousTranscript(
    val currentPhrase: String,
    val fullTranscript: String,
    val timestamp: Long,
    val phraseCount: Int
)

/**
 * Pipeline statistics for debugging and monitoring.
 */
data class ContinuousPipelineStats(
    val framesProcessed: Int,
    val framesTotal: Int,
    val keypointSuccess: Int,
    val keypointFailure: Int,
    val avgInferenceTimeMs: Long,
    val ctcStats: com.fslr.pansinayan.inference.CTCStats,
    val bufferStats: com.fslr.pansinayan.inference.CTCBufferStats,
    val timeSinceLastFrame: Long,
    val timeSinceLastKeypoint: Long,
    val isRecording: Boolean,
    val transcriptPhrases: Int
) {
    override fun toString(): String {
        // Format stats in 2 columns for compact display
        val col1Line1 = "Frames: $framesProcessed/$framesTotal"
        val col2Line1 = "Inference: ${avgInferenceTimeMs}ms"
        val col1Line2 = "Keypoints: $keypointSuccess/$keypointFailure"
        val col2Line2 = "Buffer: ${bufferStats.totalFrames}"
        val col1Line3 = "CTC: ${ctcStats.successfulDecodings}/${ctcStats.totalDecodings}"
        val col2Line3 = "Transcript: $transcriptPhrases phrases"
        val col1Line4 = "Last frame: ${timeSinceLastFrame}ms"
        val col2Line4 = "Recording: ${if (isRecording) "Yes" else "No"}"
        
        return """
            ${col1Line1.padEnd(28)} ${col2Line1}
            ${col1Line2.padEnd(28)} ${col2Line2}
            ${col1Line3.padEnd(28)} ${col2Line3}
            ${col1Line4.padEnd(28)} ${col2Line4}
        """.trimIndent()
    }
}

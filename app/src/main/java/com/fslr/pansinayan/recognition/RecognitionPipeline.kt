package com.fslr.pansinayan.recognition

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import androidx.lifecycle.LifecycleOwner
import com.fslr.pansinayan.camera.CameraManager
import com.fslr.pansinayan.inference.SequenceBufferManager
import com.fslr.pansinayan.inference.TFLiteModelRunner
import com.fslr.pansinayan.mediapipe.MediaPipeProcessor
import com.fslr.pansinayan.utils.LabelMapper
import kotlinx.coroutines.*
import java.util.concurrent.atomic.AtomicInteger

/**
 * Orchestrates the complete recognition pipeline.
 * 
 * Pipeline flow:
 * Camera → MediaPipe → Buffer → TFLite → Temporal Logic → UI Callback
 * 
 * Responsibilities:
 * - Coordinate all components
 * - Manage threading and async operations
 * - Handle lifecycle (start/stop/pause)
 * - Implement inference scheduling (don't run every frame)
 * - Provide callbacks for UI updates
 * 
 * Usage:
 *   val pipeline = RecognitionPipeline(context, lifecycleOwner, previewView) { event ->
 *       // Update UI with recognized sign
 *       updateDisplay(event.label, event.confidence)
 *   }
 *   pipeline.start()
 *   // ... later ...
 *   pipeline.stop()
 */
class RecognitionPipeline(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: androidx.camera.view.PreviewView,
    private val onSignRecognized: (RecognizedSign) -> Unit,
    private val onFrameUpdate: ((keypoints: FloatArray?, isOccluded: Boolean) -> Unit)? = null
) {
    companion object {
        private const val TAG = "RecognitionPipeline"
        private const val INFERENCE_INTERVAL = 10  // Run inference every N frames
    }

    // Core components
    private lateinit var cameraManager: CameraManager
    private lateinit var mediaPipeProcessor: MediaPipeProcessor
    private lateinit var bufferManager: SequenceBufferManager
    private lateinit var modelRunner: TFLiteModelRunner
    private lateinit var temporalRecognizer: TemporalRecognizer
    private lateinit var labelMapper: LabelMapper

    // Coroutine scope for async operations
    private val pipelineScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    
    // Frame counter
    private val frameCounter = AtomicInteger(0)
    
    // State
    private var isRunning = false

    /**
     * Initialize all components.
     * Call this before start().
     */
    fun initialize() {
        try {
            Log.i(TAG, "Initializing recognition pipeline...")

            cameraManager = CameraManager(context, lifecycleOwner, previewView, targetFps = 15)
            mediaPipeProcessor = MediaPipeProcessor(context)
            bufferManager = SequenceBufferManager(windowSize = 90, maxGap = 5)
            modelRunner = TFLiteModelRunner(context, modelPath = "sign_transformer_quant.tflite")
            temporalRecognizer = TemporalRecognizer(
                stabilityThreshold = 5,
                confidenceThreshold = 0.6f,
                cooldownMs = 1000
            )
            labelMapper = LabelMapper(context)

            Log.i(TAG, "All components initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize pipeline", e)
            throw e
        }
    }

    /**
     * Start the recognition pipeline.
     */
    fun start() {
        if (isRunning) {
            Log.w(TAG, "Pipeline already running")
            return
        }

        isRunning = true
        frameCounter.set(0)
        
        Log.i(TAG, "Starting recognition pipeline...")

        // Start camera with frame callback
        cameraManager.startCamera { frame ->
            processFrame(frame)
        }
    }

    /**
     * Process a single frame through the pipeline.
     */
    private fun processFrame(frame: Bitmap) {
        pipelineScope.launch {
            try {
                val currentFrame = frameCounter.incrementAndGet()

                // Step 1: Extract keypoints from frame
                val keypoints = mediaPipeProcessor.extractKeypoints(frame)

                if (keypoints == null) {
                    Log.w(TAG, "Failed to extract keypoints from frame $currentFrame")
                } else {
                    Log.d(TAG, "Successfully extracted keypoints from frame $currentFrame")
                }

                // Step 2: Detect occlusion
                val isOccluded = keypoints?.let { mediaPipeProcessor.detectOcclusion(it) } ?: false
                
                // Step 3: Notify frame update callback (for UI updates like skeleton overlay)
                onFrameUpdate?.let { callback ->
                    withContext(Dispatchers.Main) {
                        callback(keypoints, isOccluded)
                    }
                }
                
                // Step 4: Add to sliding window buffer
                bufferManager.addFrame(keypoints)

                // Step 5: Run inference every N frames (not every frame for performance)
                if (currentFrame % INFERENCE_INTERVAL == 0) {
                    runInferenceIfReady()
                }

            } catch (e: Exception) {
                Log.e(TAG, "Frame processing failed", e)
            }
        }
    }

    /**
     * Run inference if buffer has enough data.
     */
    private suspend fun runInferenceIfReady() {
        // Get current sequence from buffer
        val sequence = bufferManager.getSequence() ?: run {
            Log.d(TAG, "Buffer not ready (size: ${bufferManager.getBufferSize()})")
            return
        }

        // Run TFLite inference
        val inferenceResult = modelRunner.runInference(sequence) ?: run {
            Log.w(TAG, "Inference failed")
            return
        }

        // Process through temporal recognizer
        val recognitionEvent = temporalRecognizer.processNewPrediction(
            glossId = inferenceResult.glossPrediction,
            confidence = inferenceResult.glossConfidence
        )

        // If stable sign detected, emit to UI
        if (recognitionEvent != null) {
            val label = labelMapper.getGlossLabel(recognitionEvent.glossId)
            val recognizedSign = RecognizedSign(
                glossId = recognitionEvent.glossId,
                label = label,
                confidence = recognitionEvent.confidence,
                timestamp = recognitionEvent.timestamp
            )

            // Invoke callback on main thread
            withContext(Dispatchers.Main) {
                onSignRecognized(recognizedSign)
            }

            Log.i(TAG, "Sign recognized: $label (conf: ${recognitionEvent.confidence})")
        }
    }

    /**
     * Stop the recognition pipeline.
     */
    fun stop() {
        if (!isRunning) {
            return
        }

        isRunning = false
        
        Log.i(TAG, "Stopping recognition pipeline...")

        // Stop camera
        cameraManager.stopCamera()

        // Cancel all coroutines
        pipelineScope.cancel()

        // Clear buffer
        bufferManager.clear()

        // Reset temporal state
        temporalRecognizer.reset()

        Log.i(TAG, "Pipeline stopped")
    }

    /**
     * Release all resources.
     * Call this in onDestroy().
     */
    fun release() {
        stop()
        
        mediaPipeProcessor.release()
        modelRunner.release()
        
        Log.i(TAG, "All resources released")
    }

    /**
     * Get pipeline statistics for debugging.
     */
    fun getStats(): PipelineStats {
        val (cameraProcessed, cameraTotal) = cameraManager.getStats()
        val (mediapipeSuccess, mediapipeFailure) = mediaPipeProcessor.getStats()
        val avgInferenceTime = modelRunner.getAverageInferenceTime()
        val temporalStats = temporalRecognizer.getStats()
        
        return PipelineStats(
            framesProcessed = cameraProcessed,
            framesTotal = cameraTotal,
            keypointSuccess = mediapipeSuccess,
            keypointFailure = mediapipeFailure,
            avgInferenceTimeMs = avgInferenceTime,
            temporalStats = temporalStats,
            bufferSize = bufferManager.getBufferSize()
        )
    }

    /**
     * Pause recognition (stop processing but keep resources).
     */
    fun pause() {
        // Implement pause logic if needed
        Log.i(TAG, "Pipeline paused")
    }

    /**
     * Resume recognition after pause.
     */
    fun resume() {
        // Implement resume logic if needed
        Log.i(TAG, "Pipeline resumed")
    }
}

/**
 * Represents a recognized sign with metadata.
 */
data class RecognizedSign(
    val glossId: Int,
    val label: String,
    val confidence: Float,
    val timestamp: Long
)

/**
 * Pipeline statistics for debugging and monitoring.
 */
data class PipelineStats(
    val framesProcessed: Int,
    val framesTotal: Int,
    val keypointSuccess: Int,
    val keypointFailure: Int,
    val avgInferenceTimeMs: Long,
    val temporalStats: String,
    val bufferSize: Int
) {
    override fun toString(): String {
        return """
            Pipeline Stats:
            - Frames: $framesProcessed / $framesTotal
            - Keypoints: $keypointSuccess success / $keypointFailure failures
            - Inference: ${avgInferenceTimeMs}ms avg
            - Buffer: $bufferSize frames
            - Temporal: $temporalStats
        """.trimIndent()
    }
}


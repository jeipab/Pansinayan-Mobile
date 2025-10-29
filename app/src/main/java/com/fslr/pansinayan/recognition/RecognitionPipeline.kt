package com.fslr.pansinayan.recognition

import android.content.Context
import android.util.Log
import androidx.camera.core.ImageProxy
import androidx.lifecycle.LifecycleOwner
import com.fslr.pansinayan.camera.CameraManager
import com.fslr.pansinayan.inference.CTCDecoder
import com.fslr.pansinayan.inference.ModelRunner
import com.fslr.pansinayan.inference.PyTorchModelRunner
import com.fslr.pansinayan.inference.PreprocessingUtils
import com.fslr.pansinayan.inference.SequenceBufferManager
import com.fslr.pansinayan.mediapipe.MediaPipeProcessor
import com.fslr.pansinayan.utils.LabelMapper
import kotlinx.coroutines.*
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/**
 * Orchestrates the complete recognition pipeline with health monitoring.
 * 
 * Pipeline flow:
 * Camera → MediaPipe → Buffer → PyTorch → Temporal Logic → UI Callback
 * 
 * Responsibilities:
 * - Coordinate all components
 * - Manage threading and async operations
 * - Handle lifecycle (start/stop/pause)
 * - Implement inference scheduling
 * - Monitor pipeline health and recover from freezes
 * - Handle recording state changes
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
    private val onFrameUpdate: ((keypoints: FloatArray?, imageWidth: Int, imageHeight: Int, isOccluded: Boolean) -> Unit)? = null
) : MediaPipeProcessor.KeypointListener {
    companion object {
        private const val TAG = "RecognitionPipeline"
        private const val INFERENCE_INTERVAL = 10  // Run inference every N frames
        private const val HEALTH_CHECK_INTERVAL_MS = 2000L  // Check pipeline health every 2 seconds
        private const val FRAME_TIMEOUT_MS = 3000L  // Consider pipeline frozen after 3 seconds
    }

    // Core components
    private lateinit var cameraManager: CameraManager
    private lateinit var mediaPipeProcessor: MediaPipeProcessor
    private lateinit var bufferManager: SequenceBufferManager
    private lateinit var ctcRunner: ModelRunner
    private lateinit var temporalRecognizer: TemporalRecognizer
    private lateinit var ctcAggregator: CtcAggregator
    private var ctcWindowSize: Int = 120
    private var ctcStride: Int = 40
    private lateinit var labelMapper: LabelMapper
    private var totalInferenceTimeMs: Long = 0L
    private var inferenceCount: Long = 0L
    private var debugLogging: Boolean = false

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

    /**
     * Initialize all components.
     * Call this before start().
     */
    fun initialize() {
        try {
            Log.i(TAG, "Initializing recognition pipeline...")

            cameraManager = CameraManager(context, lifecycleOwner, previewView, targetFps = 30)
            mediaPipeProcessor = MediaPipeProcessor(context, this)

            // Prefer PyTorch .pt if present; fallback to TFLite
            // Strictly use PyTorch; fail fast if asset is missing
            context.assets.open("ctc/SignTransformerCtc_best.pt").close()
            ctcRunner = PyTorchModelRunner(
                context = context,
                assetModelPath = "ctc/SignTransformerCtc_best.pt",
                metadataPath = "ctc/SignTransformerCtc_best.model.json"
            )

            ctcWindowSize = ctcRunner.meta.window_size_hint
            ctcStride = ctcRunner.meta.stride_hint
            bufferManager = SequenceBufferManager(windowSize = ctcWindowSize, maxGap = 5)
            ctcAggregator = CtcAggregator(iouThreshold = 0.5f)
            
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
        if (isRunning.get()) {
            Log.w(TAG, "Pipeline already running")
            return
        }

        isRunning.set(true)
        isPaused.set(false)
        frameCounter.set(0)
        lastFrameTime.set(System.currentTimeMillis())
        lastKeypointTime.set(System.currentTimeMillis())
        
        Log.i(TAG, "Starting recognition pipeline...")

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

    fun setDebugLogging(enabled: Boolean) {
        debugLogging = enabled
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
                
                // Use new hand-face occlusion detection (aligned with preprocessing pipeline)
                val isOccluded = keypoints?.let { mediaPipeProcessor.detectHandFaceOcclusion(it) } ?: false
                
                onFrameUpdate?.let { callback ->
                    withContext(Dispatchers.Main) {
                        callback(keypoints, imageWidth, imageHeight, isOccluded)
                    }
                }
                
                bufferManager.addFrame(keypoints)

                runCtcIfReady(currentFrame)
            } catch (e: Exception) {
                Log.e(TAG, "Frame processing failed", e)
            }
        }
    }

    override fun onError(error: String) {
        Log.e(TAG, "MediaPipe error: $error")
    }

    /**
     * Run inference if buffer has enough data.
     */
    private suspend fun runCtcIfReady(currentFrame: Int) {
        val pop = bufferManager.popWindowIfReady(ctcStride) ?: return
        val (windowSeq, missingRatio) = pop
        if (missingRatio > 0.5f) {
            Log.d(TAG, "Skipping inference due to missing ratio: $missingRatio")
            return
        }

        try {
            val startTime = System.currentTimeMillis()
            val clampedSeq = PreprocessingUtils.clamp01(windowSeq)
            val outputs = ctcRunner.run(clampedSeq)
            val infMs = System.currentTimeMillis() - startTime
            totalInferenceTimeMs += infMs
            inferenceCount += 1
            val logProbs = outputs.logProbs[0] // [T, num_ctc]

            if (debugLogging) {
                // Compute per-frame argmax IDs (including blank) to verify output dynamics
                val T = logProbs.size
                if (T > 0) {
                    val arg = IntArray(T) { t ->
                        var a = 0
                        var m = logProbs[t][0]
                        for (c in 1 until logProbs[t].size) {
                            if (logProbs[t][c] > m) { m = logProbs[t][c]; a = c }
                        }
                        a
                    }
                    // Mode id over the window
                    val counts = HashMap<Int, Int>()
                    for (v in arg) counts[v] = (counts[v] ?: 0) + 1
                    var modeId = arg[0]
                    var modeCnt = 0
                    for ((k, v) in counts) if (v > modeCnt) { modeCnt = v; modeId = k }
                    val head = arg.take(minOf(20, arg.size)).joinToString(",")
                    val tail = arg.takeLast(minOf(10, arg.size)).joinToString(",")
                    val pct = (modeCnt * 100) / T
                    Log.i(TAG, "CTC debug: mode=$modeId (${pct}%), blank=${ctcRunner.meta.blank_id}, head=[$head], tail=[$tail]")
                }
            }

            val tokens = CTCDecoder.greedy(logProbs, ctcRunner.meta.blank_id)

            // Estimate absolute window start frame index
            val windowStartAbs = currentFrame - windowSeq.size + 1
            val newOnes = ctcAggregator.addWindowTokens(windowStartAbs, tokens)

            if (newOnes.isNotEmpty()) {
                for (tk in newOnes) {
                    val glossLabel = labelMapper.getGlossLabel(tk.id)
                    var categoryId = 0
                    var categoryConfidence = 0f
                    outputs.catLogits?.let { cat ->
                        // Average logits over token span and argmax
                        val twoD = cat[0] // [T, num_cat]
                        val startT = maxOf(0, tk.startT)
                        val endT = minOf(twoD.size - 1, tk.endT)
                        if (endT >= startT && twoD.isNotEmpty()) {
                            val numCat = twoD[0].size
                            val avg = FloatArray(numCat) { 0f }
                            var count = 0
                            for (t in startT..endT) {
                                val row = twoD[t]
                                for (c in 0 until numCat) avg[c] += row[c]
                                count += 1
                            }
                            if (count > 0) {
                                for (c in avg.indices) avg[c] /= count.toFloat()
                                var arg = 0
                                var best = avg[0]
                                for (c in 1 until avg.size) if (avg[c] > best) { best = avg[c]; arg = c }
                                categoryId = arg
                                
                                // Calculate category confidence using softmax
                                val maxLogit = avg.maxOrNull() ?: 0f
                                var expSum = 0f
                                for (c in avg.indices) {
                                    expSum += kotlin.math.exp(avg[c] - maxLogit)
                                }
                                categoryConfidence = kotlin.math.exp(avg[categoryId] - maxLogit) / expSum
                            }
                        }
                    }
                    val categoryLabel = labelMapper.getCategoryLabel(categoryId)
                    val recognizedSign = RecognizedSign(
                        glossId = tk.id,
                        glossLabel = glossLabel,
                        categoryId = categoryId,
                        categoryLabel = categoryLabel,
                        confidence = tk.confidence,
                        categoryConfidence = categoryConfidence,
                        timestamp = System.currentTimeMillis()
                    )
                    withContext(Dispatchers.Main) { onSignRecognized(recognizedSign) }
                    Log.i(TAG, "CTC token: ${tk.id} $glossLabel conf=${"%.2f".format(tk.confidence)} frames=[${tk.startT}-${tk.endT}]")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "CTC inference failed", e)
        }
    }

    // Removed legacy single-label classification postprocessing

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
        Log.i(TAG, "Starting pipeline recovery. Reason: $reason")
        
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
                
                Log.i(TAG, "Pipeline recovery completed")
            } catch (e: Exception) {
                Log.e(TAG, "Pipeline recovery failed", e)
            }
        }
    }

    /**
     * Notify pipeline that recording has started.
     */
    fun onRecordingStarted() {
        Log.i(TAG, "Recording started, protecting pipeline...")
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
        Log.i(TAG, "Recording stopped, resuming normal operation...")
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
     * Stop the recognition pipeline.
     */
    fun stop() {
        if (!isRunning.get()) {
            return
        }

        isRunning.set(false)
        
        Log.i(TAG, "Stopping recognition pipeline...")

        // Stop health monitor
        healthMonitorJob?.cancel()

        // Stop camera
        cameraManager.stopCamera()

        // DON'T cancel pipelineScope here - it breaks restart!
        // Scope will be cancelled in release() when truly done
        
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
        
        // Now cancel the scope (activity is being destroyed)
        pipelineScope.cancel()
        
        cameraManager.release()
        mediaPipeProcessor.release()
        ctcRunner.release()
        
        Log.i(TAG, "All resources released")
    }

    /**
     * Get pipeline statistics for debugging.
     */
    fun getStats(): PipelineStats {
        val (cameraProcessed, cameraTotal) = cameraManager.getStats()
        val (mediapipeSuccess, mediapipeFailure) = mediaPipeProcessor.getStats()
        val avgInferenceTime = if (inferenceCount > 0) totalInferenceTimeMs / inferenceCount else 0L
        val temporalStats = temporalRecognizer.getStats()
        
        val timeSinceLastFrame = System.currentTimeMillis() - lastFrameTime.get()
        val timeSinceLastKeypoint = System.currentTimeMillis() - lastKeypointTime.get()
        
        return PipelineStats(
            framesProcessed = cameraProcessed,
            framesTotal = cameraTotal,
            keypointSuccess = mediapipeSuccess,
            keypointFailure = mediapipeFailure,
            avgInferenceTimeMs = avgInferenceTime,
            temporalStats = temporalStats,
            bufferSize = bufferManager.getBufferSize(),
            timeSinceLastFrame = timeSinceLastFrame,
            timeSinceLastKeypoint = timeSinceLastKeypoint,
            isRecording = isRecording.get()
        )
    }

    /**
     * Pause recognition (stop processing but keep resources).
     */
    fun pause() {
        isPaused.set(true)
        Log.i(TAG, "Pipeline paused")
    }

    /**
     * Resume recognition after pause.
     */
    fun resume() {
        isPaused.set(false)
        lastFrameTime.set(System.currentTimeMillis())
        lastKeypointTime.set(System.currentTimeMillis())
        Log.i(TAG, "Pipeline resumed")
    }

    /**
     * Switch active CTC model at runtime.
     * Blocks frame processing during swap, rebuilds buffer/aggregator from metadata hints,
     * warms up the new model, and then resumes.
     */
    fun switchModel(ptPath: String, metadataPath: String, preferGpu: Boolean = false) {
        pipelineScope.launch(Dispatchers.IO) {
            try {
                Log.i(TAG, "Switching CTC model to path=$ptPath meta=$metadataPath ...")
                // Pause processing
                isPaused.set(true)

                // Release previous runner
                try { ctcRunner.release() } catch (_: Throwable) {}

                // Instantiate new runner (PyTorch only)
                if (!(ptPath.endsWith(".pt") || ptPath.endsWith(".ptl"))) {
                    throw IllegalArgumentException("PyTorch model path must end with .pt or .ptl")
                }
                ctcRunner = PyTorchModelRunner(
                    context = context,
                    assetModelPath = ptPath,
                    metadataPath = metadataPath
                )

                // Update window/stride from new metadata
                ctcWindowSize = ctcRunner.meta.window_size_hint
                ctcStride = ctcRunner.meta.stride_hint

                // Rebuild buffer and aggregator
                bufferManager.clear()
                bufferManager = SequenceBufferManager(windowSize = ctcWindowSize, maxGap = 5)
                ctcAggregator.clear()

                // Reset perf stats
                totalInferenceTimeMs = 0L
                inferenceCount = 0L

                // Warm-up run with a zero window to allocate tensors
                try {
                    val dummy = Array(ctcWindowSize) { FloatArray(178) { 0f } }
                    ctcRunner.run(dummy)
                } catch (_: Throwable) {}

                // Resume
                lastFrameTime.set(System.currentTimeMillis())
                lastKeypointTime.set(System.currentTimeMillis())
                isPaused.set(false)

                Log.i(TAG, "Model switch completed")
            } catch (t: Throwable) {
                Log.e(TAG, "Model switch failed", t)
                isPaused.set(false)
            }
        }
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
}

/**
 * Represents a recognized sign with metadata.
 */
data class RecognizedSign(
    val glossId: Int,
    val glossLabel: String,
    val categoryId: Int,
    val categoryLabel: String,
    val confidence: Float,  // Gloss confidence
    val categoryConfidence: Float,  // Category confidence
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
    val bufferSize: Int,
    val timeSinceLastFrame: Long,
    val timeSinceLastKeypoint: Long,
    val isRecording: Boolean
) {
    override fun toString(): String {
        // Format stats in 2 columns for compact display
        val col1Line1 = "Frames: $framesProcessed/$framesTotal"
        val col2Line1 = "Inference: ${avgInferenceTimeMs}ms"
        val col1Line2 = "Keypoints: $keypointSuccess/$keypointFailure"
        val col2Line2 = "Buffer: $bufferSize"
        val col1Line3 = "Temporal: $temporalStats"
        val col2Line3 = "Recording: ${if (isRecording) "Yes" else "No"}"
        val col1Line4 = "Last frame: ${timeSinceLastFrame}ms"
        val col2Line4 = "Last KP: ${timeSinceLastKeypoint}ms"
        
        return """
            ${col1Line1.padEnd(28)} ${col2Line1}
            ${col1Line2.padEnd(28)} ${col2Line2}
            ${col1Line3.padEnd(28)} ${col2Line3}
            ${col1Line4.padEnd(28)} ${col2Line4}
        """.trimIndent()
    }
}


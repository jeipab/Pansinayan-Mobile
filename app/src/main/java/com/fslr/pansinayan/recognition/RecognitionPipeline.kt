package com.fslr.pansinayan.recognition

import android.content.Context
import android.util.Log
import androidx.camera.core.ImageProxy
import androidx.lifecycle.LifecycleOwner
import com.fslr.pansinayan.camera.CameraManager
import com.fslr.pansinayan.inference.CTCDecoder
import com.fslr.pansinayan.inference.CtcOutputs
import com.fslr.pansinayan.inference.InferenceManager
import com.fslr.pansinayan.inference.PreprocessingUtils
import com.fslr.pansinayan.mediapipe.MediaPipeProcessor
import com.fslr.pansinayan.utils.LabelMapper
import kotlinx.coroutines.*
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import com.fslr.pansinayan.network.NetworkClient
import com.fslr.pansinayan.utils.ModeManager

/**
 * Orchestrates the complete recognition pipeline with activity-driven inference.
 * 
 * Pipeline flow:
 * Camera → MediaPipe → Activity Detection → Sign Boundary Detection → Adaptive Buffer
 * → Inference Trigger → CTC Model → Aggregator → UI Callback
 * 
 * Responsibilities:
 * - Coordinate all components
 * - Manage threading and async operations
 * - Handle lifecycle (start/stop/pause)
 * - Activity-driven inference scheduling
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
        private const val HEALTH_CHECK_INTERVAL_MS = 2000L  // Check pipeline health every 2 seconds
        private const val FRAME_TIMEOUT_MS = 3000L  // Consider pipeline frozen after 3 seconds
    }

    // Core components
    private lateinit var cameraManager: CameraManager
    private lateinit var mediaPipeProcessor: MediaPipeProcessor
    private lateinit var bufferManager: AdaptiveBufferManager
    private lateinit var activityDetector: ActivityDetector
    private lateinit var boundaryDetector: SignBoundaryDetector
    private lateinit var inferenceTrigger: InferenceTrigger
    private lateinit var inferenceManager: InferenceManager
    private lateinit var ctcAggregator: CtcAggregator
    private var ctcWindowSize: Int = 150
    private lateinit var labelMapper: LabelMapper
    private var totalInferenceTimeMs: Long = 0L
    private var inferenceCount: Long = 0L
    private var debugLogging: Boolean = false
    private var lastProcessedSignEndFrame: Int = -1  // Track last processed sign to prevent duplicates

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
    private lateinit var modeManager: ModeManager

    /**
     * Initialize all components.
     * Call this before start().
     */
    fun initialize() {
        try {
            Log.i(TAG, "Initializing recognition pipeline...")

            cameraManager = CameraManager(context, lifecycleOwner, previewView, targetFps = 30)
            mediaPipeProcessor = MediaPipeProcessor(context, this)

            modeManager = ModeManager(context)
            NetworkClient.initialize(context)

            // Initialize InferenceManager (handles both online/offline with lazy loading)
            inferenceManager = InferenceManager(context, modeManager)
            
            // Get metadata (blocking call during initialization is acceptable)
            val metadata = runBlocking {
                inferenceManager.getMetadata()
            }
            if (metadata != null) {
                ctcWindowSize = metadata.window_size_hint
                Log.i(TAG, "InferenceManager initialized, window size: $ctcWindowSize")
            } else {
                Log.w(TAG, "Could not get metadata, using default window size")
                ctcWindowSize = 150
            }
            
            // Initialize activity-driven components
            // Uses improved defaults: separate arm/hand thresholds, percentile filtering, jitter filtering
            activityDetector = ActivityDetector()
            boundaryDetector = SignBoundaryDetector(
                minSignDurationMs = 500L,
                maxSignDurationMs = 5000L,
                holdPeriodMs = 300L
            )
            inferenceTrigger = InferenceTrigger(
                cooldownMs = 500L,
                maxActiveDurationMs = 5000L
            )
            bufferManager = AdaptiveBufferManager(
                maxBufferSize = 300,
                windowSize = ctcWindowSize,
                maxGap = 5
            )
            ctcAggregator = CtcAggregator(iouThreshold = 0.5f, stabilityThreshold = 1)
            labelMapper = LabelMapper(context)

            Log.i(TAG, "All components initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize pipeline", e)
            throw e
        }
    }


    /**
     * Switch between online and offline mode at runtime (non-blocking).
     */
    fun switchMode(newMode: ModeManager.InferenceMode) {
        pipelineScope.launch(Dispatchers.IO) {
            try {
                Log.i(TAG, "Switching to $newMode mode...")
                isPaused.set(true)

                // Use InferenceManager for mode switching (non-blocking, lazy initialization)
                val result = inferenceManager.switchMode(newMode)
                result.getOrThrow()

                // Update mode manager (persist the mode)
                modeManager.setMode(newMode)

                // Reset pipeline state - CRITICAL: Reset frame counter FIRST before clearing buffer
                // This ensures frame indices are consistent between buffer and boundary detector
                frameCounter.set(0)
                Log.d(TAG, "Pipeline frame counter reset to 0")
                
                bufferManager.clear()
                activityDetector.reset()
                boundaryDetector.resetToIdle()
                inferenceTrigger.reset()
                ctcAggregator.clear()
                totalInferenceTimeMs = 0L
                inferenceCount = 0L
                lastProcessedSignEndFrame = -1

                lastFrameTime.set(System.currentTimeMillis())
                lastKeypointTime.set(System.currentTimeMillis())
                isPaused.set(false)
                
                Log.i(TAG, "All pipeline components reset - ready for $newMode mode")

                Log.i(TAG, "Mode switch completed: $newMode")
                
                // Verify mode was set correctly
                val actualMode = modeManager.getCurrentMode()
                if (actualMode != newMode) {
                    Log.e(TAG, "Mode mismatch! Expected: $newMode, Actual: $actualMode")
                } else {
                    Log.i(TAG, "Mode verified: $actualMode")
                }

                // Notify on main thread
                withContext(Dispatchers.Main) {
                    android.widget.Toast.makeText(
                        context,
                        "Switched to ${newMode.name} mode",
                        android.widget.Toast.LENGTH_SHORT
                    ).show()
                    
                    // Log mode for debugging
                    Log.i(TAG, "Mode switch UI notification - Mode: $newMode")
                }

            } catch (t: Throwable) {
                Log.e(TAG, "Mode switch failed", t)
                isPaused.set(false)

                withContext(Dispatchers.Main) {
                    android.widget.Toast.makeText(
                        context,
                        "Failed to switch mode: ${t.message}",
                        android.widget.Toast.LENGTH_LONG
                    ).show()
                }
            }
        }
    }

    /**
     * Get current inference mode.
     */
    fun getCurrentMode(): ModeManager.InferenceMode = modeManager.getCurrentMode()

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
        // Reset frame counter periodically to prevent overflow
        // Use modulo to keep it manageable (reset every 1M frames)
        val currentFrame = frameCounter.get()
        if (currentFrame >= 1_000_000) {
            frameCounter.set(0)
            Log.d(TAG, "Frame counter reset to prevent overflow")
        } else if (currentFrame == 0) {
            // Only reset to 0 on first start, not on resume
            frameCounter.set(0)
        }
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
                
                // Add to buffer - ensure all keypoints are captured
                bufferManager.addFrame(keypoints)
                
                // Process activity detection (returns state and motion together to avoid race conditions)
                // Log keypoint validity for debugging
                if (currentFrame % 60 == 0 && keypoints != null) {
                    val nonZeroCount = keypoints.count { it != 0f }
                    Log.d(TAG, "Frame $currentFrame: keypoints valid=$nonZeroCount/178, buffer size=${bufferManager.getBufferSize()}")
                }
                
                val (activityState, currentMotion) = activityDetector.processFrame(keypoints)
                
                // Enhanced logging for ONLINE mode debugging
                val isOnline = modeManager.getCurrentMode() == ModeManager.InferenceMode.ONLINE
                if (isOnline && currentFrame % 30 == 0) {
                    Log.d(TAG, "Frame $currentFrame: activity=$activityState, motion=$currentMotion, " +
                            "boundary=${boundaryDetector.getState()}, buffer=${bufferManager.getBufferSize()}")
                }
                
                // Process sign boundary detection
                val boundaryEvent = boundaryDetector.processActivity(activityState, currentMotion, currentFrame)
                
                // Check for inference trigger
                boundaryEvent?.let { event ->
                    val triggerReason = inferenceTrigger.shouldTrigger(event, currentFrame)
                    triggerReason?.let { reason ->
                        triggerInference(reason, currentFrame)
                    }
                }
                
                // Check for long active period (fallback)
                if (activityDetector.isActive()) {
                    val longActiveTrigger = inferenceTrigger.checkLongActivePeriod(true, currentFrame)
                    longActiveTrigger?.let { reason ->
                        triggerInference(reason, currentFrame)
                    }
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
     * Trigger inference based on sign boundary detection.
     * 
     * Flow:
     * 1. IDLE - no signing
     * 2. ACTIVE - user is signing, keypoints are being captured
     * 3. SIGN_COMPLETE - sign ended, NOW make inference call (activity may be IDLE by now, which is expected)
     * 4. Reset and return to IDLE
     */
    private suspend fun triggerInference(reason: TriggerReason, currentFrame: Int) {
        try {
            val boundaryState = boundaryDetector.getState()
            
            // For SignComplete, we expect boundary state to be SIGN_COMPLETE
            // Activity state may already be IDLE (sign ended), which is correct
            if (reason is TriggerReason.SignComplete) {
                if (boundaryState != SignBoundaryDetector.SignState.SIGN_COMPLETE) {
                    Log.w(TAG, "Inference triggered with SignComplete but boundary state is $boundaryState - skipping")
                    return
                }
                
                // Prevent duplicate inference for the same sign
                val signEndFrame = boundaryDetector.getSignEndFrame()
                if (signEndFrame != null && signEndFrame == lastProcessedSignEndFrame) {
                    Log.d(TAG, "Sign end frame $signEndFrame already processed - skipping duplicate inference")
                    return
                }
                
                if (signEndFrame != null) {
                    lastProcessedSignEndFrame = signEndFrame
                }
                
                Log.i(TAG, "Sign completed - triggering inference (boundary: $boundaryState, endFrame: $signEndFrame)")
            }
            
            // For LongActivePeriod, ensure we're actually active
            if (reason is TriggerReason.LongActivePeriod) {
                val activityState = activityDetector.getState()
                if (activityState == ActivityDetector.ActivityState.IDLE) {
                    Log.w(TAG, "LongActivePeriod trigger while IDLE - skipping")
                    return
                }
            }

            val windowSeq = when (reason) {
                is TriggerReason.SignComplete -> {
                    // Get sign boundaries from boundary detector
                    val signStartFrame = boundaryDetector.getSignStartFrame() ?: return
                    val signEndFrame = boundaryDetector.getSignEndFrame() ?: return
                    Log.d(TAG, "Extracting sign window: frames [$signStartFrame-$signEndFrame]")
                    bufferManager.extractSignWindow(signStartFrame, signEndFrame)
                }
                is TriggerReason.LongActivePeriod -> {
                    Log.d(TAG, "Extracting long active period window: frames [${reason.signStartFrame}-${reason.signEndFrame}]")
                    bufferManager.extractSignWindow(reason.signStartFrame, reason.signEndFrame)
                }
            } ?: run {
                Log.w(TAG, "Failed to extract window for inference")
                return
            }

            // Log keypoint capture info
            val validKeypointCount = windowSeq.count { frame ->
                frame.any { v -> kotlin.math.abs(v) > 0.001f }
            }
            Log.d(TAG, "Inference window: ${windowSeq.size} frames, $validKeypointCount with valid keypoints")

            // Check for mostly zero frames (indicates missing data)
            val zeroFrameCount = windowSeq.count { frame -> 
                frame.all { v -> kotlin.math.abs(v) < 0.001f }
            }
            val missingRatio = if (windowSeq.isNotEmpty()) zeroFrameCount.toFloat() / windowSeq.size else 1f
            if (missingRatio > 0.3f) {
                Log.d(TAG, "Skipping inference due to missing ratio: $missingRatio")
                return
            }

            val startTime = System.currentTimeMillis()
            val clampedSeq = PreprocessingUtils.clamp01(windowSeq)
            
            // Log mode for debugging
            val currentMode = modeManager.getCurrentMode()
            Log.i(TAG, "Running inference - Mode: ${currentMode.name}")
            
            // Use InferenceManager (handles async, fallback automatically)
            val outputs = inferenceManager.runInference(clampedSeq)
            val infMs = System.currentTimeMillis() - startTime
            totalInferenceTimeMs += infMs
            inferenceCount += 1
            val logProbs = outputs.logProbs[0] // [T, num_ctc]

            // Get metadata for decoding
            val metadata = inferenceManager.getMetadata() ?: run {
                Log.e(TAG, "Cannot get metadata for decoding")
                return
            }

            if (debugLogging) {
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
                    val counts = HashMap<Int, Int>()
                    for (v in arg) counts[v] = (counts[v] ?: 0) + 1
                    var modeId = arg[0]
                    var modeCnt = 0
                    for ((k, v) in counts) if (v > modeCnt) { modeCnt = v; modeId = k }
                    val head = arg.take(minOf(20, arg.size)).joinToString(",")
                    val tail = arg.takeLast(minOf(10, arg.size)).joinToString(",")
                    val pct = (modeCnt * 100) / T
                    Log.i(TAG, "CTC debug: mode=$modeId (${pct}%), blank=${metadata.blank_id}, head=[$head], tail=[$tail]")
                }
            }

            val tokens = CTCDecoder.greedy(logProbs, metadata.blank_id)
            val CONFIDENCE_THRESHOLD = 0f  // Set to 0 to allow all model outputs
            val filteredTokens = tokens.filter { it.confidence >= CONFIDENCE_THRESHOLD }

            // Log all decoded tokens for debugging
            if (tokens.isNotEmpty()) {
                Log.d(TAG, "Decoded ${tokens.size} tokens: ${tokens.map { "${it.id}(${String.format("%.3f", it.confidence)})" }.joinToString(", ")}")
            } else {
                Log.d(TAG, "No tokens decoded from model output")
            }

            // Estimate window start frame
            val windowStartAbs = currentFrame - windowSeq.size + 1
            val newOnes = ctcAggregator.addWindowTokens(windowStartAbs, filteredTokens)
            
            Log.d(TAG, "After aggregation: ${newOnes.size} new tokens, activity state=${activityDetector.getState()}")

            if (newOnes.isNotEmpty()) {
                for (tk in newOnes) {
                    val glossLabel = labelMapper.getGlossLabel(tk.id)
                    var categoryId = 0
                    var categoryConfidence = 0f
                    outputs.catLogits?.let { cat ->
                        val twoD = cat[0] // [T, num_cat]
                        val startT = maxOf(0, tk.startT - windowStartAbs)
                        val endT = minOf(twoD.size - 1, tk.endT - windowStartAbs)
                        
                        if (endT >= startT && startT >= 0 && endT < twoD.size) {
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
                    withContext(Dispatchers.Main) { 
                        onSignRecognized(recognizedSign) 
                    }
                    val catConfStr = "%.2f".format(categoryConfidence)
                    val glossConfStr = "%.2f".format(tk.confidence)
                    Log.i(TAG, "CTC token: ${tk.id} $glossLabel (cat: $categoryId $categoryLabel $catConfStr) conf=$glossConfStr frames=[${tk.startT}-${tk.endT}]")
                }
            } else {
                Log.d(TAG, "No new tokens after aggregation - inference completed with no results")
            }
            
            // Reset boundary detector AFTER inference completes
            // This ensures we're ready for the next sign
            boundaryDetector.reset()
            // Clear the processed sign tracking for the next sign
            if (reason is TriggerReason.SignComplete) {
                lastProcessedSignEndFrame = -1
            }
            Log.d(TAG, "Boundary detector reset - ready for next sign")
        } catch (e: Exception) {
            val currentMode = modeManager.getCurrentMode()
            Log.e(TAG, "=== CTC INFERENCE FAILED ===")
            Log.e(TAG, "Mode: ${currentMode.name}")
            Log.e(TAG, "Is online: ${currentMode == ModeManager.InferenceMode.ONLINE}")
            Log.e(TAG, "Exception type: ${e.javaClass.simpleName}")
            Log.e(TAG, "Exception message: ${e.message}")
            
            // Log specific error details for online mode
            if (currentMode == ModeManager.InferenceMode.ONLINE) {
                when (e) {
                    is java.net.UnknownHostException -> {
                        Log.e(TAG, "Server URL unreachable - check Cloudflare tunnel")
                        Log.e(TAG, "Server URL: ${NetworkClient.getServerUrl()}")
                    }
                    is java.net.SocketTimeoutException -> {
                        Log.e(TAG, "Server request timeout - check network connection")
                    }
                    is retrofit2.HttpException -> {
                        var errorBody: String? = null
                        try {
                            errorBody = e.response()?.errorBody()?.string()
                        } catch (ex: Exception) {
                            Log.e(TAG, "Failed to read error body: ${ex.message}")
                        }
                        Log.e(TAG, "Server HTTP error: ${e.code()} ${e.message()}")
                        Log.e(TAG, "Error body: $errorBody")
                        Log.e(TAG, "Error response headers: ${e.response()?.headers()}")
                    }
                    is java.io.IOException -> {
                        Log.e(TAG, "Network IO error: ${e.message}")
                    }
                    else -> {
                        Log.e(TAG, "Unexpected error in online mode", e)
                    }
                }
            }
            
            e.printStackTrace()
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
        
        // Clear buffer and reset detectors
        bufferManager.clear()
        activityDetector.reset()
        boundaryDetector.resetToIdle()
        inferenceTrigger.reset()
        lastProcessedSignEndFrame = -1

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
        if (::inferenceManager.isInitialized) {
            inferenceManager.release()
        }
        
        Log.i(TAG, "All resources released")
    }

    /**
     * Get pipeline statistics for debugging.
     */
    fun getStats(): PipelineStats {
        val (cameraProcessed, cameraTotal) = cameraManager.getStats()
        val (mediapipeSuccess, mediapipeFailure) = mediaPipeProcessor.getStats()
        val avgInferenceTime = if (inferenceCount > 0) totalInferenceTimeMs / inferenceCount else 0L
        
        val timeSinceLastFrame = System.currentTimeMillis() - lastFrameTime.get()
        val timeSinceLastKeypoint = System.currentTimeMillis() - lastKeypointTime.get()
        
        val activityState = activityDetector.getState().name
        val boundaryState = boundaryDetector.getState().name
        val currentMotion = activityDetector.getCurrentMotion()
        val activityInfo = "$activityState / $boundaryState (motion: ${"%.4f".format(currentMotion)})"
        
        return PipelineStats(
            framesProcessed = cameraProcessed,
            framesTotal = cameraTotal,
            keypointSuccess = mediapipeSuccess,
            keypointFailure = mediapipeFailure,
            avgInferenceTimeMs = avgInferenceTime,
            temporalStats = activityInfo,
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

                // Note: switchModel only works for offline mode
                // For online mode, model type is determined by server
                if (modeManager.getCurrentMode() != ModeManager.InferenceMode.OFFLINE) {
                    Log.w(TAG, "Model switching only available in offline mode")
                    isPaused.set(false)
                    return@launch
                }

                // Validate model path
                if (!(ptPath.endsWith(".pt") || ptPath.endsWith(".ptl"))) {
                    throw IllegalArgumentException("PyTorch model path must end with .pt or .ptl")
                }

                // For now, model switching requires app restart
                // TODO: Enhance InferenceManager to support model switching
                Log.w(TAG, "Model switching via switchModel() not fully supported with InferenceManager")
                Log.w(TAG, "Please restart the app to change models")
                
                // Get metadata from current InferenceManager
                val metadata = inferenceManager.getMetadata()
                if (metadata != null) {
                    ctcWindowSize = metadata.window_size_hint
                }

                // Rebuild buffer and aggregator
                bufferManager.clear()
                bufferManager = AdaptiveBufferManager(
                    maxBufferSize = 300,
                    windowSize = ctcWindowSize,
                    maxGap = 5
                )
                activityDetector.reset()
                boundaryDetector.resetToIdle()
                inferenceTrigger.reset()
                ctcAggregator.clear()

                // Reset perf stats
                totalInferenceTimeMs = 0L
                inferenceCount = 0L
                lastProcessedSignEndFrame = -1

                // Warm-up run with a zero window to allocate tensors
                try {
                    val dummy = Array(ctcWindowSize) { FloatArray(178) { 0f } }
                    inferenceManager.runInference(dummy)
                } catch (_: Throwable) {
                    // Ignore warm-up errors
                }

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
        // Clean, organized stats display
        return buildString {
            appendLine("📊 Pipeline Stats")
            appendLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            appendLine("📹 Frames: $framesProcessed / $framesTotal")
            appendLine("🎯 Keypoints: $keypointSuccess ✓ / $keypointFailure ✗")
            appendLine("💾 Buffer: $bufferSize frames")
            appendLine("⚡ Inference: ${if (avgInferenceTimeMs > 0) "${avgInferenceTimeMs}ms" else "N/A"}")
            appendLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            appendLine("🔄 State: $temporalStats")
            appendLine("⏱️  Last Frame: ${timeSinceLastFrame}ms ago")
            appendLine("⏱️  Last Keypoint: ${timeSinceLastKeypoint}ms ago")
            appendLine("📹 Recording: ${if (isRecording) "● ON" else "○ OFF"}")
        }
    }
}


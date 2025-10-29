package com.fslr.pansinayan.camera

import android.content.Context
import android.util.Log
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import kotlinx.coroutines.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * Manages camera operations using CameraX API with automatic recovery.
 * 
 * Responsibilities:
 * - Initialize and manage CameraX lifecycle
 * - Provide live camera preview
 * - Sample frames at target FPS for processing
 * - Deliver ImageProxy frames directly to MediaPipe
 * - Automatically recover from frame freezes
 * 
 * Usage:
 *   val cameraManager = CameraManager(context, lifecycleOwner, previewView, targetFps = 15)
 *   cameraManager.startCamera { imageProxy ->
 *       // Process imageProxy with MediaPipe
 *       // Remember to close() imageProxy when done!
 *   }
 */
class CameraManager(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: PreviewView,
    private val targetFps: Int = 15  // Target frame processing rate
) {
    companion object {
        private const val TAG = "CameraManager"
        private const val FRAME_TIMEOUT_MS = 3000L  // 3 seconds without frames triggers recovery
        private const val RECOVERY_DELAY_MS = 500L
    }

    private var cameraProvider: ProcessCameraProvider? = null
    private var cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var imageAnalyzer: ImageAnalysis? = null
    private var frameCallback: ((ImageProxy) -> Unit)? = null
    
    // Camera selection
    private var currentCameraSelector: CameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA
    
    // Frame rate control
    private var lastProcessedTime = 0L
    private val frameIntervalMs = 1000L / targetFps
    
    // Statistics
    private var totalFrames = 0
    private var processedFrames = 0
    
    // Frame timeout detection
    private val lastFrameTime = AtomicLong(0)
    private val isRecovering = AtomicBoolean(false)
    private val monitorScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private var monitorJob: Job? = null
    
    // State tracking
    private var isStarted = false

    /**
     * Start camera with preview and frame analysis.
     * 
     * @param onFrameReady Callback invoked with each sampled frame (ImageProxy)
     */
    fun startCamera(onFrameReady: (ImageProxy) -> Unit) {
        frameCallback = onFrameReady
        isStarted = true
        
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()
                bindCameraUseCases()
                startFrameMonitor()
                Log.i(TAG, "Camera started successfully at $targetFps FPS target")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start camera", e)
            }
        }, ContextCompat.getMainExecutor(context))
    }

    /**
     * Bind camera use cases: preview and image analysis.
     */
    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: return

        // Preview use case - shows live camera feed
        val preview = Preview.Builder()
            .build()
            .also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

        // Image analysis use case - processes frames for recognition
        imageAnalyzer = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            .also {
                it.setAnalyzer(cameraExecutor, FrameAnalyzer())
            }

        try {
            // Unbind all use cases before rebinding
            cameraProvider.unbindAll()

            // Bind use cases to camera with current camera selector
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                currentCameraSelector,
                preview,
                imageAnalyzer
            )

            val cameraName = if (isFrontCamera()) "front" else "back"
            Log.i(TAG, "Camera use cases bound successfully ($cameraName camera)")
            
            // Update last frame time
            lastFrameTime.set(System.currentTimeMillis())
            
        } catch (e: Exception) {
            Log.e(TAG, "Use case binding failed", e)
        }
    }

    /**
     * Frame analyzer that samples frames at target FPS.
     */
    private inner class FrameAnalyzer : ImageAnalysis.Analyzer {
        override fun analyze(image: ImageProxy) {
            totalFrames++
            lastFrameTime.set(System.currentTimeMillis())
            
            val currentTime = System.currentTimeMillis()

            // Frame rate throttling - only process frames at target FPS
            if (currentTime - lastProcessedTime >= frameIntervalMs) {
                try {
                    // Pass ImageProxy directly to MediaPipe (no conversion needed!)
                    frameCallback?.invoke(image)

                    processedFrames++
                    lastProcessedTime = currentTime

                    // Log FPS periodically
                    if (processedFrames % 100 == 0) {
                        Log.d(TAG, "Processed $processedFrames frames (total: $totalFrames)")
                    }

                } catch (e: Exception) {
                    Log.e(TAG, "Frame analysis failed", e)
                    image.close()
                }
                // Note: ImageProxy will be closed by MediaPipeProcessor after processing
            } else {
                // Skip this frame to maintain target FPS
                image.close()
            }
        }
    }

    /**
     * Start monitoring for frame timeouts.
     */
    private fun startFrameMonitor() {
        monitorJob?.cancel()
        monitorJob = monitorScope.launch {
            while (isActive && isStarted) {
                delay(1000) // Check every second
                
                val timeSinceLastFrame = System.currentTimeMillis() - lastFrameTime.get()
                
                if (timeSinceLastFrame > FRAME_TIMEOUT_MS && !isRecovering.get()) {
                    Log.w(TAG, "Frame timeout detected! Time since last frame: ${timeSinceLastFrame}ms")
                    recoverCamera()
                }
            }
        }
    }

    /**
     * Recover camera when frames stop coming.
     */
    private fun recoverCamera() {
        if (isRecovering.getAndSet(true)) {
            Log.d(TAG, "Already recovering, skipping...")
            return
        }

        Log.i(TAG, "Starting camera recovery...")
        
        monitorScope.launch {
            try {
                // Step 1: Clear current analyzer
                withContext(Dispatchers.Main) {
                    imageAnalyzer?.clearAnalyzer()
                }
                
                delay(RECOVERY_DELAY_MS)
                
                // Step 2: Rebind camera use cases
                withContext(Dispatchers.Main) {
                    bindCameraUseCases()
                }
                
                delay(RECOVERY_DELAY_MS)
                
                Log.i(TAG, "Camera recovery completed")
                
            } catch (e: Exception) {
                Log.e(TAG, "Camera recovery failed", e)
            } finally {
                isRecovering.set(false)
                lastFrameTime.set(System.currentTimeMillis())
            }
        }
    }

    /**
     * Force restart camera (useful when recording starts/stops).
     */
    fun restartCamera() {
        if (!isStarted) return
        
        Log.i(TAG, "Force restarting camera...")
        monitorScope.launch {
            try {
                withContext(Dispatchers.Main) {
                    bindCameraUseCases()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to restart camera", e)
            }
        }
    }

    /**
     * Stop camera (but keep executor alive for restart).
     */
    fun stopCamera() {
        isStarted = false
        monitorJob?.cancel()
        cameraProvider?.unbindAll()
        // DON'T shutdown executor here - it breaks restart!
        // Executor will be shutdown in release() when truly done
        Log.i(TAG, "Camera stopped. Stats: $processedFrames processed / $totalFrames total frames")
    }

    /**
     * Get current FPS statistics.
     */
    fun getStats(): Pair<Int, Int> {
        return Pair(processedFrames, totalFrames)
    }

    /**
     * Switch between front and back camera.
     * Automatically rebinds camera use cases.
     */
    fun switchCamera() {
        currentCameraSelector = if (isFrontCamera()) {
            CameraSelector.DEFAULT_BACK_CAMERA
        } else {
            CameraSelector.DEFAULT_FRONT_CAMERA
        }
        
        val cameraName = if (isFrontCamera()) "front" else "back"
        Log.i(TAG, "Switching to $cameraName camera")
        
        // Rebind camera with new selector
        bindCameraUseCases()
    }

    /**
     * Check if currently using front camera.
     */
    fun isFrontCamera(): Boolean {
        return currentCameraSelector == CameraSelector.DEFAULT_FRONT_CAMERA
    }
    
    /**
     * Release all resources.
     */
    fun release() {
        stopCamera()
        monitorScope.cancel()
        // Now shutdown executor (activity is being destroyed)
        cameraExecutor.shutdown()
        Log.i(TAG, "Camera executor shutdown")
    }
}


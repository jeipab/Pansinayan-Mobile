package com.fslr.pansinayan.camera

import android.content.Context
import android.util.Log
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Manages camera operations using CameraX API.
 * 
 * Responsibilities:
 * - Initialize and manage CameraX lifecycle
 * - Provide live camera preview
 * - Sample frames at target FPS for processing
 * - Deliver ImageProxy frames directly to MediaPipe (no conversion needed)
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

    /**
     * Start camera with preview and frame analysis.
     * 
     * @param onFrameReady Callback invoked with each sampled frame (ImageProxy)
     */
    fun startCamera(onFrameReady: (ImageProxy) -> Unit) {
        frameCallback = onFrameReady
        
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()
                bindCameraUseCases()
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
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
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
     * Stop camera and release resources.
     */
    fun stopCamera() {
        cameraProvider?.unbindAll()
        cameraExecutor.shutdown()
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
}


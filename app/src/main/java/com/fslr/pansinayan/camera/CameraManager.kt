package com.fslr.pansinayan.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import android.util.Log
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Manages camera operations using CameraX API.
 * 
 * Responsibilities:
 * - Initialize and manage CameraX lifecycle
 * - Provide live camera preview
 * - Sample frames at target FPS for processing
 * - Convert camera frames to Bitmap format
 * - Deliver frames to callback for further processing
 * 
 * Usage:
 *   val cameraManager = CameraManager(context, lifecycleOwner, previewView, targetFps = 15)
 *   cameraManager.startCamera { bitmap ->
 *       // Process bitmap (extract keypoints, run inference, etc.)
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
    private var frameCallback: ((Bitmap) -> Unit)? = null
    
    // Frame rate control
    private var lastProcessedTime = 0L
    private val frameIntervalMs = 1000L / targetFps
    
    // Statistics
    private var totalFrames = 0
    private var processedFrames = 0

    /**
     * Start camera with preview and frame analysis.
     * 
     * @param onFrameReady Callback invoked with each sampled frame
     */
    fun startCamera(onFrameReady: (Bitmap) -> Unit) {
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
            .setTargetResolution(android.util.Size(640, 480))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also {
                it.setAnalyzer(cameraExecutor, FrameAnalyzer())
            }

        // Use front camera for sign language (user faces camera)
        val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

        try {
            // Unbind all use cases before rebinding
            cameraProvider.unbindAll()

            // Bind use cases to camera
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                cameraSelector,
                preview,
                imageAnalyzer
            )

            Log.i(TAG, "Camera use cases bound successfully")
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
                    // Convert ImageProxy to Bitmap
                    val bitmap = imageProxyToBitmap(image)

                    // Invoke callback with bitmap
                    frameCallback?.invoke(bitmap)

                    processedFrames++
                    lastProcessedTime = currentTime

                    // Log FPS periodically
                    if (processedFrames % 100 == 0) {
                        Log.d(TAG, "Processed $processedFrames frames (total: $totalFrames)")
                    }

                } catch (e: Exception) {
                    Log.e(TAG, "Frame analysis failed", e)
                } finally {
                    image.close()
                }
            } else {
                // Skip this frame to maintain target FPS
                image.close()
            }
        }
    }

    /**
     * Convert ImageProxy (YUV format) to Bitmap (RGB format).
     * 
     * Note: This is a simplified conversion. For better performance,
     * consider using RenderScript or native code for YUV→RGB conversion.
     */
    private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        // U and V are swapped for NV21 format
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 100, out)
        val imageBytes = out.toByteArray()

        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
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
}


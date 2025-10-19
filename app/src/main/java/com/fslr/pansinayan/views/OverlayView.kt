package com.fslr.pansinayan.views

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.util.Log
import android.view.View

/**
 * Custom view for overlaying skeleton visualization on camera preview.
 * 
 * Draws MediaPipe keypoints and connections as a skeleton overlay.
 * Keypoints are expected to be in normalized coordinates (0.0 - 1.0).
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    companion object {
        private const val TAG = "OverlayView"
    }

    private var keypoints: FloatArray? = null
    private var imageWidth = 1
    private var imageHeight = 1
    private var scaleFactor = 1f
    private var debugMode = false
    private var isFrontCamera = true
    private var lastUpdateTime = 0L
    private var frameCount = 0
    
    // Paint for drawing keypoints
    private val keypointPaint = Paint().apply {
        color = Color.rgb(0, 255, 0)
        strokeWidth = 10f
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    // Paint for drawing skeleton connections
    private val linePaint = Paint().apply {
        color = Color.rgb(0, 255, 0)
        strokeWidth = 6f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }
    
    // Paint for hands (different color)
    private val handPaint = Paint().apply {
        color = Color.rgb(255, 255, 0)
        strokeWidth = 10f
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    // Paint for hand connections
    private val handLinePaint = Paint().apply {
        color = Color.rgb(255, 255, 0)
        strokeWidth = 5f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }
    
    // Paint for debug text
    private val debugTextPaint = Paint().apply {
        color = Color.WHITE
        textSize = 24f
        isAntiAlias = true
        style = Paint.Style.FILL
    }
    
    // Paint for debug background
    private val debugBgPaint = Paint().apply {
        color = Color.argb(180, 0, 0, 0)
        style = Paint.Style.FILL
    }

    fun setKeypoints(keypoints: FloatArray?, imageWidth: Int, imageHeight: Int) {
        this.keypoints = keypoints
        this.imageWidth = imageWidth
        this.imageHeight = imageHeight
        
        scaleFactor = kotlin.math.max(width * 1f / imageWidth, height * 1f / imageHeight)
        
        frameCount++
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastUpdateTime > 2000) {
            val validPoints = keypoints?.let { countValidPoints(it) } ?: 0
            Log.d(TAG, "Overlay: $frameCount frames, $validPoints keypoints")
            Log.d(TAG, "Dimensions: overlay=${width}x${height}, image=${imageWidth}x${imageHeight}, scale=%.2f".format(scaleFactor))
            lastUpdateTime = currentTime
            frameCount = 0
        }

        invalidate()
    }
    
    /**
     * Enable or disable debug mode.
     */
    fun setDebugMode(enabled: Boolean) {
        debugMode = enabled
        invalidate()
    }
    
    /**
     * Count valid (non-zero) keypoints.
     */
    private fun countValidPoints(kp: FloatArray): Int {
        var count = 0
        for (i in 0 until minOf(78, kp.size / 2)) {
            if (isValidPoint(kp[i * 2], kp[i * 2 + 1])) {
                count++
            }
        }
        return count
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        val kp = keypoints
        
        // Draw debug info if no keypoints
        if (kp == null || kp.size < 156) {
            if (debugMode) {
                drawDebugInfo(canvas, "No keypoints available", kp?.size ?: 0)
            }
            return
        }

        drawPoseSkeleton(canvas, kp)
        drawHandSkeleton(canvas, kp, 50)
        drawHandSkeleton(canvas, kp, 92)
        
        // Draw debug info if enabled
        if (debugMode) {
            val validPoints = countValidPoints(kp)
            drawDebugInfo(canvas, "Valid keypoints: $validPoints / 78", kp.size)
        }
    }
    
    /**
     * Draw debug information overlay.
     */
    private fun drawDebugInfo(canvas: Canvas, message: String, keypointSize: Int) {
        // Draw semi-transparent background
        canvas.drawRect(10f, 10f, 400f, 100f, debugBgPaint)
        
        // Draw text
        canvas.drawText(message, 20f, 40f, debugTextPaint)
        canvas.drawText("Keypoint array size: $keypointSize", 20f, 70f, debugTextPaint)
    }

    /**
     * Draw pose skeleton with connections.
     */
    private fun drawPoseSkeleton(canvas: Canvas, kp: FloatArray) {
        // Define pose connections (simplified upper body)
        val poseConnections = listOf(
            // Face outline
            Pair(0, 1), Pair(1, 2), Pair(2, 3), Pair(3, 7),
            Pair(0, 4), Pair(4, 5), Pair(5, 6), Pair(6, 8),
            // Shoulders
            Pair(11, 12),
            // Left arm
            Pair(11, 13), Pair(13, 15),
            // Right arm
            Pair(12, 14), Pair(14, 16),
            // Torso
            Pair(11, 23), Pair(12, 24), Pair(23, 24)
        )

        // Draw connections
        for ((idx1, idx2) in poseConnections) {
            if (idx1 * 2 + 1 >= kp.size || idx2 * 2 + 1 >= kp.size) continue
            
            val x1 = kp[idx1 * 2] * imageWidth * scaleFactor
            val y1 = kp[idx1 * 2 + 1] * imageHeight * scaleFactor
            val x2 = kp[idx2 * 2] * imageWidth * scaleFactor
            val y2 = kp[idx2 * 2 + 1] * imageHeight * scaleFactor
            
            // Check if both points are valid (not zero, which indicates missing detection)
            if (isValidPoint(kp[idx1 * 2], kp[idx1 * 2 + 1]) && 
                isValidPoint(kp[idx2 * 2], kp[idx2 * 2 + 1])) {
                canvas.drawLine(x1, y1, x2, y2, linePaint)
            }
        }

        // Draw keypoints
        for (i in 0 until 25) {
            if (i * 2 + 1 >= kp.size) break
            
            val x = kp[i * 2] * imageWidth * scaleFactor
            val y = kp[i * 2 + 1] * imageHeight * scaleFactor
            if (isValidPoint(kp[i * 2], kp[i * 2 + 1])) {
                canvas.drawCircle(x, y, 8f, keypointPaint)
            }
        }
    }
    
    /**
     * Check if a point is valid (detected by MediaPipe).
     * MediaPipe returns 0.0 for missing landmarks, but 0.0 can also be a valid coordinate.
     * We use a threshold to determine if the point is likely valid.
     */
    private fun isValidPoint(x: Float, y: Float): Boolean {
        // If both coordinates are exactly 0.0, it's likely an undetected landmark
        if (x == 0f && y == 0f) return false
        // Check if coordinates are within valid normalized range
        return x in 0.0f..1.0f && y in 0.0f..1.0f
    }

    /**
     * Draw hand skeleton with connections.
     * @param startIdx Starting index in the keypoints array (50 for left, 92 for right)
     */
    private fun drawHandSkeleton(canvas: Canvas, kp: FloatArray, startIdx: Int) {
        // Hand landmark connections (standard MediaPipe hand connections)
        val handConnections = listOf(
            // Thumb
            Pair(0, 1), Pair(1, 2), Pair(2, 3), Pair(3, 4),
            // Index finger
            Pair(0, 5), Pair(5, 6), Pair(6, 7), Pair(7, 8),
            // Middle finger
            Pair(0, 9), Pair(9, 10), Pair(10, 11), Pair(11, 12),
            // Ring finger
            Pair(0, 13), Pair(13, 14), Pair(14, 15), Pair(15, 16),
            // Pinky
            Pair(0, 17), Pair(17, 18), Pair(18, 19), Pair(19, 20),
            // Palm
            Pair(5, 9), Pair(9, 13), Pair(13, 17)
        )

        // Draw connections
        for ((idx1, idx2) in handConnections) {
            val globalIdx1 = startIdx + idx1 * 2
            val globalIdx2 = startIdx + idx2 * 2
            
            if (globalIdx1 + 1 >= kp.size || globalIdx2 + 1 >= kp.size) continue
            
            val x1 = kp[globalIdx1] * imageWidth * scaleFactor
            val y1 = kp[globalIdx1 + 1] * imageHeight * scaleFactor
            val x2 = kp[globalIdx2] * imageWidth * scaleFactor
            val y2 = kp[globalIdx2 + 1] * imageHeight * scaleFactor
            
            if (isValidPoint(kp[globalIdx1], kp[globalIdx1 + 1]) && 
                isValidPoint(kp[globalIdx2], kp[globalIdx2 + 1])) {
                canvas.drawLine(x1, y1, x2, y2, handLinePaint)
            }
        }

        // Draw keypoints
        for (i in 0 until 21) {
            val globalIdx = startIdx + i * 2
            if (globalIdx + 1 >= kp.size) break
            
            val x = kp[globalIdx] * imageWidth * scaleFactor
            val y = kp[globalIdx + 1] * imageHeight * scaleFactor
            if (isValidPoint(kp[globalIdx], kp[globalIdx + 1])) {
                canvas.drawCircle(x, y, 6f, handPaint)
            }
        }
    }
}


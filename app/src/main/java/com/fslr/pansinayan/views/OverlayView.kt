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
    
    // Paint for drawing body keypoints (RED)
    private val keypointPaint = Paint().apply {
        color = Color.rgb(255, 0, 0)
        strokeWidth = 10f
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    // Paint for drawing body skeleton connections (RED)
    private val linePaint = Paint().apply {
        color = Color.rgb(255, 0, 0)
        strokeWidth = 6f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }
    
    // Paint for face keypoints (YELLOW)
    private val facePaint = Paint().apply {
        color = Color.rgb(255, 255, 0)
        strokeWidth = 10f
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    // Paint for face connections (YELLOW)
    private val faceLinePaint = Paint().apply {
        color = Color.rgb(255, 255, 0)
        strokeWidth = 5f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }
    
    // Paint for hands (GREEN)
    private val handPaint = Paint().apply {
        color = Color.rgb(0, 255, 0)
        strokeWidth = 10f
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    // Paint for hand connections (GREEN)
    private val handLinePaint = Paint().apply {
        color = Color.rgb(0, 255, 0)
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
     * New total: 89 points (25 pose + 21 left hand + 21 right hand + 22 face)
     */
    private fun countValidPoints(kp: FloatArray): Int {
        var count = 0
        for (i in 0 until minOf(89, kp.size / 2)) {
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
        if (kp == null || kp.size < 178) {
            if (debugMode) {
                drawDebugInfo(canvas, "No keypoints available", kp?.size ?: 0)
            }
            return
        }

        drawPoseSkeleton(canvas, kp)
        drawHandSkeleton(canvas, kp, 50)  // Left hand starts at index 25 (50 values in)
        drawHandSkeleton(canvas, kp, 92)  // Right hand starts at index 46 (92 values in)
        drawFaceLandmarks(canvas, kp, 134) // Face starts at index 67 (134 values in)
        
        // Draw debug info if enabled
        if (debugMode) {
            val validPoints = countValidPoints(kp)
            drawDebugInfo(canvas, "Valid keypoints: $validPoints / 89", kp.size)
        }
    }
    
    /**
     * Draw debug information overlay.
     * Positioned at bottom left, just above the skeleton toggle switch.
     */
    private fun drawDebugInfo(canvas: Canvas, message: String, keypointSize: Int) {
        // Position at bottom left (just above skeleton toggle)
        // Account for: stats card (~50dp), transcript card (~varies), skeleton toggle (~60dp), margins
        val bottomMargin = 600f  // Space for stats, transcript, and skeleton toggle
        val boxHeight = 90f
        val boxTop = height - bottomMargin - boxHeight
        val boxBottom = height - bottomMargin
        val leftMargin = 48f  // Align with other components (16dp ≈ 48px)
        
        // Draw semi-transparent background
        canvas.drawRect(leftMargin, boxTop, 520f, boxBottom, debugBgPaint)
        
        // Draw text
        canvas.drawText(message, leftMargin + 12f, boxTop + 35f, debugTextPaint)
        canvas.drawText("Keypoint array size: $keypointSize", leftMargin + 12f, boxTop + 65f, debugTextPaint)
    }

    /**
     * Draw pose skeleton with connections.
     * Note: Face keypoints (indices 0-10) from pose are now replaced by proper Face Landmarker
     */
    private fun drawPoseSkeleton(canvas: Canvas, kp: FloatArray) {
        // Define body connections (RED)
        val bodyConnections = listOf(
            // Shoulders
            Pair(11, 12),
            // Left arm
            Pair(11, 13), Pair(13, 15),
            // Right arm
            Pair(12, 14), Pair(14, 16),
            // Torso
            Pair(11, 23), Pair(12, 24), Pair(23, 24)
        )
        
        // Draw body connections (RED)
        for ((idx1, idx2) in bodyConnections) {
            if (idx1 * 2 + 1 >= kp.size || idx2 * 2 + 1 >= kp.size) continue
            
            val x1 = kp[idx1 * 2] * imageWidth * scaleFactor
            val y1 = kp[idx1 * 2 + 1] * imageHeight * scaleFactor
            val x2 = kp[idx2 * 2] * imageWidth * scaleFactor
            val y2 = kp[idx2 * 2 + 1] * imageHeight * scaleFactor
            
            if (isValidPoint(kp[idx1 * 2], kp[idx1 * 2 + 1]) && 
                isValidPoint(kp[idx2 * 2], kp[idx2 * 2 + 1])) {
                canvas.drawLine(x1, y1, x2, y2, linePaint)
            }
        }
        
        // Draw body keypoints (RED) - indices 11-24
        for (i in 11 until 25) {
            if (i * 2 + 1 >= kp.size) break
            
            val x = kp[i * 2] * imageWidth * scaleFactor
            val y = kp[i * 2 + 1] * imageHeight * scaleFactor
            if (isValidPoint(kp[i * 2], kp[i * 2 + 1])) {
                canvas.drawCircle(x, y, 12f, keypointPaint)
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
                canvas.drawCircle(x, y, 9f, handPaint)
            }
        }
    }
    
    /**
     * Draw minimal face landmarks (22 points).
     * These are: 8 lips, 6 eyes, 6 eyebrows, 2 nose
     * @param startIdx Starting index in the keypoints array (134 for face)
     */
    private fun drawFaceLandmarks(canvas: Canvas, kp: FloatArray, startIdx: Int) {
        // Lip connections: 3 upper + 3 lower + 2 corners (8 points)
        // 0: Left corner (61), 1: Upper left-outer (78), 2: Upper center (13),
        // 3: Upper right-outer (308), 4: Right corner (291), 5: Lower right-center (375),
        // 6: Lower center (14), 7: Lower left-center (146)
        val lipConnections = listOf(
            // Upper lip arc (left corner → upper curve → center → upper curve → right corner)
            Pair(0, 1),  // Left corner → upper left-outer
            Pair(1, 2),  // Upper left-outer → upper center
            Pair(2, 3),  // Upper center → upper right-outer
            Pair(3, 4),  // Upper right-outer → right corner
            
            // Lower lip arc (right corner → lower curve → center → lower curve → left corner)
            Pair(4, 5),  // Right corner → lower right-center
            Pair(5, 6),  // Lower right-center → lower center
            Pair(6, 7),  // Lower center → lower left-center
            Pair(7, 0)   // Lower left-center → left corner (close loop)
        )
        
        // Draw lip connections (YELLOW)
        for ((idx1, idx2) in lipConnections) {
            val globalIdx1 = startIdx + idx1 * 2
            val globalIdx2 = startIdx + idx2 * 2
            
            if (globalIdx1 + 1 >= kp.size || globalIdx2 + 1 >= kp.size) continue
            
            val x1 = kp[globalIdx1] * imageWidth * scaleFactor
            val y1 = kp[globalIdx1 + 1] * imageHeight * scaleFactor
            val x2 = kp[globalIdx2] * imageWidth * scaleFactor
            val y2 = kp[globalIdx2 + 1] * imageHeight * scaleFactor
            
            if (isValidPoint(kp[globalIdx1], kp[globalIdx1 + 1]) && 
                isValidPoint(kp[globalIdx2], kp[globalIdx2 + 1])) {
                canvas.drawLine(x1, y1, x2, y2, faceLinePaint)
            }
        }
        
        // Draw all 22 face keypoints (YELLOW)
        for (i in 0 until 22) {
            val globalIdx = startIdx + i * 2
            if (globalIdx + 1 >= kp.size) break
            
            val x = kp[globalIdx] * imageWidth * scaleFactor
            val y = kp[globalIdx + 1] * imageHeight * scaleFactor
            if (isValidPoint(kp[globalIdx], kp[globalIdx + 1])) {
                // Use different sizes for different features
                val radius = when (i) {
                    in 0..7 -> 8f    // Lips - medium
                    in 8..13 -> 6f   // Eyes - small
                    in 14..19 -> 6f  // Eyebrows - small
                    else -> 7f       // Nose - medium
                }
                canvas.drawCircle(x, y, radius, facePaint)
            }
        }
    }
}


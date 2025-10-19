package com.fslr.pansinayan.views

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
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

    private var keypoints: FloatArray? = null
    
    // Paint for drawing keypoints
    private val keypointPaint = Paint().apply {
        color = Color.GREEN
        strokeWidth = 8f
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    // Paint for drawing skeleton connections
    private val linePaint = Paint().apply {
        color = Color.rgb(0, 255, 0)
        strokeWidth = 5f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }
    
    // Paint for hands (different color)
    private val handPaint = Paint().apply {
        color = Color.YELLOW
        strokeWidth = 8f
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    /**
     * Update the keypoints to display.
     * @param keypoints FloatArray[156] containing normalized coordinates (x, y)
     */
    fun setKeypoints(keypoints: FloatArray?) {
        this.keypoints = keypoints
        invalidate() // Trigger redraw
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        val kp = keypoints ?: return
        if (kp.size < 156) return

        val width = width.toFloat()
        val height = height.toFloat()

        // Draw pose landmarks (indices 0-49, representing 25 points)
        drawPoseSkeleton(canvas, kp, width, height)
        
        // Draw left hand landmarks (indices 50-91, representing 21 points)
        drawHandSkeleton(canvas, kp, 50, width, height)
        
        // Draw right hand landmarks (indices 92-133, representing 21 points)
        drawHandSkeleton(canvas, kp, 92, width, height)
    }

    /**
     * Draw pose skeleton with connections.
     */
    private fun drawPoseSkeleton(canvas: Canvas, kp: FloatArray, w: Float, h: Float) {
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
            val x1 = kp[idx1 * 2] * w
            val y1 = kp[idx1 * 2 + 1] * h
            val x2 = kp[idx2 * 2] * w
            val y2 = kp[idx2 * 2 + 1] * h
            
            if (x1 > 0 && y1 > 0 && x2 > 0 && y2 > 0) {
                canvas.drawLine(x1, y1, x2, y2, linePaint)
            }
        }

        // Draw keypoints
        for (i in 0 until 25) {
            val x = kp[i * 2] * w
            val y = kp[i * 2 + 1] * h
            if (x > 0 && y > 0) {
                canvas.drawCircle(x, y, 8f, keypointPaint)
            }
        }
    }

    /**
     * Draw hand skeleton with connections.
     * @param startIdx Starting index in the keypoints array (50 for left, 92 for right)
     */
    private fun drawHandSkeleton(canvas: Canvas, kp: FloatArray, startIdx: Int, w: Float, h: Float) {
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
            val x1 = kp[startIdx + idx1 * 2] * w
            val y1 = kp[startIdx + idx1 * 2 + 1] * h
            val x2 = kp[startIdx + idx2 * 2] * w
            val y2 = kp[startIdx + idx2 * 2 + 1] * h
            
            if (x1 > 0 && y1 > 0 && x2 > 0 && y2 > 0) {
                canvas.drawLine(x1, y1, x2, y2, linePaint)
            }
        }

        // Draw keypoints
        for (i in 0 until 21) {
            val x = kp[startIdx + i * 2] * w
            val y = kp[startIdx + i * 2 + 1] * h
            if (x > 0 && y > 0) {
                canvas.drawCircle(x, y, 6f, handPaint)
            }
        }
    }
}


package com.fslr.pansinayan.recognition

import android.util.Log

/**
 * Manages keypoint buffer with sign-aware window extraction.
 * 
 * Maintains a rolling buffer of keypoints and extracts windows
 * aligned with detected sign boundaries rather than fixed strides.
 */
class AdaptiveBufferManager(
    private val maxBufferSize: Int = 300,  // ~10 seconds at 30fps
    private val windowSize: Int = 150,     // Model's expected window size
    private val maxGap: Int = 5            // Max gap for interpolation
) {
    companion object {
        private const val TAG = "AdaptiveBufferManager"
        private const val MIN_SEQUENCE_LENGTH = 60
    }

    // Frame storage with absolute frame indices
    data class FrameData(
        val keypoints: FloatArray?,
        val frameIndex: Int,
        val timestamp: Long
    )

    private val buffer = ArrayDeque<FrameData>()
    private var frameCounter: Int = 0

    /**
     * Add a new keypoint frame to the buffer.
     */
    @Synchronized
    fun addFrame(keypoints: FloatArray?) {
        val frameData = FrameData(
            keypoints = keypoints,
            frameIndex = frameCounter++,
            timestamp = System.currentTimeMillis()
        )
        
        buffer.addLast(frameData)
        
        // Remove oldest frames if buffer exceeds max size
        while (buffer.size > maxBufferSize) {
            buffer.removeFirst()
        }
    }

    /**
     * Extract a window containing a complete sign.
     * 
     * @param signStartFrame Absolute frame index where sign started
     * @param signEndFrame Absolute frame index where sign ended
     * @return Interpolated sequence array, or null if insufficient data
     */
    @Synchronized
    fun extractSignWindow(signStartFrame: Int, signEndFrame: Int): Array<FloatArray>? {
        if (buffer.isEmpty()) {
            Log.d(TAG, "Buffer empty - cannot extract window")
            return null
        }

        // Find frames in buffer that contain the sign
        val signFrames = buffer.filter { 
            it.frameIndex >= signStartFrame && it.frameIndex <= signEndFrame 
        }

        if (signFrames.isEmpty()) {
            Log.w(TAG, "No frames found for sign [$signStartFrame-$signEndFrame]")
            return null
        }

        // Calculate padding to reach window size
        val signDuration = signEndFrame - signStartFrame + 1
        val padding = if (signDuration < windowSize) {
            (windowSize - signDuration) / 2
        } else {
            0
        }

        val windowStartFrame = maxOf(0, signStartFrame - padding)
        val windowEndFrame = signEndFrame + padding

        // Extract frames for the window
        val windowFrames = buffer.filter {
            it.frameIndex >= windowStartFrame && it.frameIndex <= windowEndFrame
        }.sortedBy { it.frameIndex }

        if (windowFrames.size < MIN_SEQUENCE_LENGTH) {
            Log.d(TAG, "Insufficient frames: ${windowFrames.size} < $MIN_SEQUENCE_LENGTH")
            return null
        }

        // Convert to keypoint array and interpolate gaps
        val keypointSequence = windowFrames.map { it.keypoints }.toTypedArray()
        val interpolated = interpolateGaps(keypointSequence)

        // Pad or truncate to exact window size
        return adjustToWindowSize(interpolated)
    }

    /**
     * Extract a window centered around a specific frame.
     * Used for fallback inference triggers.
     */
    @Synchronized
    fun extractWindowAroundFrame(centerFrame: Int): Array<FloatArray>? {
        if (buffer.isEmpty()) return null

        val halfWindow = windowSize / 2
        val windowStartFrame = maxOf(0, centerFrame - halfWindow)
        val windowEndFrame = centerFrame + halfWindow

        val windowFrames = buffer.filter {
            it.frameIndex >= windowStartFrame && it.frameIndex <= windowEndFrame
        }.sortedBy { it.frameIndex }

        if (windowFrames.size < MIN_SEQUENCE_LENGTH) {
            return null
        }

        val keypointSequence = windowFrames.map { it.keypoints }.toTypedArray()
        val interpolated = interpolateGaps(keypointSequence)
        return adjustToWindowSize(interpolated)
    }

    /**
     * Get the most recent window (for fallback).
     */
    @Synchronized
    fun extractRecentWindow(): Array<FloatArray>? {
        if (buffer.size < MIN_SEQUENCE_LENGTH) return null

        val recentFrames = buffer.takeLast(windowSize)
        val keypointSequence = recentFrames.map { it.keypoints }.toTypedArray()
        val interpolated = interpolateGaps(keypointSequence)
        return adjustToWindowSize(interpolated)
    }

    /**
     * Adjust sequence to exact window size (pad or truncate).
     */
    private fun adjustToWindowSize(sequence: Array<FloatArray>): Array<FloatArray> {
        return when {
            sequence.size == windowSize -> sequence
            sequence.size < windowSize -> {
                // Pad with last frame
                val padded = Array(windowSize) { i ->
                    if (i < sequence.size) {
                        sequence[i]
                    } else {
                        sequence.lastOrNull()?.clone() ?: FloatArray(178) { 0f }
                    }
                }
                padded
            }
            else -> {
                // Truncate to window size (keep most recent)
                sequence.takeLast(windowSize).toTypedArray()
            }
        }
    }

    /**
     * Interpolate missing keypoints (null frames) using linear interpolation.
     */
    private fun interpolateGaps(sequence: Array<FloatArray?>): Array<FloatArray> {
        val result = Array(sequence.size) { FloatArray(178) { 0f } }

        for (i in sequence.indices) {
            if (sequence[i] != null) {
                result[i] = sequence[i]!!.clone()
            } else {
                // Find nearest valid frames
                var prevIdx = -1
                for (j in i - 1 downTo 0) {
                    if (sequence[j] != null) {
                        prevIdx = j
                        break
                    }
                }

                var nextIdx = -1
                for (j in i + 1 until sequence.size) {
                    if (sequence[j] != null) {
                        nextIdx = j
                        break
                    }
                }

                when {
                    prevIdx != -1 && nextIdx != -1 && (nextIdx - prevIdx) <= maxGap -> {
                        // Linear interpolation
                        val t = (i - prevIdx).toFloat() / (nextIdx - prevIdx).toFloat()
                        for (k in 0 until 178) {
                            result[i][k] = lerp(sequence[prevIdx]!![k], sequence[nextIdx]!![k], t)
                        }
                    }
                    prevIdx != -1 -> {
                        result[i] = sequence[prevIdx]!!.clone()
                    }
                    nextIdx != -1 -> {
                        result[i] = sequence[nextIdx]!!.clone()
                    }
                    // else: leave as zeros
                }
            }
        }

        return result
    }

    private fun lerp(a: Float, b: Float, t: Float): Float = a + (b - a) * t

    /**
     * Get current buffer size.
     */
    fun getBufferSize(): Int = buffer.size

    /**
     * Get frame counter (total frames processed).
     */
    fun getFrameCounter(): Int = frameCounter

    /**
     * Clear buffer.
     */
    @Synchronized
    fun clear() {
        buffer.clear()
        frameCounter = 0
        Log.d(TAG, "Buffer cleared")
    }

    /**
     * Get missing ratio for a sequence (for quality check).
     */
    fun getMissingRatio(sequence: Array<FloatArray?>): Float {
        val nullCount = sequence.count { it == null }
        return if (sequence.isNotEmpty()) nullCount.toFloat() / sequence.size else 1f
    }
}


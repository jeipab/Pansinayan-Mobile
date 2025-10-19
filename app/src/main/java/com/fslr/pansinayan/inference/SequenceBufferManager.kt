package com.fslr.pansinayan.inference

import android.util.Log
import java.util.*

/**
 * Manages a sliding window buffer of keypoint sequences.
 * 
 * Responsibilities:
 * - Maintain circular buffer of recent keypoint frames
 * - Handle missing detections (null frames)
 * - Perform gap interpolation for smoother sequences
 * - Prepare sequences for TFLite inference
 * 
 * Usage:
 *   val bufferManager = SequenceBufferManager(windowSize = 90)
 *   bufferManager.addFrame(keypoints)  // Add each new frame
 *   val sequence = bufferManager.getSequence()  // Get interpolated sequence
 */
class SequenceBufferManager(
    private val windowSize: Int = 90,  // 3 seconds at 30 FPS
    private val maxGap: Int = 5  // Maximum gap for interpolation
) {
    companion object {
        private const val TAG = "SequenceBufferManager"
        private const val MIN_SEQUENCE_LENGTH = 30  // Minimum 1 second of data
    }

    // Circular buffer to store keypoint sequences
    private val buffer = LinkedList<FloatArray?>()
    
    // Timestamps for each frame
    private val timestamps = LinkedList<Long>()

    /**
     * Add a new keypoint frame to the buffer.
     * 
     * @param keypoints FloatArray of 156 values (or null if detection failed)
     */
    @Synchronized
    fun addFrame(keypoints: FloatArray?) {
        // Add to buffer
        buffer.add(keypoints)
        timestamps.add(System.currentTimeMillis())

        // Remove oldest frame if buffer exceeds window size
        while (buffer.size > windowSize) {
            buffer.removeFirst()
            timestamps.removeFirst()
        }
    }

    /**
     * Get the current sequence for inference.
     * Performs gap interpolation for missing keypoints.
     * 
     * @return Array[T, 156] where T ≤ windowSize, or null if insufficient data
     */
    @Synchronized
    fun getSequence(): Array<FloatArray>? {
        if (buffer.size < MIN_SEQUENCE_LENGTH) {
            Log.d(TAG, "Insufficient data: ${buffer.size} frames (minimum: $MIN_SEQUENCE_LENGTH)")
            return null
        }

        // Copy buffer to array
        val sequence = buffer.toTypedArray()

        // Perform gap interpolation
        return interpolateGaps(sequence)
    }

    /**
     * Interpolate missing keypoints (null frames) using linear interpolation.
     * 
     * For each null frame:
     * 1. Find nearest valid frames before and after
     * 2. If gap <= maxGap, linearly interpolate between them
     * 3. Otherwise, copy from nearest valid frame
     * 4. If no valid frames, leave as zeros
     */
    private fun interpolateGaps(sequence: Array<FloatArray?>): Array<FloatArray> {
        val result = Array(sequence.size) { FloatArray(156) { 0f } }
        var interpolatedCount = 0

        for (i in sequence.indices) {
            if (sequence[i] != null) {
                // Valid frame - copy directly
                result[i] = sequence[i]!!.clone()
            } else {
                // Missing frame - interpolate
                
                // Find nearest valid frame before this one
                var prevIdx = -1
                for (j in i - 1 downTo 0) {
                    if (sequence[j] != null) {
                        prevIdx = j
                        break
                    }
                }

                // Find nearest valid frame after this one
                var nextIdx = -1
                for (j in i + 1 until sequence.size) {
                    if (sequence[j] != null) {
                        nextIdx = j
                        break
                    }
                }

                // Interpolate if gap is within threshold
                if (prevIdx != -1 && nextIdx != -1 && (nextIdx - prevIdx) <= maxGap) {
                    // Linear interpolation
                    val t = (i - prevIdx).toFloat() / (nextIdx - prevIdx).toFloat()
                    for (k in 0 until 156) {
                        result[i][k] = lerp(sequence[prevIdx]!![k], sequence[nextIdx]!![k], t)
                    }
                    interpolatedCount++
                } else if (prevIdx != -1) {
                    // Gap too large or no next frame - use previous frame
                    result[i] = sequence[prevIdx]!!.clone()
                } else if (nextIdx != -1) {
                    // No previous frame - use next frame
                    result[i] = sequence[nextIdx]!!.clone()
                }
                // Else: leave as zeros (no valid frames nearby)
            }
        }

        if (interpolatedCount > 0) {
            Log.d(TAG, "Interpolated $interpolatedCount frames out of ${sequence.size}")
        }

        return result
    }

    /**
     * Linear interpolation between two values.
     */
    private fun lerp(a: Float, b: Float, t: Float): Float {
        return a + (b - a) * t
    }

    /**
     * Get current buffer size.
     */
    fun getBufferSize(): Int = buffer.size

    /**
     * Clear buffer (useful for restarting recognition).
     */
    @Synchronized
    fun clear() {
        buffer.clear()
        timestamps.clear()
        Log.i(TAG, "Buffer cleared")
    }

    /**
     * Check if buffer has enough data for inference.
     */
    fun hasEnoughData(): Boolean {
        return buffer.size >= MIN_SEQUENCE_LENGTH
    }

    /**
     * Get statistics about null frames in buffer.
     */
    fun getNullFrameStats(): Pair<Int, Int> {
        val nullCount = buffer.count { it == null }
        val totalCount = buffer.size
        return Pair(nullCount, totalCount)
    }
}


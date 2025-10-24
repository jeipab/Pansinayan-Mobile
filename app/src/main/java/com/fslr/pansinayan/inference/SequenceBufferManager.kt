package com.fslr.pansinayan.inference

import android.util.Log
import java.util.*

/**
 * Manages a rolling buffer of keypoint sequences for continuous CTC recognition.
 * 
 * Responsibilities:
 * - Maintain rolling buffer of recent keypoint frames (up to 300 frames)
 * - Handle missing detections (null frames) with interpolation
 * - Support continuous streaming without reset between predictions
 * - Prepare sequences for CTC model inference with zero-padding
 * 
 * Usage:
 *   val bufferManager = CTCSequenceBufferManager()
 *   bufferManager.addFrame(keypoints)  // Add each new frame
 *   val sequence = bufferManager.getCurrentSequence()  // Get rolling sequence
 */
class CTCSequenceBufferManager(
    private val maxWindowSize: Int = 300,  // 10 seconds at 30 FPS for CTC models
    private val maxGap: Int = 5  // Maximum gap for interpolation
) {
    companion object {
        private const val TAG = "CTCSequenceBufferManager"
        private const val MIN_SEQUENCE_LENGTH = 30  // Minimum 1 second of data for CTC
    }

    // Rolling buffer to store keypoint sequences
    private val buffer = LinkedList<FloatArray?>()
    
    // Timestamps for each frame
    private val timestamps = LinkedList<Long>()
    
    // Frame counter for continuous tracking
    private var frameCounter = 0L

    /**
     * Add a new keypoint frame to the rolling buffer.
     * 
     * @param keypoints FloatArray of 178 values (89 keypoints × 2, or null if detection failed)
     */
    @Synchronized
    fun addFrame(keypoints: FloatArray?) {
        // Add to buffer
        buffer.add(keypoints)
        timestamps.add(System.currentTimeMillis())
        frameCounter++

        // Remove oldest frame if buffer exceeds max window size
        while (buffer.size > maxWindowSize) {
            buffer.removeFirst()
            timestamps.removeFirst()
        }
    }

    /**
     * Get the current rolling sequence for CTC inference.
     * Performs gap interpolation for missing keypoints.
     * 
     * @return Array[T, 178] where T ≤ maxWindowSize, or null if insufficient data
     */
    @Synchronized
    fun getCurrentSequence(): Array<FloatArray>? {
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
     * Get a sliding window sequence of specified length.
     * Useful for different inference strategies.
     * 
     * @param length Desired sequence length (≤ maxWindowSize)
     * @return Array[length, 178] or null if insufficient data
     */
    @Synchronized
    fun getSlidingWindowSequence(length: Int): Array<FloatArray>? {
        val actualLength = minOf(length, buffer.size, maxWindowSize)
        
        if (actualLength < MIN_SEQUENCE_LENGTH) {
            Log.d(TAG, "Insufficient data: ${buffer.size} frames (requested: $length, minimum: $MIN_SEQUENCE_LENGTH)")
            return null
        }

        // Get the most recent frames
        val recentFrames = buffer.takeLast(actualLength).toTypedArray()
        
        // Perform gap interpolation
        return interpolateGaps(recentFrames)
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
        val result = Array(sequence.size) { FloatArray(178) { 0f } }
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
                    for (k in 0 until 178) {
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
     * Get frame counter for continuous tracking.
     */
    fun getFrameCounter(): Long = frameCounter

    /**
     * Clear buffer (useful for restarting recognition).
     */
    @Synchronized
    fun clear() {
        buffer.clear()
        timestamps.clear()
        frameCounter = 0L
        Log.i(TAG, "CTC buffer cleared")
    }

    /**
     * Check if buffer has enough data for CTC inference.
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

    /**
     * Get buffer statistics for debugging.
     */
    fun getBufferStats(): CTCBufferStats {
        val nullCount = buffer.count { it == null }
        val validCount = buffer.size - nullCount
        val nullRate = if (buffer.size > 0) (nullCount.toFloat() / buffer.size * 100) else 0f
        
        return CTCBufferStats(
            totalFrames = buffer.size,
            validFrames = validCount,
            nullFrames = nullCount,
            nullRate = nullRate,
            frameCounter = frameCounter,
            bufferUtilization = (buffer.size.toFloat() / maxWindowSize * 100)
}

/**
 * Statistics for CTC sequence buffer.
 */
data class CTCBufferStats(
    val totalFrames: Int,
    val validFrames: Int,
    val nullFrames: Int,
    val nullRate: Float,
    val frameCounter: Long,
    val bufferUtilization: Float
)


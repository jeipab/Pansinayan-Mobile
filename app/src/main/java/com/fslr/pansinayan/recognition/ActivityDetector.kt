package com.fslr.pansinayan.recognition

import android.util.Log
import kotlin.math.sqrt

/**
 * Detects user activity (signing vs idle) based on keypoint motion.
 * 
 * Computes motion metrics from keypoints and classifies activity state.
 * This enables inference to be triggered only when user is actively signing.
 */
class ActivityDetector(
    private val motionThreshold: Float = 0.001f,  // Lowered threshold for better sensitivity
    private val activeFramesRequired: Int = 3,    // Reduced frames required for faster response
    private val idleFramesRequired: Int = 10,     // Reduced frames for faster idle detection
    private val motionWindowSize: Int = 10
) {
    companion object {
        private const val TAG = "ActivityDetector"
    }

    enum class ActivityState {
        IDLE,       // No significant motion detected
        ACTIVE,     // Active signing detected
        TRANSITION  // Between idle and active
    }

    private var previousKeypoints: FloatArray? = null
    // Use array-based circular buffer instead of ArrayDeque to avoid iterator issues
    private val motionHistory = FloatArray(motionWindowSize)
    private var motionHistorySize = 0
    private var motionHistoryIndex = 0
    private val motionHistoryLock = Any()
    private var consecutiveActiveFrames = 0
    private var consecutiveIdleFrames = 0
    @Volatile
    private var currentState = ActivityState.IDLE

    /**
     * Process new keypoints and update activity state.
     * 
     * @param keypoints Current frame keypoints (178 values)
     * @return Pair of (ActivityState, currentMotion) for efficient access
     */
    fun processFrame(keypoints: FloatArray?): Pair<ActivityState, Float> {
        if (keypoints == null) {
            // Missing keypoints - treat as idle
            updateMotionScore(0f)
            val motion = getAverageMotion()
            return Pair(currentState, motion)
        }

        val motionScore = computeMotionScore(keypoints)
        updateMotionScore(motionScore)
        
        val state = updateState()
        val motion = getAverageMotion()
        return Pair(state, motion)
    }

    /**
     * Compute motion score by comparing current and previous keypoints.
     * Focuses on hand and upper body pose keypoints.
     */
    private fun computeMotionScore(current: FloatArray): Float {
        val previous = previousKeypoints
        if (previous == null) {
            // First frame - initialize and return 0 (no motion yet)
            previousKeypoints = current.clone()
            return 0f
        }
        
        var totalMotion = 0f
        var validKeypoints = 0
        
        // Hand keypoints: indices 50-133 (42 left + 42 right = 84 values)
        for (i in 50 until 134) {
            // Only count non-zero keypoints (valid detections)
            if (current[i] != 0f || previous[i] != 0f) {
                val delta = current[i] - previous[i]
                totalMotion += delta * delta
                validKeypoints++
            }
        }
        
        // Upper body pose keypoints: indices 0-49 (25 points × 2 = 50 values)
        for (i in 0 until 50) {
            // Only count non-zero keypoints (valid detections)
            if (current[i] != 0f || previous[i] != 0f) {
                val delta = current[i] - previous[i]
                totalMotion += delta * delta
                validKeypoints++
            }
        }
        
        previousKeypoints = current.clone()
        
        // Normalize by number of valid keypoints to get average motion per keypoint
        // Then scale up since we're using squared differences
        if (validKeypoints > 0) {
            val avgMotion = totalMotion / validKeypoints
            // Use sqrt to get magnitude, then scale for better sensitivity
            return sqrt(avgMotion) * 10f  // Scale factor to make motion more detectable
        }
        
        return 0f
    }

    /**
     * Update motion history using circular buffer.
     */
    private fun updateMotionScore(score: Float) {
        synchronized(motionHistoryLock) {
            // Add to circular buffer
            motionHistory[motionHistoryIndex] = score
            motionHistoryIndex = (motionHistoryIndex + 1) % motionWindowSize
            if (motionHistorySize < motionWindowSize) {
                motionHistorySize++
            }
        }
    }

    /**
     * Get average motion over recent frames.
     */
    private fun getAverageMotion(): Float {
        synchronized(motionHistoryLock) {
            if (motionHistorySize == 0) return 0f
            // Calculate average from circular buffer
            var sum = 0f
            if (motionHistorySize < motionWindowSize) {
                // Buffer not full yet - read from start (all values are valid and in order)
                for (i in 0 until motionHistorySize) {
                    sum += motionHistory[i]
                }
                return sum / motionHistorySize
            } else {
                // Buffer is full - read most recent motionWindowSize values
                // motionHistoryIndex points to where we'll write next (oldest value)
                // Read from motionHistoryIndex wrapping around
                for (i in 0 until motionWindowSize) {
                    val idx = (motionHistoryIndex + i) % motionWindowSize
                    sum += motionHistory[idx]
                }
                return sum / motionWindowSize
            }
        }
    }

    /**
     * Update activity state based on motion patterns.
     */
    private fun updateState(): ActivityState {
        val avgMotion = getAverageMotion()
        val isMotionAboveThreshold = avgMotion >= motionThreshold
        
        // Enhanced debug logging to diagnose issues
        if (motionHistorySize % 10 == 0 && motionHistorySize > 0) {
            Log.d(TAG, "Motion: avg=$avgMotion, threshold=$motionThreshold, above=${isMotionAboveThreshold}, " +
                    "state=$currentState, activeFrames=$consecutiveActiveFrames, idleFrames=$consecutiveIdleFrames, " +
                    "historySize=$motionHistorySize")
        }
        
        // Log state transitions immediately
        val oldState = currentState

        when (currentState) {
            ActivityState.IDLE -> {
                if (isMotionAboveThreshold) {
                    consecutiveActiveFrames++
                    consecutiveIdleFrames = 0
                    
                    if (consecutiveActiveFrames >= activeFramesRequired) {
                        currentState = ActivityState.ACTIVE
                        Log.i(TAG, "State: IDLE → ACTIVE (motion: $avgMotion, threshold: $motionThreshold)")
                    } else {
                        currentState = ActivityState.TRANSITION
                        if (oldState != ActivityState.TRANSITION) {
                            Log.d(TAG, "State: IDLE → TRANSITION (motion: $avgMotion, frames: $consecutiveActiveFrames/$activeFramesRequired)")
                        }
                    }
                } else {
                    consecutiveActiveFrames = 0
                    consecutiveIdleFrames++
                }
            }
            
            ActivityState.ACTIVE -> {
                if (!isMotionAboveThreshold) {
                    consecutiveIdleFrames++
                    consecutiveActiveFrames = 0
                    
                    if (consecutiveIdleFrames >= idleFramesRequired) {
                        currentState = ActivityState.IDLE
                        Log.i(TAG, "State: ACTIVE → IDLE (motion: $avgMotion)")
                    } else {
                        currentState = ActivityState.TRANSITION
                        if (oldState != ActivityState.TRANSITION) {
                            Log.d(TAG, "State: ACTIVE → TRANSITION (motion: $avgMotion, frames: $consecutiveIdleFrames/$idleFramesRequired)")
                        }
                    }
                } else {
                    consecutiveIdleFrames = 0
                    consecutiveActiveFrames++
                }
            }
            
            ActivityState.TRANSITION -> {
                if (isMotionAboveThreshold) {
                    consecutiveActiveFrames++
                    consecutiveIdleFrames = 0
                    if (consecutiveActiveFrames >= activeFramesRequired) {
                        currentState = ActivityState.ACTIVE
                        Log.i(TAG, "State: TRANSITION → ACTIVE (motion: $avgMotion)")
                    }
                } else {
                    consecutiveIdleFrames++
                    consecutiveActiveFrames = 0
                    if (consecutiveIdleFrames >= idleFramesRequired) {
                        currentState = ActivityState.IDLE
                        Log.i(TAG, "State: TRANSITION → IDLE (motion: $avgMotion)")
                    }
                }
            }
        }

        return currentState
    }

    /**
     * Get current activity state.
     */
    fun getState(): ActivityState = currentState

    /**
     * Check if currently active.
     */
    fun isActive(): Boolean = currentState == ActivityState.ACTIVE

    /**
     * Reset detector state.
     */
    fun reset() {
        synchronized(motionHistoryLock) {
            previousKeypoints = null
            motionHistorySize = 0
            motionHistoryIndex = 0
            // Clear array by setting to zero (optional, but cleaner)
            motionHistory.fill(0f)
            consecutiveActiveFrames = 0
            consecutiveIdleFrames = 0
            currentState = ActivityState.IDLE
        }
        Log.d(TAG, "Activity detector reset")
    }

    /**
     * Get current motion score (for debugging).
     */
    fun getCurrentMotion(): Float = getAverageMotion()
}


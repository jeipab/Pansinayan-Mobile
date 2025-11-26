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
    private val motionThreshold: Float = 0.01f,
    private val activeFramesRequired: Int = 5,
    private val idleFramesRequired: Int = 15,
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
    private val motionHistory = ArrayDeque<Float>(motionWindowSize)
    private var consecutiveActiveFrames = 0
    private var consecutiveIdleFrames = 0
    private var currentState = ActivityState.IDLE

    /**
     * Process new keypoints and update activity state.
     * 
     * @param keypoints Current frame keypoints (178 values)
     * @return Current activity state
     */
    fun processFrame(keypoints: FloatArray?): ActivityState {
        if (keypoints == null) {
            // Missing keypoints - treat as idle
            updateMotionScore(0f)
            return currentState
        }

        val motionScore = computeMotionScore(keypoints)
        updateMotionScore(motionScore)
        
        return updateState()
    }

    /**
     * Compute motion score by comparing current and previous keypoints.
     * Focuses on hand and upper body pose keypoints.
     */
    private fun computeMotionScore(current: FloatArray): Float {
        val previous = previousKeypoints ?: return 0f
        
        var totalMotion = 0f
        
        // Hand keypoints: indices 50-133 (42 left + 42 right = 84 values)
        for (i in 50 until 134) {
            val delta = current[i] - previous[i]
            totalMotion += delta * delta
        }
        
        // Upper body pose keypoints: indices 0-49 (25 points × 2 = 50 values)
        for (i in 0 until 50) {
            val delta = current[i] - previous[i]
            totalMotion += delta * delta
        }
        
        previousKeypoints = current.clone()
        return sqrt(totalMotion)
    }

    /**
     * Update motion history and compute average.
     */
    private fun updateMotionScore(score: Float) {
        motionHistory.addLast(score)
        if (motionHistory.size > motionWindowSize) {
            motionHistory.removeFirst()
        }
    }

    /**
     * Get average motion over recent frames.
     */
    private fun getAverageMotion(): Float {
        if (motionHistory.isEmpty()) return 0f
        return motionHistory.average().toFloat()
    }

    /**
     * Update activity state based on motion patterns.
     */
    private fun updateState(): ActivityState {
        val avgMotion = getAverageMotion()
        val isMotionAboveThreshold = avgMotion >= motionThreshold

        when (currentState) {
            ActivityState.IDLE -> {
                if (isMotionAboveThreshold) {
                    consecutiveActiveFrames++
                    consecutiveIdleFrames = 0
                    
                    if (consecutiveActiveFrames >= activeFramesRequired) {
                        currentState = ActivityState.ACTIVE
                        Log.d(TAG, "State: IDLE → ACTIVE (motion: $avgMotion)")
                    } else {
                        currentState = ActivityState.TRANSITION
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
                        Log.d(TAG, "State: ACTIVE → IDLE (motion: $avgMotion)")
                    } else {
                        currentState = ActivityState.TRANSITION
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
                        Log.d(TAG, "State: TRANSITION → ACTIVE")
                    }
                } else {
                    consecutiveIdleFrames++
                    consecutiveActiveFrames = 0
                    if (consecutiveIdleFrames >= idleFramesRequired) {
                        currentState = ActivityState.IDLE
                        Log.d(TAG, "State: TRANSITION → IDLE")
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
        previousKeypoints = null
        motionHistory.clear()
        consecutiveActiveFrames = 0
        consecutiveIdleFrames = 0
        currentState = ActivityState.IDLE
        Log.d(TAG, "Activity detector reset")
    }

    /**
     * Get current motion score (for debugging).
     */
    fun getCurrentMotion(): Float = getAverageMotion()
}


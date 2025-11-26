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
    private val armMotionThreshold: Float = 0.003f,    // Higher threshold for arm movement (sign start)
    private val handMotionThreshold: Float = 0.002f,   // Medium threshold for hand movement (active signing)
    private val jitterFilterThreshold: Float = 0.0005f, // Lower threshold to filter out noise
    private val activeFramesRequired: Int = 8,          // Increased - require sustained motion
    private val idleFramesRequired: Int = 20,           // Increased - require sustained rest
    private val motionWindowSize: Int = 15,             // Larger window for better smoothing
    private val motionPercentile: Float = 0.75f         // Use 75th percentile instead of average
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
    // Separate motion tracking for arms and hands
    private val armMotionHistory = FloatArray(motionWindowSize)
    private val handMotionHistory = FloatArray(motionWindowSize)
    private var motionHistorySize = 0
    private var motionHistoryIndex = 0
    private val motionHistoryLock = Any()
    private var consecutiveActiveFrames = 0
    private var consecutiveIdleFrames = 0
    @Volatile
    private var currentState = ActivityState.IDLE
    
    // Pose keypoint indices for arms (shoulders and elbows)
    // MediaPipe pose: 11=left shoulder, 12=right shoulder, 13=left elbow, 14=right elbow
    // In our array: pose is 0-49 (25 points × 2), so:
    // Left shoulder: ~22-23, Right shoulder: ~24-25, Left elbow: ~26-27, Right elbow: ~28-29
    private val armKeypointIndices = listOf(22, 23, 24, 25, 26, 27, 28, 29) // Shoulders and elbows

    /**
     * Process new keypoints and update activity state.
     * 
     * @param keypoints Current frame keypoints (178 values)
     * @return Pair of (ActivityState, currentMotion) for efficient access
     */
    fun processFrame(keypoints: FloatArray?): Pair<ActivityState, Float> {
        if (keypoints == null) {
            // Missing keypoints - treat as idle
            updateMotionScores(0f, 0f)
            val motion = getAverageMotion()
            return Pair(currentState, motion)
        }

        val (armMotion, handMotion) = computeMotionScores(keypoints)
        updateMotionScores(armMotion, handMotion)
        
        val state = updateState()
        val avgMotion = getAverageMotion()
        return Pair(state, avgMotion)
    }

    /**
     * Compute separate motion scores for arms and hands.
     * Returns Pair(armMotion, handMotion) for pattern-based detection.
     */
    private fun computeMotionScores(current: FloatArray): Pair<Float, Float> {
        val previous = previousKeypoints
        if (previous == null) {
            // First frame - initialize and return 0 (no motion yet)
            previousKeypoints = current.clone()
            return Pair(0f, 0f)
        }
        
        // Compute arm motion (shoulders and elbows)
        var armMotion = 0f
        var armValidPoints = 0
        for (i in armKeypointIndices) {
            if (i < current.size && i < previous.size) {
                if (current[i] != 0f || previous[i] != 0f) {
                    val delta = current[i] - previous[i]
                    armMotion += delta * delta
                    armValidPoints++
                }
            }
        }
        
        // Compute hand motion (all hand keypoints)
        var handMotion = 0f
        var handValidPoints = 0
        // Hand keypoints: indices 50-133 (42 left + 42 right = 84 values)
        for (i in 50 until 134) {
            if (i < current.size && i < previous.size) {
                if (current[i] != 0f || previous[i] != 0f) {
                    val delta = current[i] - previous[i]
                    handMotion += delta * delta
                    handValidPoints++
                }
            }
        }
        
        previousKeypoints = current.clone()
        
        // Normalize and scale
        val armScore = if (armValidPoints > 0) {
            sqrt(armMotion / armValidPoints) * 10f
        } else 0f
        
        val handScore = if (handValidPoints > 0) {
            sqrt(handMotion / handValidPoints) * 10f
        } else 0f
        
        return Pair(armScore, handScore)
    }

    /**
     * Update motion history for both arms and hands using circular buffer.
     */
    private fun updateMotionScores(armScore: Float, handScore: Float) {
        synchronized(motionHistoryLock) {
            // Filter out jitter (very small movements)
            val filteredArm = if (armScore < jitterFilterThreshold) 0f else armScore
            val filteredHand = if (handScore < jitterFilterThreshold) 0f else handScore
            
            // Store combined motion (weighted: arms indicate sign start, hands indicate active signing)
            val combinedMotion = filteredArm * 0.4f + filteredHand * 0.6f
            
            // Use arm motion for arm history, combined for overall
            armMotionHistory[motionHistoryIndex] = filteredArm
            handMotionHistory[motionHistoryIndex] = filteredHand
            
            motionHistoryIndex = (motionHistoryIndex + 1) % motionWindowSize
            if (motionHistorySize < motionWindowSize) {
                motionHistorySize++
            }
        }
    }

    /**
     * Get percentile-based motion (75th percentile) to filter outliers and jitter.
     */
    private fun getPercentileMotion(percentile: Float): Float {
        synchronized(motionHistoryLock) {
            if (motionHistorySize == 0) return 0f
            
            // Collect motion values
            val values = mutableListOf<Float>()
            val count = if (motionHistorySize < motionWindowSize) motionHistorySize else motionWindowSize
            
            for (i in 0 until count) {
                val idx = if (motionHistorySize < motionWindowSize) {
                    i
                } else {
                    (motionHistoryIndex + i) % motionWindowSize
                }
                // Use combined motion (weighted arms + hands)
                val combined = armMotionHistory[idx] * 0.4f + handMotionHistory[idx] * 0.6f
                values.add(combined)
            }
            
            if (values.isEmpty()) return 0f
            
            // Sort and get percentile
            values.sort()
            val percentileIndex = ((values.size - 1) * percentile).toInt()
            return values[percentileIndex]
        }
    }
    
    /**
     * Get average motion for display/debugging.
     */
    private fun getAverageMotion(): Float {
        return getPercentileMotion(0.5f) // Use median for display
    }
    
    /**
     * Get arm and hand motion separately for pattern detection.
     */
    private fun getArmAndHandMotion(): Pair<Float, Float> {
        synchronized(motionHistoryLock) {
            if (motionHistorySize == 0) return Pair(0f, 0f)
            
            val count = if (motionHistorySize < motionWindowSize) motionHistorySize else motionWindowSize
            val armValues = mutableListOf<Float>()
            val handValues = mutableListOf<Float>()
            
            for (i in 0 until count) {
                val idx = if (motionHistorySize < motionWindowSize) {
                    i
                } else {
                    (motionHistoryIndex + i) % motionWindowSize
                }
                armValues.add(armMotionHistory[idx])
                handValues.add(handMotionHistory[idx])
            }
            
            armValues.sort()
            handValues.sort()
            
            val armPercentile = if (armValues.isNotEmpty()) {
                val idx = ((armValues.size - 1) * motionPercentile).toInt()
                armValues[idx]
            } else 0f
            
            val handPercentile = if (handValues.isNotEmpty()) {
                val idx = ((handValues.size - 1) * motionPercentile).toInt()
                handValues[idx]
            } else 0f
            
            return Pair(armPercentile, handPercentile)
        }
    }

    /**
     * Update activity state based on motion patterns.
     * Uses pattern-based detection: arms → hands for sign start, hands → arms → rest for sign end.
     */
    private fun updateState(): ActivityState {
        val (armMotion, handMotion) = getArmAndHandMotion()
        val combinedMotion = armMotion * 0.4f + handMotion * 0.6f
        
        // Pattern-based thresholds
        // Sign start: require arm motion first (arms moving indicates sign preparation)
        // Active signing: require hand motion (hands moving indicates active signing)
        // Sign end: both below threshold (resting)
        val hasArmMotion = armMotion >= armMotionThreshold
        val hasHandMotion = handMotion >= handMotionThreshold
        val hasSignificantMotion = combinedMotion >= handMotionThreshold
        
        // Enhanced debug logging
        if (motionHistorySize % 30 == 0 && motionHistorySize > 0) {
            Log.d(TAG, "Motion: arm=$armMotion, hand=$handMotion, combined=$combinedMotion, " +
                    "state=$currentState, activeFrames=$consecutiveActiveFrames, idleFrames=$consecutiveIdleFrames")
        }
        
        val oldState = currentState

        when (currentState) {
            ActivityState.IDLE -> {
                // Sign start pattern: arms start moving first, then hands
                // Require either: (1) both arm and hand motion, or (2) sustained hand motion
                if (hasSignificantMotion && (hasArmMotion || hasHandMotion)) {
                    consecutiveActiveFrames++
                    consecutiveIdleFrames = 0
                    
                    if (consecutiveActiveFrames >= activeFramesRequired) {
                        currentState = ActivityState.ACTIVE
                        Log.i(TAG, "State: IDLE → ACTIVE (arm=$armMotion, hand=$handMotion)")
                    } else {
                        currentState = ActivityState.TRANSITION
                        if (oldState != ActivityState.TRANSITION) {
                            Log.d(TAG, "State: IDLE → TRANSITION (arm=$armMotion, hand=$handMotion, frames: $consecutiveActiveFrames/$activeFramesRequired)")
                        }
                    }
                } else {
                    consecutiveActiveFrames = 0
                    consecutiveIdleFrames++
                }
            }
            
            ActivityState.ACTIVE -> {
                // Sign end pattern: motion stops (both arms and hands below threshold)
                if (!hasSignificantMotion) {
                    consecutiveIdleFrames++
                    consecutiveActiveFrames = 0
                    
                    if (consecutiveIdleFrames >= idleFramesRequired) {
                        currentState = ActivityState.IDLE
                        Log.i(TAG, "State: ACTIVE → IDLE (arm=$armMotion, hand=$handMotion)")
                    } else {
                        currentState = ActivityState.TRANSITION
                        if (oldState != ActivityState.TRANSITION) {
                            Log.d(TAG, "State: ACTIVE → TRANSITION (arm=$armMotion, hand=$handMotion, frames: $consecutiveIdleFrames/$idleFramesRequired)")
                        }
                    }
                } else {
                    consecutiveIdleFrames = 0
                    consecutiveActiveFrames++
                }
            }
            
            ActivityState.TRANSITION -> {
                if (hasSignificantMotion) {
                    consecutiveActiveFrames++
                    consecutiveIdleFrames = 0
                    if (consecutiveActiveFrames >= activeFramesRequired) {
                        currentState = ActivityState.ACTIVE
                        Log.i(TAG, "State: TRANSITION → ACTIVE (arm=$armMotion, hand=$handMotion)")
                    }
                } else {
                    consecutiveIdleFrames++
                    consecutiveActiveFrames = 0
                    if (consecutiveIdleFrames >= idleFramesRequired) {
                        currentState = ActivityState.IDLE
                        Log.i(TAG, "State: TRANSITION → IDLE (arm=$armMotion, hand=$handMotion)")
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
            // Clear arrays by setting to zero
            armMotionHistory.fill(0f)
            handMotionHistory.fill(0f)
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


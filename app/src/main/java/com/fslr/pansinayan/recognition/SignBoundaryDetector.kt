package com.fslr.pansinayan.recognition

import android.util.Log

/**
 * Detects sign boundaries (start and end) based on activity patterns.
 * 
 * Monitors activity state transitions and motion patterns to identify
 * when individual signs begin and end. This enables inference to be
 * triggered only when complete signs are captured.
 */
class SignBoundaryDetector(
    private val minSignDurationMs: Long = 500L,
    private val maxSignDurationMs: Long = 5000L,
    private val holdPeriodMs: Long = 300L,
    private val motionSpikeMultiplier: Float = 2.0f
) {
    companion object {
        private const val TAG = "SignBoundaryDetector"
    }

    enum class SignState {
        IDLE,           // No sign activity
        SIGN_STARTING,  // Sign may be starting
        SIGN_ACTIVE,    // Sign is actively being performed
        SIGN_ENDING,    // Sign may be ending
        SIGN_COMPLETE   // Sign completed, ready for inference
    }

    private var currentState = SignState.IDLE
    private var signStartTime: Long = 0L
    private var signStartFrame: Int = 0
    private var lastActiveTime: Long = 0L
    private var lastActiveFrame: Int = 0
    private var holdStartTime: Long = 0L
    private var frameCounter: Int = 0
    private var baselineMotion: Float = 0f

    /**
     * Process activity state update and detect sign boundaries.
     * 
     * @param activityState Current activity state from ActivityDetector
     * @param currentMotion Current motion score
     * @param frameIndex Current frame index
     * @return SignState and optional boundary event
     */
    fun processActivity(
        activityState: ActivityDetector.ActivityState,
        currentMotion: Float,
        frameIndex: Int
    ): BoundaryEvent? {
        val currentTime = System.currentTimeMillis()
        frameCounter = frameIndex

        // Update baseline motion (exponential moving average)
        if (baselineMotion == 0f) {
            baselineMotion = currentMotion
        } else {
            baselineMotion = baselineMotion * 0.95f + currentMotion * 0.05f
        }

        val boundaryEvent = when (currentState) {
            SignState.IDLE -> {
                handleIdleState(activityState, currentMotion, currentTime, frameIndex)
            }
            
            SignState.SIGN_STARTING -> {
                handleStartingState(activityState, currentMotion, currentTime, frameIndex)
            }
            
            SignState.SIGN_ACTIVE -> {
                handleActiveState(activityState, currentMotion, currentTime, frameIndex)
            }
            
            SignState.SIGN_ENDING -> {
                handleEndingState(activityState, currentMotion, currentTime, frameIndex)
            }
            
            SignState.SIGN_COMPLETE -> {
                // Should be reset immediately after processing
                null
            }
        }

        return boundaryEvent
    }

    private fun handleIdleState(
        activityState: ActivityDetector.ActivityState,
        motion: Float,
        time: Long,
        frame: Int
    ): BoundaryEvent? {
        if (activityState == ActivityDetector.ActivityState.ACTIVE) {
            // Check for motion spike (sign start indicator)
            val isSpike = motion >= baselineMotion * motionSpikeMultiplier
            
            if (isSpike || activityState == ActivityDetector.ActivityState.ACTIVE) {
                signStartTime = time
                signStartFrame = frame
                lastActiveTime = time
                lastActiveFrame = frame
                currentState = SignState.SIGN_STARTING
                Log.d(TAG, "Sign STARTING detected at frame $frame (motion: $motion)")
                return BoundaryEvent.SignStart(signStartFrame, signStartTime)
            }
        }
        return null
    }

    private fun handleStartingState(
        activityState: ActivityDetector.ActivityState,
        motion: Float,
        time: Long,
        frame: Int
    ): BoundaryEvent? {
        if (activityState == ActivityDetector.ActivityState.ACTIVE) {
            // Motion sustained - sign is active
            lastActiveTime = time
            lastActiveFrame = frame
            currentState = SignState.SIGN_ACTIVE
            Log.d(TAG, "Sign ACTIVE at frame $frame")
        } else if (activityState == ActivityDetector.ActivityState.IDLE) {
            // Activity stopped too quickly - false start
            currentState = SignState.IDLE
            Log.d(TAG, "False start - returning to IDLE")
        }
        return null
    }

    private fun handleActiveState(
        activityState: ActivityDetector.ActivityState,
        motion: Float,
        time: Long,
        frame: Int
    ): BoundaryEvent? {
        // Check for max duration (force completion)
        val signDuration = time - signStartTime
        if (signDuration >= maxSignDurationMs) {
            Log.d(TAG, "Sign max duration reached - forcing completion")
            currentState = SignState.SIGN_COMPLETE
            return BoundaryEvent.SignEnd(lastActiveFrame, time, true)
        }

        if (activityState == ActivityDetector.ActivityState.ACTIVE) {
            lastActiveTime = time
            lastActiveFrame = frame
        } else if (activityState == ActivityDetector.ActivityState.IDLE) {
            // Activity stopped - sign may be ending
            if (holdStartTime == 0L) {
                holdStartTime = time
            }
            
            val holdDuration = time - holdStartTime
            if (holdDuration >= holdPeriodMs) {
                // Hold period completed - sign ended
                val signDuration = time - signStartTime
                if (signDuration >= minSignDurationMs) {
                    currentState = SignState.SIGN_COMPLETE
                    Log.d(TAG, "Sign END detected at frame $frame (duration: ${signDuration}ms)")
                    return BoundaryEvent.SignEnd(lastActiveFrame, time, false)
                } else {
                    // Too short - false sign
                    Log.d(TAG, "Sign too short (${signDuration}ms) - ignoring")
                    reset()
                }
            } else {
                currentState = SignState.SIGN_ENDING
            }
        }
        return null
    }

    private fun handleEndingState(
        activityState: ActivityDetector.ActivityState,
        motion: Float,
        time: Long,
        frame: Int
    ): BoundaryEvent? {
        if (activityState == ActivityDetector.ActivityState.ACTIVE) {
            // Activity resumed - sign continues
            holdStartTime = 0L
            lastActiveTime = time
            lastActiveFrame = frame
            currentState = SignState.SIGN_ACTIVE
            Log.d(TAG, "Sign resumed - back to ACTIVE")
        } else {
            // Still idle - check hold period
            if (holdStartTime == 0L) {
                holdStartTime = time
            }
            
            val holdDuration = time - holdStartTime
            if (holdDuration >= holdPeriodMs) {
                val signDuration = time - signStartTime
                if (signDuration >= minSignDurationMs) {
                    currentState = SignState.SIGN_COMPLETE
                    Log.d(TAG, "Sign END detected at frame $frame (duration: ${signDuration}ms)")
                    return BoundaryEvent.SignEnd(lastActiveFrame, time, false)
                } else {
                    reset()
                }
            }
        }
        return null
    }

    /**
     * Get current sign state.
     */
    fun getState(): SignState = currentState

    /**
     * Get sign start frame (if sign is active or completed).
     * Preserves the start frame even after reset to allow inference to access it.
     */
    fun getSignStartFrame(): Int? = if (currentState != SignState.IDLE) signStartFrame else null

    /**
     * Get sign end frame (if sign is completed).
     * Preserves the end frame even if state has changed, as long as we have a valid sign.
     */
    fun getSignEndFrame(): Int? = if (currentState == SignState.SIGN_COMPLETE) lastActiveFrame else null

    /**
     * Reset detector after processing a completed sign.
     */
    fun reset() {
        currentState = SignState.IDLE
        signStartTime = 0L
        signStartFrame = 0
        lastActiveTime = 0L
        lastActiveFrame = 0
        holdStartTime = 0L
    }

    /**
     * Reset to IDLE state (for explicit reset).
     */
    fun resetToIdle() {
        reset()
        Log.d(TAG, "Boundary detector reset to IDLE")
    }
}

/**
 * Events emitted when sign boundaries are detected.
 */
sealed class BoundaryEvent {
    data class SignStart(val frameIndex: Int, val timestamp: Long) : BoundaryEvent()
    data class SignEnd(val frameIndex: Int, val timestamp: Long, val forced: Boolean) : BoundaryEvent()
}


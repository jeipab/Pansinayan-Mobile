package com.fslr.pansinayan.recognition

import android.util.Log

/**
 * Controls when inference should be triggered.
 * 
 * Manages inference triggering based on sign boundary events,
 * with fallback mechanisms to ensure signs aren't missed.
 */
class InferenceTrigger(
    private val cooldownMs: Long = 500L,
    private val maxActiveDurationMs: Long = 5000L
) {
    companion object {
        private const val TAG = "InferenceTrigger"
    }

    private var lastInferenceTime: Long = 0L
    private var signStartTime: Long = 0L
    private var lastActiveFrame: Int = 0

    /**
     * Check if inference should be triggered based on boundary event.
     * 
     * @param event Boundary event (SignStart or SignEnd)
     * @param currentFrame Current frame index
     * @return TriggerReason if inference should run, null otherwise
     */
    fun shouldTrigger(event: BoundaryEvent, currentFrame: Int): TriggerReason? {
        val currentTime = System.currentTimeMillis()

        return when (event) {
            is BoundaryEvent.SignStart -> {
                signStartTime = event.timestamp
                lastActiveFrame = event.frameIndex
                null // Don't trigger on start, wait for end
            }
            
            is BoundaryEvent.SignEnd -> {
                // Check cooldown
                val timeSinceLastInference = currentTime - lastInferenceTime
                if (timeSinceLastInference < cooldownMs) {
                    Log.d(TAG, "Inference cooldown active (${timeSinceLastInference}ms < ${cooldownMs}ms)")
                    return null
                }

                lastInferenceTime = currentTime
                lastActiveFrame = event.frameIndex
                
                TriggerReason.SignComplete(
                    signEndFrame = event.frameIndex,
                    forced = event.forced
                )
            }
        }
    }

    /**
     * Check if inference should be triggered due to long active period.
     * Fallback mechanism to prevent missing very long signs.
     * 
     * @param isActive Whether sign is currently active
     * @param currentFrame Current frame index
     * @return TriggerReason if inference should run, null otherwise
     */
    fun checkLongActivePeriod(isActive: Boolean, currentFrame: Int): TriggerReason? {
        if (!isActive) {
            signStartTime = 0L
            return null
        }

        val currentTime = System.currentTimeMillis()
        
        if (signStartTime == 0L) {
            signStartTime = currentTime
            lastActiveFrame = currentFrame
            return null
        }

        val activeDuration = currentTime - signStartTime
        if (activeDuration >= maxActiveDurationMs) {
            // Force inference for long active period
            val reason = TriggerReason.LongActivePeriod(
                signStartFrame = lastActiveFrame - (maxActiveDurationMs / 33).toInt(), // Approximate
                signEndFrame = currentFrame
            )
            signStartTime = currentTime // Reset to prevent immediate re-trigger
            lastInferenceTime = currentTime
            Log.d(TAG, "Long active period trigger (${activeDuration}ms)")
            return reason
        }

        return null
    }

    /**
     * Reset trigger state.
     */
    fun reset() {
        lastInferenceTime = 0L
        signStartTime = 0L
        lastActiveFrame = 0
    }
}

/**
 * Reasons for triggering inference.
 */
sealed class TriggerReason {
    data class SignComplete(
        val signEndFrame: Int,
        val forced: Boolean
    ) : TriggerReason()
    
    data class LongActivePeriod(
        val signStartFrame: Int,
        val signEndFrame: Int
    ) : TriggerReason()
}


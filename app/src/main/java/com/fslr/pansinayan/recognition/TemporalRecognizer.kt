package com.fslr.pansinayan.recognition

import android.util.Log
import java.util.*

/**
 * Implements temporal recognition logic to detect stable sign predictions.
 * 
 * Responsibilities:
 * - Track prediction history over time
 * - Detect stable predictions (same label for N consecutive frames)
 * - Detect sign transitions (label change with high confidence)
 * - Filter noise and jitter
 * - Emit recognized signs only when stable
 * 
 * Algorithm:
 * 1. Maintain history of recent predictions
 * 2. Check if last N predictions are:
 *    - Same gloss ID (stability)
 *    - Above confidence threshold (quality)
 * 3. If yes, emit the recognized sign
 * 4. Implement cooldown to avoid duplicate detections
 * 
 * Usage:
 *   val recognizer = TemporalRecognizer(stabilityFrames = 5, confidenceThreshold = 0.6f)
 *   val result = recognizer.processNewPrediction(glossId, confidence)
 *   if (result != null) {
 *       // Stable sign detected!
 *       displaySign(result.glossId, result.label)
 *   }
 */
class TemporalRecognizer(
    private val stabilityThreshold: Int = 5,  // Number of consecutive stable frames required
    private val confidenceThreshold: Float = 0.6f,  // Minimum confidence to consider
    private val cooldownMs: Long = 1000  // Cooldown period after detecting a sign (ms)
) {
    companion object {
        private const val TAG = "TemporalRecognizer"
        private const val MAX_HISTORY_SIZE = 20
    }

    // History of recent predictions: (glossId, confidence, timestamp)
    private val predictionHistory = LinkedList<Triple<Int, Float, Long>>()
    
    // Last emitted sign info (for cooldown)
    private var lastEmittedGlossId: Int? = null
    private var lastEmittedTime: Long = 0L
    
    // Statistics
    private var totalPredictions = 0
    private var stableDetections = 0

    /**
     * Process a new prediction from the model.
     * 
     * @param glossId Predicted gloss class ID
     * @param confidence Prediction confidence (0.0 - 1.0)
     * @param timestamp Optional timestamp (default: current time)
     * @return RecognitionEvent if a stable sign is detected, null otherwise
     */
    fun processNewPrediction(
        glossId: Int,
        confidence: Float,
        timestamp: Long = System.currentTimeMillis()
    ): RecognitionEvent? {
        totalPredictions++

        // Add to history
        predictionHistory.add(Triple(glossId, confidence, timestamp))
        
        // Maintain max history size
        while (predictionHistory.size > MAX_HISTORY_SIZE) {
            predictionHistory.removeFirst()
        }

        // Check if we have enough history
        if (predictionHistory.size < stabilityThreshold) {
            return null
        }

        // Get recent predictions (last N frames)
        val recentPredictions = predictionHistory.takeLast(stabilityThreshold)

        // Check stability criteria:
        // 1. All predictions have the same gloss ID
        val allSameGloss = recentPredictions.all { it.first == glossId }
        
        // 2. All predictions meet confidence threshold
        val allConfident = recentPredictions.all { it.second >= confidenceThreshold }

        // If both criteria met, this is a stable detection
        if (allSameGloss && allConfident) {
            // Check cooldown period to avoid duplicate emissions
            val timeSinceLastEmit = timestamp - lastEmittedTime
            val isSameAsLastEmitted = (glossId == lastEmittedGlossId)
            
            if (!isSameAsLastEmitted || timeSinceLastEmit >= cooldownMs) {
                // Emit this recognition!
                stableDetections++
                lastEmittedGlossId = glossId
                lastEmittedTime = timestamp
                
                val avgConfidence = recentPredictions.map { it.second }.average().toFloat()
                
                Log.i(TAG, "Stable detection #$stableDetections: Gloss $glossId (conf: $avgConfidence)")
                
                return RecognitionEvent(
                    glossId = glossId,
                    confidence = avgConfidence,
                    timestamp = timestamp,
                    sequenceNumber = stableDetections
                )
            } else {
                // In cooldown period for same sign
                Log.d(TAG, "Suppressed duplicate: Gloss $glossId (cooldown active)")
            }
        }

        return null
    }

    /**
     * Reset temporal state.
     * Call this when starting a new recognition session.
     */
    fun reset() {
        predictionHistory.clear()
        lastEmittedGlossId = null
        lastEmittedTime = 0L
        Log.i(TAG, "Temporal recognizer reset")
    }

    /**
     * Get current stability status (for debugging/UI).
     * 
     * @return Triple(currentGlossId, consecutiveCount, isStable)
     */
    fun getStabilityStatus(): Triple<Int?, Int, Boolean> {
        if (predictionHistory.isEmpty()) {
            return Triple(null, 0, false)
        }

        val lastGlossId = predictionHistory.last().first
        var consecutiveCount = 0

        // Count consecutive predictions of the same gloss from the end
        for (i in predictionHistory.indices.reversed()) {
            if (predictionHistory[i].first == lastGlossId) {
                consecutiveCount++
            } else {
                break
            }
        }

        val isStable = consecutiveCount >= stabilityThreshold &&
                       predictionHistory.takeLast(stabilityThreshold).all { it.second >= confidenceThreshold }

        return Triple(lastGlossId, consecutiveCount, isStable)
    }

    /**
     * Get statistics for debugging.
     */
    fun getStats(): String {
        val detectionRate = if (totalPredictions > 0) {
            (stableDetections.toFloat() / totalPredictions * 100)
        } else 0f
        
        return "Total: $totalPredictions, Stable: $stableDetections (${String.format("%.1f", detectionRate)}%)"
    }

    /**
     * Adjust thresholds dynamically (for tuning).
     */
    fun updateThresholds(newStabilityThreshold: Int? = null, newConfidenceThreshold: Float? = null) {
        // Note: This modifies companion object values in a real implementation
        // For now, just log the change request
        Log.i(TAG, "Threshold update requested: stability=$newStabilityThreshold, confidence=$newConfidenceThreshold")
    }
}

/**
 * Represents a recognized sign event.
 */
data class RecognitionEvent(
    val glossId: Int,
    val confidence: Float,
    val timestamp: Long,
    val sequenceNumber: Int
)


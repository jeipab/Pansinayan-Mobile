package com.fslr.pansinayan.inference

import android.util.Log
import kotlin.math.exp

/**
 * CTC decoder for continuous sign language recognition.
 * Implements greedy decoding with per-frame category predictions.
 */
class CTCDecoder(
    private val blankId: Int = 105,
    private val numCtcClasses: Int = 106,
    private val numCategories: Int = 10
) {
    companion object {
        private const val TAG = "CTCDecoder"
    }

    /**
     * Decode CTC predictions with per-sign categories.
     * 
     * @param ctcLogProbs [T * 106] flattened log probabilities
     * @param categoryLogits [T * 10] flattened raw logits
     * @param numFrames Total frames in sequence
     * @return List of decoded sign predictions
     */
    fun decode(
        ctcLogProbs: FloatArray,
        categoryLogits: FloatArray,
        numFrames: Int
    ): List<SignPrediction> {
        // Step 1: Greedy decode CTC
        val decodedGlosses = greedyDecode(ctcLogProbs, numFrames)
        
        if (decodedGlosses.isEmpty()) {
            Log.d(TAG, "No signs decoded (all blanks)")
            return emptyList()
        }
        
        // Step 2: Softmax category logits
        val categoryProbs = softmax(categoryLogits, numFrames, numCategories)
        
        // Step 3: Assign categories to signs
        val categoryResults = assignCategories(decodedGlosses, categoryProbs, numFrames)
        
        // Step 4: Build predictions
        val framesPerSign = numFrames.toFloat() / decodedGlosses.size
        val predictions = decodedGlosses.mapIndexed { idx, glossId ->
            val startFrame = (idx * framesPerSign).toInt()
            val endFrame = ((idx + 1) * framesPerSign).toInt().coerceAtMost(numFrames)
            val (categoryId, categoryConf) = categoryResults[idx]
            
            SignPrediction(
                glossId = glossId,
                categoryId = categoryId,
                startFrame = startFrame,
                endFrame = endFrame,
                categoryConfidence = categoryConf
            )
        }
        
        Log.d(TAG, "Decoded ${predictions.size} signs: ${predictions.map { it.glossId }}")
        return predictions
    }

    /**
     * Greedy CTC decode: argmax at each timestep, then collapse.
     */
    private fun greedyDecode(ctcLogProbs: FloatArray, numFrames: Int): List<Int> {
        // Step 1: Argmax at each timestep
        val bestPath = IntArray(numFrames)
        for (t in 0 until numFrames) {
            var maxIdx = 0
            var maxVal = ctcLogProbs[t * numCtcClasses]
            for (c in 1 until numCtcClasses) {
                val value = ctcLogProbs[t * numCtcClasses + c]
                if (value > maxVal) {
                    maxVal = value
                    maxIdx = c
                }
            }
            bestPath[t] = maxIdx
        }
        
        // Step 2: CTC collapse
        return collapseCtc(bestPath.toList())
    }

    /**
     * CTC collapsing: remove consecutive duplicates, then remove blanks.
     * Example: [1, 1, 105, 2, 2, 105, 3] → [1, 2, 3]
     */
    private fun collapseCtc(sequence: List<Int>): List<Int> {
        // Remove consecutive duplicates
        val collapsed = mutableListOf<Int>()
        for (token in sequence) {
            if (collapsed.isEmpty() || token != collapsed.last()) {
                collapsed.add(token)
            }
        }
        
        // Remove blanks
        return collapsed.filter { it != blankId }
    }

    /**
     * Apply softmax to category logits per frame.
     */
    private fun softmax(logits: FloatArray, numFrames: Int, numClasses: Int): FloatArray {
        val probs = FloatArray(numFrames * numClasses)
        
        for (t in 0 until numFrames) {
            val offset = t * numClasses
            
            // Find max for numerical stability
            var maxLogit = logits[offset]
            for (c in 1 until numClasses) {
                maxLogit = maxOf(maxLogit, logits[offset + c])
            }
            
            // Compute exp and sum
            var sumExp = 0f
            for (c in 0 until numClasses) {
                val expVal = exp(logits[offset + c] - maxLogit)
                probs[offset + c] = expVal
                sumExp += expVal
            }
            
            // Normalize
            for (c in 0 until numClasses) {
                probs[offset + c] /= sumExp
            }
        }
        
        return probs
    }

    /**
     * Assign categories to signs using frame distribution.
     * Averages probabilities over each sign's frame range.
     */
    private fun assignCategories(
        decodedGlosses: List<Int>,
        categoryProbs: FloatArray,
        numFrames: Int
    ): List<Pair<Int, Float>> {
        val framesPerSign = numFrames.toFloat() / decodedGlosses.size
        
        return decodedGlosses.mapIndexed { idx, _ ->
            val startFrame = (idx * framesPerSign).toInt()
            val endFrame = ((idx + 1) * framesPerSign).toInt().coerceAtMost(numFrames)
            
            // Average probabilities over sign's frame range
            val avgProbs = FloatArray(numCategories) { 0f }
            for (t in startFrame until endFrame) {
                for (c in 0 until numCategories) {
                    avgProbs[c] += categoryProbs[t * numCategories + c]
                }
            }
            val numFramesInSign = endFrame - startFrame
            for (c in 0 until numCategories) {
                avgProbs[c] /= numFramesInSign
            }
            
            // Argmax for final category
            var predCat = 0
            var catConf = avgProbs[0]
            for (c in 1 until numCategories) {
                if (avgProbs[c] > catConf) {
                    catConf = avgProbs[c]
                    predCat = c
                }
            }
            
            Pair(predCat, catConf)
        }
    }
}


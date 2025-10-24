package com.fslr.pansinayan.inference

import android.util.Log
import com.fslr.pansinayan.utils.LabelMapper

/**
 * CTC decoder for continuous sign language recognition.
 * 
 * Implements greedy CTC decoding algorithm:
 * 1. Apply argmax across classes for each time step
 * 2. Collapse consecutive duplicates
 * 3. Remove blank tokens (index = num_classes)
 * 4. Map remaining indices to gloss strings
 * 
 * Input: Raw logits from CTC model [T, num_classes + 1]
 * Output: Decoded gloss sequence as string
 * 
 * Usage:
 *   val decoder = CTCDecoder(labelMapper)
 *   val decoded = decoder.decode(logits)
 *   println("Recognized: $decoded")
 */
class CTCDecoder(
    private val labelMapper: LabelMapper
) {
    companion object {
        private const val TAG = "CTCDecoder"
        private const val BLANK_TOKEN_INDEX = 105  // Last index is blank token
        private const val OUTPUT_CLASSES = 106  // 105 glosses + 1 blank
    }

    // Statistics
    private var totalDecodings = 0
    private var successfulDecodings = 0
    private var totalDecodingTimeMs = 0L

    /**
     * Decode CTC logits to gloss sequence using greedy algorithm.
     * 
     * @param logits Flattened logits array [T * (num_classes + 1)]
     * @param sequenceLength Actual sequence length T (≤ 300)
     * @return Decoded gloss sequence as space-separated string
     */
    fun decode(logits: FloatArray, sequenceLength: Int = 300): String {
        val startTime = System.currentTimeMillis()
        totalDecodings++

        try {
            // Step 1: Apply argmax across classes for each time step
            val argmaxSequence = mutableListOf<Int>()
            
            for (t in 0 until sequenceLength) {
                val startIdx = t * OUTPUT_CLASSES
                val endIdx = startIdx + OUTPUT_CLASSES
                
                if (endIdx <= logits.size) {
                    val timeStepLogits = logits.sliceArray(startIdx until endIdx)
                    val argmaxIdx = timeStepLogits.indices.maxByOrNull { timeStepLogits[it] } ?: BLANK_TOKEN_INDEX
                    argmaxSequence.add(argmaxIdx)
                } else {
                    // Handle edge case where logits array is shorter than expected
                    argmaxSequence.add(BLANK_TOKEN_INDEX)
                }
            }

            // Step 2: Collapse consecutive duplicates
            val collapsedSequence = collapseDuplicates(argmaxSequence)

            // Step 3: Remove blank tokens
            val filteredSequence = collapsedSequence.filter { it != BLANK_TOKEN_INDEX }

            // Step 4: Map indices to gloss strings
            val glossSequence = filteredSequence.map { glossId ->
                labelMapper.getGlossLabel(glossId)
            }

            val decodedString = glossSequence.joinToString(" ")
            
            val decodingTime = System.currentTimeMillis() - startTime
            totalDecodingTimeMs += decodingTime
            
            if (decodedString.isNotEmpty()) {
                successfulDecodings++
                Log.d(TAG, "Decoded: '$decodedString' (${decodingTime}ms)")
            } else {
                Log.d(TAG, "Empty sequence decoded (${decodingTime}ms)")
            }

            return decodedString

        } catch (e: Exception) {
            Log.e(TAG, "CTC decoding failed", e)
            return ""
        }
    }

    /**
     * Collapse consecutive duplicate tokens in the sequence.
     * 
     * Example: [1, 1, 1, 2, 2, 3, 3, 3, 3] → [1, 2, 3]
     */
    private fun collapseDuplicates(sequence: List<Int>): List<Int> {
        if (sequence.isEmpty()) return emptyList()

        val collapsed = mutableListOf<Int>()
        var lastToken = sequence[0]
        collapsed.add(lastToken)

        for (i in 1 until sequence.size) {
            if (sequence[i] != lastToken) {
                collapsed.add(sequence[i])
                lastToken = sequence[i]
            }
        }

        return collapsed
    }

    /**
     * Decode with confidence scoring (optional enhancement).
     * Returns both decoded string and average confidence.
     */
    fun decodeWithConfidence(logits: FloatArray, sequenceLength: Int = 300): Pair<String, Float> {
        val startTime = System.currentTimeMillis()
        totalDecodings++

        try {
            // Step 1: Apply argmax and collect confidences
            val argmaxSequence = mutableListOf<Int>()
            val confidences = mutableListOf<Float>()
            
            for (t in 0 until sequenceLength) {
                val startIdx = t * OUTPUT_CLASSES
                val endIdx = startIdx + OUTPUT_CLASSES
                
                if (endIdx <= logits.size) {
                    val timeStepLogits = logits.sliceArray(startIdx until endIdx)
                    val argmaxIdx = timeStepLogits.indices.maxByOrNull { timeStepLogits[it] } ?: BLANK_TOKEN_INDEX
                    val confidence = timeStepLogits[argmaxIdx]
                    
                    argmaxSequence.add(argmaxIdx)
                    confidences.add(confidence)
                } else {
                    argmaxSequence.add(BLANK_TOKEN_INDEX)
                    confidences.add(0f)
                }
            }

            // Step 2: Collapse consecutive duplicates (keep max confidence for each token)
            val collapsedSequence = mutableListOf<Int>()
            val collapsedConfidences = mutableListOf<Float>()
            
            if (argmaxSequence.isNotEmpty()) {
                var lastToken = argmaxSequence[0]
                var maxConfidence = confidences[0]
                collapsedSequence.add(lastToken)
                
                for (i in 1 until argmaxSequence.size) {
                    if (argmaxSequence[i] != lastToken) {
                        collapsedConfidences.add(maxConfidence)
                        collapsedSequence.add(argmaxSequence[i])
                        lastToken = argmaxSequence[i]
                        maxConfidence = confidences[i]
                    } else {
                        maxConfidence = maxOf(maxConfidence, confidences[i])
                    }
                }
                collapsedConfidences.add(maxConfidence)
            }

            // Step 3: Remove blank tokens
            val filteredSequence = mutableListOf<Int>()
            val filteredConfidences = mutableListOf<Float>()
            
            for (i in collapsedSequence.indices) {
                if (collapsedSequence[i] != BLANK_TOKEN_INDEX) {
                    filteredSequence.add(collapsedSequence[i])
                    filteredConfidences.add(collapsedConfidences[i])
                }
            }

            // Step 4: Map to gloss strings
            val glossSequence = filteredSequence.map { glossId ->
                labelMapper.getGlossLabel(glossId)
            }

            val decodedString = glossSequence.joinToString(" ")
            val avgConfidence = if (filteredConfidences.isNotEmpty()) {
                filteredConfidences.average().toFloat()
            } else 0f

            val decodingTime = System.currentTimeMillis() - startTime
            totalDecodingTimeMs += decodingTime
            
            if (decodedString.isNotEmpty()) {
                successfulDecodings++
                Log.d(TAG, "Decoded with confidence: '$decodedString' (conf: ${String.format("%.3f", avgConfidence)}, ${decodingTime}ms)")
            }

            return Pair(decodedString, avgConfidence)

        } catch (e: Exception) {
            Log.e(TAG, "CTC decoding with confidence failed", e)
            return Pair("", 0f)
        }
    }

    /**
     * Reset decoder statistics.
     */
    fun reset() {
        totalDecodings = 0
        successfulDecodings = 0
        totalDecodingTimeMs = 0L
        Log.i(TAG, "CTC decoder statistics reset")
    }

    /**
     * Get decoder statistics for debugging.
     */
    fun getStats(): String {
        val successRate = if (totalDecodings > 0) {
            (successfulDecodings.toFloat() / totalDecodings * 100)
        } else 0f
        
        val avgTime = if (totalDecodings > 0) {
            totalDecodingTimeMs / totalDecodings
        } else 0L
        
        return "Decodings: $successfulDecodings/$totalDecodings (${String.format("%.1f", successRate)}%), avg: ${avgTime}ms"
    }

    /**
     * Get detailed statistics.
     */
    fun getDetailedStats(): CTCStats {
        val successRate = if (totalDecodings > 0) {
            (successfulDecodings.toFloat() / totalDecodings * 100)
        } else 0f
        
        val avgTime = if (totalDecodings > 0) {
            totalDecodingTimeMs / totalDecodings
        } else 0L
        
        return CTCStats(
            totalDecodings = totalDecodings,
            successfulDecodings = successfulDecodings,
            successRate = successRate,
            avgDecodingTimeMs = avgTime
        )
    }
}

/**
 * Statistics for CTC decoder.
 */
data class CTCStats(
    val totalDecodings: Int,
    val successfulDecodings: Int,
    val successRate: Float,
    val avgDecodingTimeMs: Long
)

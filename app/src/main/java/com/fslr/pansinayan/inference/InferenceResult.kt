package com.fslr.pansinayan.inference

/**
 * Data class to hold inference results.
 */
data class InferenceResult(
    val glossPrediction: Int,
    val glossConfidence: Float,
    val glossProbabilities: FloatArray,
    val glossTop5: List<Pair<Int, Float>>,
    val categoryPrediction: Int,
    val categoryConfidence: Float,
    val categoryProbabilities: FloatArray,
    val inferenceTimeMs: Long,
    val sequenceLength: Int,
    // CTC-specific fields
    val isCTC: Boolean = false,
    val ctcPredictions: List<SignPrediction>? = null
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as InferenceResult

        if (glossPrediction != other.glossPrediction) return false
        if (glossConfidence != other.glossConfidence) return false
        if (!glossProbabilities.contentEquals(other.glossProbabilities)) return false
        if (categoryPrediction != other.categoryPrediction) return false
        if (categoryConfidence != other.categoryConfidence) return false
        if (!categoryProbabilities.contentEquals(other.categoryProbabilities)) return false
        if (inferenceTimeMs != other.inferenceTimeMs) return false

        return true
    }

    override fun hashCode(): Int {
        var result = glossPrediction
        result = 31 * result + glossConfidence.hashCode()
        result = 31 * result + glossProbabilities.contentHashCode()
        result = 31 * result + categoryPrediction
        result = 31 * result + categoryConfidence.hashCode()
        result = 31 * result + categoryProbabilities.contentHashCode()
        result = 31 * result + inferenceTimeMs.hashCode()
        return result
    }

    override fun toString(): String {
        return "InferenceResult(gloss=$glossPrediction, conf=${String.format("%.3f", glossConfidence)}, time=${inferenceTimeMs}ms)"
    }
}


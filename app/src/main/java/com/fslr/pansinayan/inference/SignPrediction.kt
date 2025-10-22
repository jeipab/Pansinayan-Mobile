package com.fslr.pansinayan.inference

/**
 * Represents a decoded sign with temporal boundaries.
 */
data class SignPrediction(
    val glossId: Int,
    val categoryId: Int,
    val startFrame: Int,
    val endFrame: Int,
    val categoryConfidence: Float
)


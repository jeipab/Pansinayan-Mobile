package com.fslr.pansinayan.database

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Entity representing a single recognition event in the history.
 * 
 * This table stores all sign recognitions for later review and analysis.
 * Includes timestamp, predicted gloss with confidence, predicted category with confidence.
 */
@Entity(tableName = "recognition_history")
data class RecognitionHistory(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val timestamp: Long,
    val glossLabel: String,
    val glossConfidence: Float,
    val categoryLabel: String,
    val categoryConfidence: Float,
    val occlusionStatus: String, // "Occluded" or "Not Occluded"
    val modelUsed: String // "Transformer" or "GRU"
)


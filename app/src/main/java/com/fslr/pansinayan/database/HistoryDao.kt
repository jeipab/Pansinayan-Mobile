package com.fslr.pansinayan.database

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

/**
 * Data Access Object for recognition history.
 * 
 * Provides methods to insert, query, and delete history records.
 */
@Dao
interface HistoryDao {
    /**
     * Insert a new recognition event into history.
     */
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(history: RecognitionHistory)

    /**
     * Get all history entries for a specific model, ordered by most recent first.
     * Returns a Flow for real-time updates.
     */
    @Query("SELECT * FROM recognition_history WHERE modelUsed = :modelName ORDER BY timestamp DESC")
    fun getHistoryByModel(modelName: String): Flow<List<RecognitionHistory>>

    /**
     * Get all history entries, ordered by most recent first.
     */
    @Query("SELECT * FROM recognition_history ORDER BY timestamp DESC")
    suspend fun getAllHistory(): List<RecognitionHistory>

    /**
     * Clear all history records.
     */
    @Query("DELETE FROM recognition_history")
    suspend fun clearAllHistory()
}


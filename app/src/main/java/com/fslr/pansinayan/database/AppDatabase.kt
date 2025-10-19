package com.fslr.pansinayan.database

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

/**
 * Room database for the Sign Language Recognition app.
 * 
 * Manages persistent storage of recognition history.
 * 
 * Usage:
 *   val db = AppDatabase.getDatabase(context)
 *   val historyDao = db.historyDao()
 */
@Database(entities = [RecognitionHistory::class], version = 1, exportSchema = false)
abstract class AppDatabase : RoomDatabase() {
    abstract fun historyDao(): HistoryDao

    companion object {
        @Volatile
        private var INSTANCE: AppDatabase? = null

        /**
         * Get database instance (singleton pattern).
         */
        fun getDatabase(context: Context): AppDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "slr_history_database"
                ).build()
                INSTANCE = instance
                instance
            }
        }
    }
}


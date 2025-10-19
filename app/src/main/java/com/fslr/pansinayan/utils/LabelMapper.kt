package com.fslr.pansinayan.utils

import android.content.Context
import android.util.Log
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.io.InputStreamReader

/**
 * Maps gloss and category IDs to human-readable labels.
 * 
 * Loads label_mapping.json from assets and provides lookup functions.
 * 
 * Usage:
 *   val mapper = LabelMapper(context)
 *   val label = mapper.getGlossLabel(42)  // "WEDNESDAY"
 *   val category = mapper.getCategoryLabel(4)  // "DAYS"
 */
class LabelMapper(private val context: Context) {
    companion object {
        private const val TAG = "LabelMapper"
        private const val LABEL_FILE = "label_mapping.json"
    }

    private val glossMapping = mutableMapOf<Int, String>()
    private val categoryMapping = mutableMapOf<Int, String>()

    init {
        loadMappings()
    }

    /**
     * Load label mappings from assets/label_mapping.json.
     */
    private fun loadMappings() {
        try {
            val inputStream = context.assets.open(LABEL_FILE)
            val reader = InputStreamReader(inputStream)
            
            val gson = Gson()
            val type = object : TypeToken<LabelMappingData>() {}.type
            val data: LabelMappingData = gson.fromJson(reader, type)

            // Convert String keys to Int
            data.glosses.forEach { (key, value) ->
                glossMapping[key.toInt()] = value
            }
            
            data.categories.forEach { (key, value) ->
                categoryMapping[key.toInt()] = value
            }

            reader.close()

            Log.i(TAG, "Label mappings loaded: ${glossMapping.size} glosses, ${categoryMapping.size} categories")

        } catch (e: Exception) {
            Log.e(TAG, "Failed to load label mappings", e)
            throw RuntimeException("Could not load $LABEL_FILE from assets", e)
        }
    }

    /**
     * Get gloss label by ID.
     * 
     * @param glossId Gloss class ID (0-104)
     * @return Human-readable label (e.g., "GOOD MORNING")
     */
    fun getGlossLabel(glossId: Int): String {
        return glossMapping[glossId] ?: "UNKNOWN_$glossId"
    }

    /**
     * Get category label by ID.
     * 
     * @param categoryId Category class ID (0-9)
     * @return Category name (e.g., "GREETING")
     */
    fun getCategoryLabel(categoryId: Int): String {
        return categoryMapping[categoryId] ?: "UNKNOWN_$categoryId"
    }

    /**
     * Get all gloss labels.
     * @return Map of ID to label
     */
    fun getAllGlosses(): Map<Int, String> {
        return glossMapping.toMap()
    }

    /**
     * Get all category labels.
     * @return Map of ID to category name
     */
    fun getAllCategories(): Map<Int, String> {
        return categoryMapping.toMap()
    }

    /**
     * Search for gloss ID by label (case-insensitive).
     * 
     * @param label Label to search for
     * @return Gloss ID or null if not found
     */
    fun findGlossIdByLabel(label: String): Int? {
        return glossMapping.entries.firstOrNull { 
            it.value.equals(label, ignoreCase = true) 
        }?.key
    }
}

/**
 * Data class for JSON deserialization.
 */
private data class LabelMappingData(
    val glosses: Map<String, String>,
    val categories: Map<String, String>
)


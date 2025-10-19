package com.fslr.pansinayan.utils

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData

/**
 * ModelSelector - Manages model selection and switching between different TFLite models.
 * 
 * This class provides a centralized way to manage which sign language recognition model
 * is currently active in the app. It supports multiple models (Transformer and MediaPipeGRU)
 * and allows dynamic switching during runtime.
 * 
 * Features:
 * - Model enumeration and metadata
 * - Persistent model selection (saved in SharedPreferences)
 * - Observable model changes via LiveData
 * - Model availability checking
 * - Performance tracking
 * 
 * Usage:
 *     val modelSelector = ModelSelector(context)
 *     modelSelector.selectModel(ModelType.TRANSFORMER)
 *     val currentModel = modelSelector.currentModel.value
 *     
 *     // Observe model changes
 *     modelSelector.currentModel.observe(lifecycleOwner) { model ->
 *         // Update UI or reload model
 *     }
 */
class ModelSelector(private val context: Context) {
    
    companion object {
        private const val TAG = "ModelSelector"
        private const val PREFS_NAME = "model_selector_prefs"
        private const val KEY_SELECTED_MODEL = "selected_model"
        
        // Model file paths in assets
        const val TRANSFORMER_MODEL_PATH = "sign_transformer_quant.tflite"
        const val MEDIAPIPE_GRU_MODEL_PATH = "sign_mediapipe_gru_quant.tflite"
        
        // Default model selection
        val DEFAULT_MODEL = ModelType.TRANSFORMER
    }
    
    /**
     * Enum class defining available model types with their characteristics.
     */
    enum class ModelType(
        val displayName: String,
        val modelPath: String,
        val description: String,
        val estimatedSize: String,
        val estimatedLatency: String,
        val accuracy: String,
        val isRecommended: Boolean
    ) {
        TRANSFORMER(
            displayName = "Transformer",
            modelPath = TRANSFORMER_MODEL_PATH,
            description = "Multi-head attention transformer encoder. Best accuracy with reasonable speed.",
            estimatedSize = "~1.5 MB",
            estimatedLatency = "150-250ms",
            accuracy = "★★★★★ (Highest)",
            isRecommended = true
        ),
        
        MEDIAPIPE_GRU(
            displayName = "GRU Baseline",
            modelPath = MEDIAPIPE_GRU_MODEL_PATH,
            description = "Lightweight GRU network. Faster inference with slightly lower accuracy.",
            estimatedSize = "~500 KB",
            estimatedLatency = "50-100ms",
            accuracy = "★★★★☆ (High)",
            isRecommended = false
        );
        
        /**
         * Get a user-friendly summary of the model.
         */
        fun getSummary(): String {
            return """
                $displayName
                Size: $estimatedSize
                Speed: $estimatedLatency
                Accuracy: $accuracy
                ${if (isRecommended) "✓ Recommended" else ""}
            """.trimIndent()
        }
        
        companion object {
            /**
             * Get ModelType from string name.
             */
            fun fromString(name: String): ModelType? {
                return values().find { it.name.equals(name, ignoreCase = true) }
            }
        }
    }
    
    /**
     * Data class to store model performance metrics.
     */
    data class ModelPerformance(
        val modelType: ModelType,
        val avgInferenceTime: Float,
        val minInferenceTime: Float,
        val maxInferenceTime: Float,
        val totalInferences: Int,
        var lastInferenceTime: Float = 0f
    ) {
        /**
         * Update performance metrics with a new inference time.
         */
        fun update(inferenceTime: Float): ModelPerformance {
            return copy(
                avgInferenceTime = (avgInferenceTime * totalInferences + inferenceTime) / (totalInferences + 1),
                minInferenceTime = minOf(minInferenceTime, inferenceTime),
                maxInferenceTime = maxOf(maxInferenceTime, inferenceTime),
                totalInferences = totalInferences + 1,
                lastInferenceTime = inferenceTime
            )
        }
        
        /**
         * Get a formatted string of performance metrics.
         */
        fun getFormattedStats(): String {
            return """
                ${modelType.displayName} Performance:
                • Avg: ${avgInferenceTime.toInt()}ms
                • Min: ${minInferenceTime.toInt()}ms
                • Max: ${maxInferenceTime.toInt()}ms
                • Count: $totalInferences
            """.trimIndent()
        }
    }
    
    // SharedPreferences for persistent storage
    private val prefs: SharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    
    // LiveData for current model (observable by UI)
    private val _currentModel = MutableLiveData<ModelType>()
    val currentModel: LiveData<ModelType> = _currentModel
    
    // Performance tracking for each model
    private val _performanceMap = mutableMapOf<ModelType, ModelPerformance>()
    
    init {
        // Load saved model preference or use default
        val savedModelName = prefs.getString(KEY_SELECTED_MODEL, null)
        val initialModel = savedModelName?.let { ModelType.fromString(it) } ?: DEFAULT_MODEL
        _currentModel.value = initialModel
        
        // Initialize performance trackers
        ModelType.values().forEach { modelType ->
            _performanceMap[modelType] = ModelPerformance(
                modelType = modelType,
                avgInferenceTime = 0f,
                minInferenceTime = Float.MAX_VALUE,
                maxInferenceTime = 0f,
                totalInferences = 0
            )
        }
        
        Log.i(TAG, "ModelSelector initialized with model: ${initialModel.displayName}")
    }
    
    /**
     * Select and activate a specific model.
     * 
     * @param modelType The model type to select
     * @return true if selection was successful, false otherwise
     */
    fun selectModel(modelType: ModelType): Boolean {
        // Check if model file exists
        if (!isModelAvailable(modelType)) {
            Log.e(TAG, "Model file not found: ${modelType.modelPath}")
            return false
        }
        
        // Update current model
        _currentModel.value = modelType
        
        // Save preference
        prefs.edit().putString(KEY_SELECTED_MODEL, modelType.name).apply()
        
        Log.i(TAG, "Model selected: ${modelType.displayName}")
        return true
    }
    
    /**
     * Check if a specific model file is available in assets.
     * 
     * @param modelType The model type to check
     * @return true if model file exists, false otherwise
     */
    fun isModelAvailable(modelType: ModelType): Boolean {
        return try {
            context.assets.open(modelType.modelPath).use { true }
        } catch (e: Exception) {
            Log.w(TAG, "Model not available: ${modelType.modelPath}")
            false
        }
    }
    
    /**
     * Get a list of all available models.
     * 
     * @return List of ModelType that have their files present in assets
     */
    fun getAvailableModels(): List<ModelType> {
        return ModelType.values().filter { isModelAvailable(it) }
    }
    
    /**
     * Get the currently selected model type.
     * 
     * @return Current ModelType
     */
    fun getCurrentModelType(): ModelType {
        return _currentModel.value ?: DEFAULT_MODEL
    }
    
    /**
     * Get the file path of the currently selected model.
     * 
     * @return Model file path in assets
     */
    fun getCurrentModelPath(): String {
        return getCurrentModelType().modelPath
    }
    
    /**
     * Record inference time for performance tracking.
     * 
     * @param inferenceTime Inference time in milliseconds
     */
    fun recordInferenceTime(inferenceTime: Float) {
        val modelType = getCurrentModelType()
        _performanceMap[modelType]?.let { currentPerf ->
            _performanceMap[modelType] = currentPerf.update(inferenceTime)
        }
    }
    
    /**
     * Get performance metrics for a specific model.
     * 
     * @param modelType The model type to get metrics for
     * @return ModelPerformance object or null if not available
     */
    fun getPerformance(modelType: ModelType): ModelPerformance? {
        return _performanceMap[modelType]
    }
    
    /**
     * Get performance metrics for the current model.
     * 
     * @return ModelPerformance object or null if not available
     */
    fun getCurrentPerformance(): ModelPerformance? {
        return getPerformance(getCurrentModelType())
    }
    
    /**
     * Get all performance metrics as a map.
     * 
     * @return Map of ModelType to ModelPerformance
     */
    fun getAllPerformance(): Map<ModelType, ModelPerformance> {
        return _performanceMap.toMap()
    }
    
    /**
     * Reset performance metrics for all models.
     */
    fun resetPerformanceMetrics() {
        ModelType.values().forEach { modelType ->
            _performanceMap[modelType] = ModelPerformance(
                modelType = modelType,
                avgInferenceTime = 0f,
                minInferenceTime = Float.MAX_VALUE,
                maxInferenceTime = 0f,
                totalInferences = 0
            )
        }
        Log.i(TAG, "Performance metrics reset")
    }
    
    /**
     * Get a comparison summary of all models.
     * 
     * @return Formatted string comparing all models
     */
    fun getComparisonSummary(): String {
        val availableModels = getAvailableModels()
        val currentModel = getCurrentModelType()
        
        return buildString {
            appendLine("=== Model Comparison ===")
            appendLine()
            availableModels.forEach { model ->
                val isCurrent = model == currentModel
                val prefix = if (isCurrent) "► " else "  "
                appendLine("${prefix}${model.displayName}${if (model.isRecommended) " ⭐" else ""}")
                appendLine("  ${model.description}")
                appendLine("  ${model.estimatedSize} | ${model.estimatedLatency}")
                
                // Add performance metrics if available
                _performanceMap[model]?.let { perf ->
                    if (perf.totalInferences > 0) {
                        appendLine("  Measured: ${perf.avgInferenceTime.toInt()}ms avg (${perf.totalInferences} runs)")
                    }
                }
                appendLine()
            }
        }
    }
    
    /**
     * Log current model information and performance.
     */
    fun logCurrentStatus() {
        val model = getCurrentModelType()
        val performance = getCurrentPerformance()
        
        Log.i(TAG, "=== Current Model Status ===")
        Log.i(TAG, "Model: ${model.displayName}")
        Log.i(TAG, "Path: ${model.modelPath}")
        Log.i(TAG, "Available: ${isModelAvailable(model)}")
        
        performance?.let {
            Log.i(TAG, "Performance: ${it.getFormattedStats()}")
        }
    }
}


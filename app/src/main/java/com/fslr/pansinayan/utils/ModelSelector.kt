package com.fslr.pansinayan.utils

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData

/**
 * ModelSelector - Manages model selection and switching between different PyTorch CTC models.
 *
 * This class provides a centralized way to manage which sign language recognition model
 * is currently active in the app. It supports Transformer and MediaPipeGRU PyTorch models
 * and allows dynamic switching during runtime.
 */
class ModelSelector(private val context: Context) {
    
    companion object {
        private const val TAG = "ModelSelector"
        private const val PREFS_NAME = "model_selector_prefs"
        private const val KEY_SELECTED_MODEL = "selected_model"
        
        // CTC model file paths in assets (PyTorch Lite only)
        const val TRANSFORMER_MODEL_PATH = "SignTransformerCtc_best.ptl"
        const val MEDIAPIPE_GRU_MODEL_PATH = "MediaPipeGRUCtc_best.ptl"
        const val TRANSFORMER_META_PATH = "SignTransformerCtc_best.model.json"
        const val MEDIAPIPE_GRU_META_PATH = "MediaPipeGRUCtc_best.model.json"
        
        // Default model selection
        val DEFAULT_MODEL = ModelType.TRANSFORMER
    }
    
    enum class ModelType(
        val displayName: String,
        val modelPath: String,
        val metadataPath: String,
        val description: String,
        val estimatedSize: String,
        val estimatedLatency: String,
        val accuracy: String,
        val isRecommended: Boolean
    ) {
        TRANSFORMER(
            displayName = "Transformer",
            modelPath = TRANSFORMER_MODEL_PATH,
            metadataPath = TRANSFORMER_META_PATH,
            description = "PyTorch Transformer encoder CTC.",
            estimatedSize = "~73 MB",
            estimatedLatency = "150-250ms",
            accuracy = "★★★★★",
            isRecommended = true
        ),
        
        MEDIAPIPE_GRU(
            displayName = "GRU Baseline",
            modelPath = MEDIAPIPE_GRU_MODEL_PATH,
            metadataPath = MEDIAPIPE_GRU_META_PATH,
            description = "PyTorch GRU baseline CTC.",
            estimatedSize = "~10 MB",
            estimatedLatency = "50-100ms",
            accuracy = "★★★★☆",
            isRecommended = false
        );
        
        companion object {
            fun fromString(name: String): ModelType? = values().find { it.name.equals(name, ignoreCase = true) }
        }
    }
    
    private val prefs: SharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    private val _currentModel = MutableLiveData<ModelType>()
    val currentModel: LiveData<ModelType> = _currentModel
    
    init {
        val savedModelName = prefs.getString(KEY_SELECTED_MODEL, null)
        val initialModel = savedModelName?.let { ModelType.fromString(it) } ?: DEFAULT_MODEL
        _currentModel.value = initialModel
        Log.i(TAG, "ModelSelector initialized with model: ${initialModel.displayName}")
    }
    
    fun selectModel(modelType: ModelType): Boolean {
        if (!isModelAvailable(modelType)) {
            Log.e(TAG, "Model file not found: ${modelType.modelPath}")
            return false
        }
        _currentModel.value = modelType
        prefs.edit().putString(KEY_SELECTED_MODEL, modelType.name).apply()
        Log.i(TAG, "Model selected: ${modelType.displayName}")
        return true
    }
    
    fun isModelAvailable(modelType: ModelType): Boolean = try {
        context.assets.open(modelType.modelPath).use { true }
    } catch (e: Exception) {
        Log.w(TAG, "Model not available: ${modelType.modelPath}")
        false
    }
    
    fun getCurrentModelType(): ModelType = _currentModel.value ?: DEFAULT_MODEL
    fun getCurrentModelPath(): String = getCurrentModelType().modelPath
    fun getCurrentMetadataPath(): String = getCurrentModelType().metadataPath
}


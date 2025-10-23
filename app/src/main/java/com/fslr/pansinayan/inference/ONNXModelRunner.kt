package com.fslr.pansinayan.inference

import android.content.Context
import android.util.Log
import com.google.gson.Gson
import com.google.gson.JsonObject
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession.SessionOptions
import ai.onnxruntime.OrtException
import java.io.IOException
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.Collections

/**
 * ONNX Runtime-based model runner for sign language recognition.
 * 
 * This class handles loading and running ONNX models for CTC-based
 * continuous sign language recognition using Microsoft ONNX Runtime.
 */
class ONNXModelRunner(
    private val context: Context,
    private val modelName: String = "sign_transformer_ctc.onnx"
) {
    
    companion object {
        private const val TAG = "ONNXModelRunner"
        private const val INPUT_NAME = "input"
        private const val OUTPUT_NAME = "output"
        private const val MAX_SEQUENCE_LENGTH = 300
        private const val INPUT_DIMENSION = 178
    }
    
    private var session: OrtSession? = null
    private var environment: OrtEnvironment? = null
    private var labelMapping: Map<String, String>? = null
    private var inferenceTimes = mutableListOf<Long>()
    
    init {
        initializeModel()
        loadLabelMapping()
    }
    
    /**
     * Initialize the ONNX model session
     */
    private fun initializeModel() {
        try {
            environment = OrtEnvironment.getEnvironment()
            
            // Load model from assets
            val modelBytes = context.assets.open(modelName).readBytes()
            val sessionOptions = OrtSession.SessionOptions()
            
            // Configure session options for better performance
            sessionOptions.setIntraOpNumThreads(4)
            sessionOptions.setInterOpNumThreads(2)
            
            session = environment!!.createSession(modelBytes, sessionOptions)
            
            Log.d(TAG, "ONNX model loaded successfully: $modelName")
            Log.d(TAG, "Input names: ${session!!.inputNames}")
            Log.d(TAG, "Output names: ${session!!.outputNames}")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize ONNX model", e)
            throw RuntimeException("Failed to load ONNX model: ${e.message}")
        }
    }
    
    /**
     * Load label mapping from assets
     */
    private fun loadLabelMapping() {
        try {
            val jsonString = context.assets.open("label_mapping_greeting.json").bufferedReader().use { it.readText() }
            val gson = Gson()
            val jsonObject = gson.fromJson(jsonString, JsonObject::class.java)
            
            // Extract gloss labels
            val glosses = jsonObject.getAsJsonObject("glosses")
            labelMapping = mutableMapOf<String, String>().apply {
                glosses.entrySet().forEach { entry ->
                    put(entry.key, entry.value.asString)
                }
            }
            
            Log.d(TAG, "Label mapping loaded: ${labelMapping!!.size} labels")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load label mapping", e)
            // Create fallback mapping
            labelMapping = mapOf(
                "0" to "GOOD MORNING",
                "1" to "GOOD AFTERNOON", 
                "2" to "GOOD EVENING",
                "3" to "HELLO",
                "4" to "HOW ARE YOU",
                "5" to "IM FINE",
                "6" to "NICE TO MEET YOU",
                "7" to "THANK YOU",
                "8" to "YOURE WELCOME",
                "9" to "SEE YOU TOMORROW"
            )
        }
    }
    
    fun runInference(keypointSequence: Array<FloatArray>): InferenceResult? {
        val currentSession = session ?: run {
            Log.e(TAG, "Model not initialized")
            return null
        }
        
        val startTime = System.currentTimeMillis()
        
        try {
            val batchSize = 1
            val sequenceLength = keypointSequence.size.coerceAtMost(MAX_SEQUENCE_LENGTH)
            
            // Prepare input tensor
            val inputShape = longArrayOf(batchSize.toLong(), sequenceLength.toLong(), INPUT_DIMENSION.toLong())
            val inputBuffer = FloatBuffer.allocate(batchSize * sequenceLength * INPUT_DIMENSION)
            
            // Fill input buffer with keypoint data
            for (i in 0 until sequenceLength) {
                val keypoints = keypointSequence[i]
                for (j in keypoints.indices) {
                    inputBuffer.put(keypoints[j])
                }
                // Pad remaining dimensions if needed
                for (j in keypoints.size until INPUT_DIMENSION) {
                    inputBuffer.put(0.0f)
                }
            }
            
            // Create ONNX tensor
            val inputTensor = OnnxTensor.createTensor(
                environment!!,
                inputBuffer,
                inputShape
            )
            
            // Run inference
            val inputs = Collections.singletonMap(INPUT_NAME, inputTensor)
            val results = currentSession.run(inputs)
            
            // Extract output
            val outputTensor = results.get(0) as OnnxTensor
            val outputBuffer = outputTensor.floatBuffer
            
            // Process CTC output
            val predictions = processCTCOutput(outputBuffer, sequenceLength)
            
            // Clean up
            inputTensor.close()
            outputTensor.close()
            results.close()
            
            val inferenceTime = System.currentTimeMillis() - startTime
            inferenceTimes.add(inferenceTime)
            
            // Keep only last 100 inference times for average calculation
            if (inferenceTimes.size > 100) {
                inferenceTimes.removeAt(0)
            }
            
            return InferenceResult(
                glossPrediction = 0, // CTC doesn't have single gloss prediction
                glossConfidence = calculateConfidence(predictions),
                glossProbabilities = FloatArray(11), // Placeholder
                glossTop5 = emptyList(), // CTC doesn't have top-k
                categoryPrediction = 0, // Greetings category
                categoryConfidence = 0.9f, // High confidence for greetings
                categoryProbabilities = floatArrayOf(0.9f, 0.1f), // Placeholder
                inferenceTimeMs = inferenceTime,
                sequenceLength = sequenceLength,
                isCTC = true,
                ctcPredictions = predictions.mapIndexed { index, prediction ->
                    SignPrediction(
                        glossId = index,
                        categoryId = 0,
                        startFrame = index * 10,
                        endFrame = (index + 1) * 10,
                        categoryConfidence = calculateConfidence(predictions)
                    )
                }
            )
            
        } catch (e: Exception) {
            Log.e(TAG, "Inference failed", e)
            return null
        }
    }
    
    /**
     * Process CTC output to extract predicted sequence
     */
    private fun processCTCOutput(outputBuffer: FloatBuffer, sequenceLength: Int): List<String> {
        val predictions = mutableListOf<String>()
        val numClasses = 11 // 10 greetings + 1 blank
        
        // Reshape output: [sequence_length, num_classes]
        val logProbs = Array(sequenceLength) { FloatArray(numClasses) }
        
        for (t in 0 until sequenceLength) {
            for (c in 0 until numClasses) {
                logProbs[t][c] = outputBuffer.get()
            }
        }
        
        // Simple CTC decoding (greedy)
        var prevClass = -1
        for (t in 0 until sequenceLength) {
            var maxClass = 0
            var maxProb = logProbs[t][0]
            
            for (c in 1 until numClasses) {
                if (logProbs[t][c] > maxProb) {
                    maxProb = logProbs[t][c]
                    maxClass = c
                }
            }
            
            // CTC blank token is class 10 (0-indexed)
            if (maxClass != 10 && maxClass != prevClass) {
                val label = labelMapping?.get(maxClass.toString()) ?: "UNKNOWN"
                predictions.add(label)
            }
            prevClass = maxClass
        }
        
        return predictions
    }
    
    /**
     * Calculate confidence score for predictions
     */
    private fun calculateConfidence(predictions: List<String>): Float {
        return if (predictions.isNotEmpty()) 0.85f else 0.0f // Placeholder confidence
    }
    
    /**
     * Get model information
     */
    fun getModelInfo(): Map<String, Any> {
        return mapOf(
            "model_name" to modelName,
            "input_dimension" to INPUT_DIMENSION,
            "max_sequence_length" to MAX_SEQUENCE_LENGTH,
            "num_classes" to 11,
            "label_count" to (labelMapping?.size ?: 0)
        )
    }
    
    /**
     * Get average inference time in milliseconds
     */
    fun getAverageInferenceTime(): Long {
        return if (inferenceTimes.isNotEmpty()) {
            inferenceTimes.average().toLong()
        } else {
            0L
        }
    }
    
    /**
     * Release resources
     */
    fun release() {
        try {
            session?.close()
            environment?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error closing ONNX resources", e)
        }
    }
}
package com.fslr.pansinayan.inference

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp

/**
 * Runs TensorFlow Lite model inference for sign language classification.
 * 
 * Responsibilities:
 * - Load TFLite model from assets/classification/
 * - Configure GPU acceleration if available
 * - Run inference on keypoint sequences
 * - Post-process outputs (softmax, argmax, top-k)
 * 
 * Model Specs:
 * - Input: [1, T, 178] where T is sequence length (89 keypoints × 2)
 * - Output: [1, 105] classification logits
 * 
 * Keypoint Structure (89 keypoints total):
 * - Pose: 25 points × 2 = 50 values [indices 0-49]
 * - Left hand: 21 points × 2 = 42 values [indices 50-91]
 * - Right hand: 21 points × 2 = 42 values [indices 92-133]
 * - Face: 22 points × 2 = 44 values [indices 134-177]
 * 
 * Usage:
 *   val runner = TFLiteModelRunner(context, "classification/sign_transformer_fp16.tflite")
 *   val result = runner.runInference(sequence)
 *   println("Predicted: ${result.glossPrediction} (${result.glossConfidence})")
 */
class TFLiteModelRunner(
    private val context: Context,
    private val modelPath: String = "classification/sign_transformer_fp16.tflite"
) {
    companion object {
        private const val TAG = "TFLiteModelRunner"
        private const val INPUT_DIM = 178  // 89 keypoints × 2
        private const val OUTPUT_GLOSS_CLASSES = 105
    }
    
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    
    // Performance metrics
    private var inferenceCount = 0
    private var totalInferenceTimeMs = 0L

    init {
        loadModel()
    }

    /**
     * Load TFLite model from assets and configure interpreter.
     */
    private fun loadModel() {
        try {
            // Check if model file exists
            val assetList = context.assets.list("") ?: emptyArray()
            if (!assetList.contains(modelPath)) {
                Log.e(TAG, "Model file not found: $modelPath")
                Log.d(TAG, "Available assets: ${assetList.joinToString()}")
                return
            }

            val modelBuffer = loadModelFile()
            val options = Interpreter.Options()

            // Default to CPU for broad compatibility
            options.setNumThreads(4)

            interpreter = Interpreter(modelBuffer, options)
            Log.i(TAG, "TFLite classification model loaded successfully: $modelPath")

            // Log input/output tensor shapes for debugging
            val inputShape = interpreter?.getInputTensor(0)?.shape()
            val outputShape = interpreter?.getOutputTensor(0)?.shape()
            Log.d(TAG, "Model input shape: ${inputShape?.contentToString()}")
            Log.d(TAG, "Model output shape: ${outputShape?.contentToString()}")

        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model", e)
            throw RuntimeException("Model loading failed. Ensure $modelPath is in assets/", e)
        }
    }

    /**
     * Load model file from assets as MappedByteBuffer.
     */
    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /**
     * Run inference on a keypoint sequence.
     * 
     * @param sequence Array of shape [T, 178] where T is sequence length (89 keypoints × 2)
     * @return InferenceResult containing predictions and confidence scores
     */
    fun runInference(sequence: Array<FloatArray>): InferenceResult? {
        val interpreter = interpreter ?: run {
            Log.e(TAG, "Interpreter not initialized")
            return null
        }

        try {
            val startTime = System.currentTimeMillis()

            // Prepare input tensor [1, T, 178]
            val inputBuffer = prepareInputBuffer(sequence)

            // Prepare output tensor [1, 105]
            val glossLogits = Array(1) { FloatArray(OUTPUT_GLOSS_CLASSES) }

            // Run inference
            interpreter.run(inputBuffer, glossLogits)

            val inferenceTime = System.currentTimeMillis() - startTime
            
            // Update statistics
            inferenceCount++
            totalInferenceTimeMs += inferenceTime
            
            // Log performance periodically
            if (inferenceCount % 10 == 0) {
                val avgTime = totalInferenceTimeMs / inferenceCount
                Log.d(TAG, "Inference #$inferenceCount: ${inferenceTime}ms (avg: ${avgTime}ms)")
            }

            // Apply softmax to convert logits to probabilities
            val glossProbs = softmax(glossLogits[0])

            // Get top predictions
            val topGlossIdx = glossProbs.indices.maxByOrNull { glossProbs[it] } ?: 0

            // Get top-5 gloss predictions
            val top5Indices = glossProbs.indices.sortedByDescending { glossProbs[it] }.take(5)

            return InferenceResult(
                glossPrediction = topGlossIdx,
                glossConfidence = glossProbs[topGlossIdx],
                glossProbabilities = glossProbs,
                glossTop5 = top5Indices.map { Pair(it, glossProbs[it]) },
                categoryPrediction = 0, // Will be determined by label mapping
                categoryConfidence = 0.0f, // Will be calculated separately
                categoryProbabilities = FloatArray(10), // Placeholder
                inferenceTimeMs = inferenceTime,
                sequenceLength = sequence.size
            )

        } catch (e: Exception) {
            Log.e(TAG, "Classification inference failed", e)
            return null
        }
    }

    /**
     * Prepare input buffer from sequence.
     * Converts Array[T, 178] to ByteBuffer in format expected by TFLite.
     */
    private fun prepareInputBuffer(sequence: Array<FloatArray>): ByteBuffer {
        val batchSize = 1
        val seqLen = sequence.size
        val featDim = INPUT_DIM

        // Allocate direct byte buffer (4 bytes per float)
        val bufferSize = batchSize * seqLen * featDim * 4
        val buffer = ByteBuffer.allocateDirect(bufferSize)
        buffer.order(ByteOrder.nativeOrder())

        // Fill buffer with sequence data in row-major order
        for (t in 0 until seqLen) {
            for (i in 0 until featDim) {
                buffer.putFloat(sequence[t][i])
            }
        }

        buffer.rewind()
        return buffer
    }

    /**
     * Apply softmax to convert logits to probabilities.
     */
    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        val exps = FloatArray(logits.size) { i ->
            exp((logits[i] - maxLogit).toDouble()).toFloat()
        }
        val sumExps = exps.sum()
        return FloatArray(logits.size) { i -> exps[i] / sumExps }
    }

    /**
     * Release TFLite resources.
     */
    fun release() {
        interpreter?.close()
        gpuDelegate?.close()
        
        val avgTime = if (inferenceCount > 0) totalInferenceTimeMs / inferenceCount else 0
        Log.i(TAG, "TFLite resources released. Stats: $inferenceCount inferences, avg time: ${avgTime}ms")
    }

    /**
     * Get average inference time.
     */
    fun getAverageInferenceTime(): Long {
        return if (inferenceCount > 0) totalInferenceTimeMs / inferenceCount else 0
    }
}


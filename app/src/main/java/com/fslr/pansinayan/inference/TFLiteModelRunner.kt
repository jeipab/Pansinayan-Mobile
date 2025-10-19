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
 * Runs TensorFlow Lite model inference for sign language recognition.
 * 
 * Responsibilities:
 * - Load TFLite model from assets
 * - Configure GPU acceleration if available
 * - Run inference on keypoint sequences
 * - Post-process outputs (softmax, argmax, top-k)
 * 
 * Model Specs:
 * - Input: [1, T, 156] where T is sequence length
 * - Output 0: [1, 105] gloss logits
 * - Output 1: [1, 10] category logits
 * 
 * Usage:
 *   val runner = TFLiteModelRunner(context)
 *   val result = runner.runInference(sequence)
 *   println("Predicted: ${result.glossPrediction} (${result.glossConfidence})")
 */
class TFLiteModelRunner(
    private val context: Context,
    private val modelPath: String = "sign_transformer_quant.tflite"
) {
    companion object {
        private const val TAG = "TFLiteModelRunner"
        private const val INPUT_DIM = 156
        private const val OUTPUT_GLOSS_CLASSES = 105
        private const val OUTPUT_CATEGORY_CLASSES = 10
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

            // Try to use GPU delegate for faster inference
            try {
                gpuDelegate = GpuDelegate()
                options.addDelegate(gpuDelegate)
                Log.i(TAG, "GPU delegate enabled for TensorFlow Lite")
            } catch (e: Exception) {
                Log.w(TAG, "GPU delegate not available, using CPU threads instead", e)
                options.setNumThreads(4)
            }

            interpreter = Interpreter(modelBuffer, options)
            Log.i(TAG, "TFLite model loaded successfully: $modelPath")

            // Log input/output tensor shapes for debugging
            val inputShape = interpreter?.getInputTensor(0)?.shape()
            val outputShape0 = interpreter?.getOutputTensor(0)?.shape()
            val outputShape1 = interpreter?.getOutputTensor(1)?.shape()
            Log.d(TAG, "Model input shape: ${inputShape?.contentToString()}")
            Log.d(TAG, "Model output shapes: ${outputShape0?.contentToString()}, ${outputShape1?.contentToString()}")

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
     * @param sequence Array of shape [T, 156] where T is sequence length
     * @return InferenceResult containing predictions and confidence scores
     */
    fun runInference(sequence: Array<FloatArray>): InferenceResult? {
        val interpreter = interpreter ?: run {
            Log.e(TAG, "Interpreter not initialized")
            return null
        }

        try {
            val startTime = System.currentTimeMillis()

            // Prepare input tensor [1, T, 156]
            val inputBuffer = prepareInputBuffer(sequence)

            // Prepare output tensors
            val glossLogits = Array(1) { FloatArray(OUTPUT_GLOSS_CLASSES) }
            val categoryLogits = Array(1) { FloatArray(OUTPUT_CATEGORY_CLASSES) }

            // Run inference
            val outputs = mapOf(
                0 to glossLogits,
                1 to categoryLogits
            )
            interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)

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
            val categoryProbs = softmax(categoryLogits[0])

            // Get top predictions
            val topGlossIdx = glossProbs.indices.maxByOrNull { glossProbs[it] } ?: 0
            val topCategoryIdx = categoryProbs.indices.maxByOrNull { categoryProbs[it] } ?: 0

            // Get top-5 gloss predictions
            val top5Indices = glossProbs.indices.sortedByDescending { glossProbs[it] }.take(5)

            return InferenceResult(
                glossPrediction = topGlossIdx,
                glossConfidence = glossProbs[topGlossIdx],
                glossProbabilities = glossProbs,
                glossTop5 = top5Indices.map { Pair(it, glossProbs[it]) },
                categoryPrediction = topCategoryIdx,
                categoryConfidence = categoryProbs[topCategoryIdx],
                categoryProbabilities = categoryProbs,
                inferenceTimeMs = inferenceTime,
                sequenceLength = sequence.size
            )

        } catch (e: Exception) {
            Log.e(TAG, "Inference failed", e)
            return null
        }
    }

    /**
     * Prepare input buffer from sequence.
     * Converts Array[T, 156] to ByteBuffer in format expected by TFLite.
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
    val sequenceLength: Int
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


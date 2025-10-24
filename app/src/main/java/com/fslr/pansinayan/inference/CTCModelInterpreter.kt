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

/**
 * TensorFlow Lite interpreter for CTC-based continuous sign language recognition.
 * 
 * Responsibilities:
 * - Load CTC models from assets/ctc/
 * - Configure GPU acceleration if available
 * - Run inference on keypoint sequences
 * - Return raw logits for CTC decoding
 * 
 * Model Specs:
 * - Input: [1, T, 178] where T is sequence length (89 keypoints × 2)
 * - Output: [1, T, num_classes + 1] where last index is blank token
 * - Fixed sequence length: T = 300 frames (10 seconds at 30 FPS)
 * 
 * Keypoint Structure (89 keypoints total):
 * - Pose: 25 points × 2 = 50 values [indices 0-49]
 * - Left hand: 21 points × 2 = 42 values [indices 50-91]
 * - Right hand: 21 points × 2 = 42 values [indices 92-133]
 * - Face: 22 points × 2 = 44 values [indices 134-177]
 * 
 * Usage:
 *   val interpreter = CTCModelInterpreter(context, "ctc/sign_transformer_ctc_fp16.tflite")
 *   val logits = interpreter.runInference(sequence)
 *   val decoded = ctcDecoder.decode(logits)
 */
class CTCModelInterpreter(
    private val context: Context,
    private val modelPath: String = "ctc/sign_transformer_ctc_fp16.tflite"
) {
    companion object {
        private const val TAG = "CTCModelInterpreter"
        private const val INPUT_DIM = 178  // 89 keypoints × 2
        private const val FIXED_SEQUENCE_LENGTH = 300  // Fixed T for CTC models
        private const val OUTPUT_CLASSES = 106  // 105 glosses + 1 blank token
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
     * Load CTC model from assets and configure interpreter.
     */
    private fun loadModel() {
        try {
            // Check if model file exists
            val assetList = context.assets.list("") ?: emptyArray()
            if (!assetList.contains(modelPath)) {
                Log.e(TAG, "CTC model file not found: $modelPath")
                Log.d(TAG, "Available assets: ${assetList.joinToString()}")
                return
            }

            val modelBuffer = loadModelFile()
            val options = Interpreter.Options()

            // Try to use GPU delegate for faster inference
            try {
                gpuDelegate = GpuDelegate()
                options.addDelegate(gpuDelegate)
                Log.i(TAG, "GPU delegate enabled for CTC TensorFlow Lite")
            } catch (e: Exception) {
                Log.w(TAG, "GPU delegate not available, using CPU threads instead", e)
                options.setNumThreads(4)
            }

            interpreter = Interpreter(modelBuffer, options)
            Log.i(TAG, "CTC TFLite model loaded successfully: $modelPath")

            // Log input/output tensor shapes for debugging
            val inputShape = interpreter?.getInputTensor(0)?.shape()
            val outputShape = interpreter?.getOutputTensor(0)?.shape()
            Log.d(TAG, "CTC model input shape: ${inputShape?.contentToString()}")
            Log.d(TAG, "CTC model output shape: ${outputShape?.contentToString()}")

        } catch (e: Exception) {
            Log.e(TAG, "Failed to load CTC model", e)
            throw RuntimeException("CTC model loading failed. Ensure $modelPath is in assets/", e)
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
     * Run CTC inference on a keypoint sequence.
     * 
     * @param sequence Array of shape [T, 178] where T ≤ FIXED_SEQUENCE_LENGTH
     * @return Raw logits array of shape [T, num_classes + 1] for CTC decoding
     */
    fun runInference(sequence: Array<FloatArray>): FloatArray? {
        val interpreter = interpreter ?: run {
            Log.e(TAG, "Interpreter not initialized")
            return null
        }

        try {
            val startTime = System.currentTimeMillis()

            // Prepare input tensor [1, T, 178] with zero-padding if needed
            val inputBuffer = prepareInputBuffer(sequence)

            // Prepare output tensor [1, T, num_classes + 1]
            val outputLogits = Array(1) { Array(FIXED_SEQUENCE_LENGTH) { FloatArray(OUTPUT_CLASSES) } }

            // Run inference
            interpreter.run(inputBuffer, outputLogits)

            val inferenceTime = System.currentTimeMillis() - startTime
            
            // Update statistics
            inferenceCount++
            totalInferenceTimeMs += inferenceTime
            
            // Log performance periodically
            if (inferenceCount % 10 == 0) {
                val avgTime = totalInferenceTimeMs / inferenceCount
                Log.d(TAG, "CTC Inference #$inferenceCount: ${inferenceTime}ms (avg: ${avgTime}ms)")
            }

            // Flatten output to [T, num_classes + 1] for CTC decoding
            val flattenedLogits = FloatArray(FIXED_SEQUENCE_LENGTH * OUTPUT_CLASSES)
            var idx = 0
            for (t in 0 until FIXED_SEQUENCE_LENGTH) {
                for (c in 0 until OUTPUT_CLASSES) {
                    flattenedLogits[idx++] = outputLogits[0][t][c]
                }
            }

            return flattenedLogits

        } catch (e: Exception) {
            Log.e(TAG, "CTC inference failed", e)
            return null
        }
    }

    /**
     * Prepare input buffer from sequence with zero-padding.
     * Converts Array[T, 178] to ByteBuffer in format expected by TFLite.
     * Pads with zeros if sequence is shorter than FIXED_SEQUENCE_LENGTH.
     */
    private fun prepareInputBuffer(sequence: Array<FloatArray>): ByteBuffer {
        val batchSize = 1
        val seqLen = minOf(sequence.size, FIXED_SEQUENCE_LENGTH)
        val featDim = INPUT_DIM

        // Allocate direct byte buffer (4 bytes per float)
        val bufferSize = batchSize * FIXED_SEQUENCE_LENGTH * featDim * 4
        val buffer = ByteBuffer.allocateDirect(bufferSize)
        buffer.order(ByteOrder.nativeOrder())

        // Fill buffer with sequence data in row-major order
        for (t in 0 until FIXED_SEQUENCE_LENGTH) {
            if (t < seqLen) {
                // Copy actual sequence data
                for (i in 0 until featDim) {
                    buffer.putFloat(sequence[t][i])
                }
            } else {
                // Zero-pad remaining frames
                for (i in 0 until featDim) {
                    buffer.putFloat(0f)
                }
            }
        }

        buffer.rewind()
        return buffer
    }

    /**
     * Release TFLite resources.
     */
    fun release() {
        interpreter?.close()
        gpuDelegate?.close()
        
        val avgTime = if (inferenceCount > 0) totalInferenceTimeMs / inferenceCount else 0
        Log.i(TAG, "CTC TFLite resources released. Stats: $inferenceCount inferences, avg time: ${avgTime}ms")
    }

    /**
     * Get average inference time.
     */
    fun getAverageInferenceTime(): Long {
        return if (inferenceCount > 0) totalInferenceTimeMs / inferenceCount else 0
    }

    /**
     * Get model info for debugging.
     */
    fun getModelInfo(): String {
        return "CTC Model: $modelPath, Input: [1, $FIXED_SEQUENCE_LENGTH, $INPUT_DIM], Output: [1, $FIXED_SEQUENCE_LENGTH, $OUTPUT_CLASSES]"
    }
}

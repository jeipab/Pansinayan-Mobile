package com.fslr.pansinayan.inference

import android.content.Context
import android.util.Log
import com.fslr.pansinayan.network.InferenceRequest
import com.fslr.pansinayan.network.NetworkClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Model runner that sends keypoints to remote server for inference.
 */
class RemoteModelRunner(
    private val context: Context,
    private val metadataPath: String
) : ModelRunner {

    companion object {
        private const val TAG = "RemoteModelRunner"
    }

    override val meta: CtcModelMetadata = ModelMetadataLoader.loadFromAssets(context, metadataPath)

    private val modelType: String = when {
        metadataPath.contains("Transformer", ignoreCase = true) -> "transformer"
        metadataPath.contains("GRU", ignoreCase = true) -> "gru"
        else -> "transformer"
    }

    override fun run(sequence: Array<FloatArray>): CtcOutputs {
        throw UnsupportedOperationException(
            "RemoteModelRunner.run() is blocking. Use runAsync() from a coroutine instead."
        )
    }

    /**
     * Run inference asynchronously on remote server.
     */
    suspend fun runAsync(sequence: Array<FloatArray>): CtcOutputs = withContext(Dispatchers.IO) {
        try {
            val startTime = System.currentTimeMillis()

            // Convert sequence to list format for JSON
            val keypointsList = sequence.map { it.toList() }

            val request = InferenceRequest(
                keypoints = keypointsList,
                model_type = modelType
            )

            Log.d(TAG, "Sending ${sequence.size} frames to server (model=$modelType)...")

            // Send to server
            val response = NetworkClient.getApi().predict(request)

            if (!response.isSuccessful) {
                throw RuntimeException("Server error: ${response.code()} ${response.message()}")
            }

            val body = response.body() ?: throw RuntimeException("Empty response from server")

            val inferenceTime = System.currentTimeMillis() - startTime
            Log.i(TAG, "Remote inference: ${body.inference_time_ms}ms (server) + ${inferenceTime - body.inference_time_ms.toLong()}ms (network)")

            // Convert response back to CtcOutputs format
            val logProbs = body.ctc_log_probs.map { it.toFloatArray() }.toTypedArray()
            val catLogits = body.cat_logits?.map { it.toFloatArray() }?.toTypedArray()

            return@withContext CtcOutputs(
                logProbs = arrayOf(logProbs),
                catLogits = catLogits?.let { arrayOf(it) }
            )

        } catch (e: Exception) {
            Log.e(TAG, "Remote inference failed", e)
            throw e
        }
    }

    override fun release() {
        Log.i(TAG, "RemoteModelRunner released")
    }
}
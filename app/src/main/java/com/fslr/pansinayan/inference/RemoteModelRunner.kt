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
    override suspend fun runAsync(sequence: Array<FloatArray>): CtcOutputs = withContext(Dispatchers.IO) {
        try {
            val startTime = System.currentTimeMillis()

            // Convert sequence to list format for JSON
            val keypointsList = sequence.map { it.toList() }

            val request = InferenceRequest(
                keypoints = keypointsList,
                model_type = modelType
            )

            Log.i(TAG, "=== SENDING INFERENCE REQUEST ===")
            Log.i(TAG, "Sequence length: ${sequence.size} frames")
            Log.i(TAG, "Model type: $modelType")
            Log.i(TAG, "Server URL: ${NetworkClient.getServerUrl()}")
            Log.i(TAG, "Keypoints per frame: ${sequence.firstOrNull()?.size ?: 0}")

            // Send to server
            Log.d(TAG, "Calling NetworkClient.getApi().predict()...")
            val response = NetworkClient.getApi().predict(request)
            Log.i(TAG, "Response received: code=${response.code()}, isSuccessful=${response.isSuccessful}")

            if (!response.isSuccessful) {
                var errorBody: String? = null
                try {
                    errorBody = response.errorBody()?.string()
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to read error body: ${e.message}")
                }
                Log.e(TAG, "=== SERVER ERROR RESPONSE ===")
                Log.e(TAG, "HTTP Code: ${response.code()}")
                Log.e(TAG, "HTTP Message: ${response.message()}")
                Log.e(TAG, "Error Body: $errorBody")
                Log.e(TAG, "Headers: ${response.headers()}")
                val errorMsg = errorBody ?: response.message()
                throw RuntimeException("Server error ${response.code()}: $errorMsg")
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
            Log.e(TAG, "=== REMOTE INFERENCE FAILED ===")
            Log.e(TAG, "Exception type: ${e.javaClass.simpleName}")
            Log.e(TAG, "Exception message: ${e.message}")
            Log.e(TAG, "Server URL: ${NetworkClient.getServerUrl()}")
            e.printStackTrace()
            throw e
        }
    }

    override fun release() {
        Log.i(TAG, "RemoteModelRunner released")
    }
}
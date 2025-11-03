package com.fslr.pansinayan.network

import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST

/**
 * Retrofit API interface for Pansinayan server.
 */
interface PansinayanApi {

    @GET("health")
    suspend fun healthCheck(): Response<HealthResponse>

    @POST("predict")
    suspend fun predict(@Body request: InferenceRequest): Response<InferenceResponse>
}

data class InferenceRequest(
    val keypoints: List<List<Float>>,
    val model_type: String  // "transformer" or "gru"
)

data class InferenceResponse(
    val ctc_log_probs: List<List<Float>>,
    val cat_logits: List<List<Float>>?,
    val sequence_length: Int,
    val inference_time_ms: Float,
    val model_used: String
)

data class HealthResponse(
    val status: String,
    val models_loaded: List<String>,
    val device: String,
    val gpu_available: Boolean
)
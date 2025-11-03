package com.fslr.pansinayan.network

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

/**
 * Singleton network client for server communication.
 */
object NetworkClient {
    private const val TAG = "NetworkClient"
    private const val PREFS_NAME = "pansinayan_network"
    private const val KEY_SERVER_URL = "server_url"
    private const val DEFAULT_SERVER_URL = "http://192.168.1.100:8000"  // CHANGE THIS to your server

    private lateinit var api: PansinayanApi
    private lateinit var prefs: SharedPreferences

    fun initialize(context: Context) {
        prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

        val serverUrl = getServerUrl()
        Log.i(TAG, "Initializing NetworkClient with server: $serverUrl")

        val loggingInterceptor = HttpLoggingInterceptor().apply {
            level = HttpLoggingInterceptor.Level.BASIC
        }

        val okHttpClient = OkHttpClient.Builder()
            .addInterceptor(loggingInterceptor)
            .connectTimeout(10, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .build()

        val retrofit = Retrofit.Builder()
            .baseUrl(serverUrl)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()

        api = retrofit.create(PansinayanApi::class.java)
    }

    fun getApi(): PansinayanApi = api

    fun getServerUrl(): String {
        return prefs.getString(KEY_SERVER_URL, DEFAULT_SERVER_URL) ?: DEFAULT_SERVER_URL
    }

    fun setServerUrl(url: String) {
        prefs.edit().putString(KEY_SERVER_URL, url).apply()
        Log.i(TAG, "Server URL updated to: $url")
    }

    /**
     * Test server connectivity.
     */
    suspend fun testConnection(): Boolean {
        return try {
            val response = api.healthCheck()
            val isHealthy = response.isSuccessful && response.body()?.status == "healthy"
            Log.i(TAG, "Health check: ${if (isHealthy) "✓" else "✗"}")
            isHealthy
        } catch (e: Exception) {
            Log.e(TAG, "Health check failed", e)
            false
        }
    }
}
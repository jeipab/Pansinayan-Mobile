package com.fslr.pansinayan.inference

import android.content.Context
import android.util.Log
import com.fslr.pansinayan.network.NetworkClient
import com.fslr.pansinayan.utils.ModeManager
import kotlinx.coroutines.*

/**
 * Manages inference runners for both online and offline modes.
 * Provides automatic fallback, lazy initialization, and graceful mode switching.
 */
class InferenceManager(
    private val context: Context,
    private val modeManager: ModeManager
) {
    companion object {
        private const val TAG = "InferenceManager"
    }

    private var offlineRunner: PyTorchModelRunner? = null
    private var onlineRunner: RemoteModelRunner? = null
    private var connectionMonitor: ConnectionMonitor? = null
    private val managerScope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    /**
     * Run inference using the current mode's runner, with automatic fallback.
     */
    suspend fun runInference(sequence: Array<FloatArray>): CtcOutputs {
        val runner = getCurrentRunner()
            ?: throw IllegalStateException("No inference runner available")
        return runner.runAsync(sequence)
    }

    /**
     * Get the current runner based on mode, with automatic fallback to offline.
     */
    private suspend fun getCurrentRunner(): ModelRunner? {
        return when (modeManager.getCurrentMode()) {
            ModeManager.InferenceMode.OFFLINE -> getOfflineRunner()
            ModeManager.InferenceMode.ONLINE -> {
                getOnlineRunner() ?: run {
                    Log.w(TAG, "Online runner unavailable, falling back to offline")
                    getOfflineRunner()
                }
            }
        }
    }

    /**
     * Get or create offline runner (lazy initialization).
     */
    private suspend fun getOfflineRunner(): PyTorchModelRunner? {
        if (offlineRunner == null) {
            try {
                context.assets.open("SignTransformerCtc_best.ptl").close()
                offlineRunner = PyTorchModelRunner(
                    context = context,
                    assetModelPath = "SignTransformerCtc_best.ptl",
                    metadataPath = "SignTransformerCtc_best.model.json"
                )
                Log.i(TAG, "Offline runner initialized")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize offline runner", e)
                return null
            }
        }
        return offlineRunner
    }

    /**
     * Get or create online runner (lazy initialization with connection check).
     */
    private suspend fun getOnlineRunner(): RemoteModelRunner? {
        if (onlineRunner == null) {
            try {
                // Test connection before creating runner
                if (!NetworkClient.testConnection()) {
                    Log.w(TAG, "Server connection failed, cannot create online runner")
                    return null
                }

                onlineRunner = RemoteModelRunner(
                    context = context,
                    metadataPath = "SignTransformerCtc_best.model.json"
                )
                Log.i(TAG, "Online runner initialized")

                // Start connection monitoring
                startConnectionMonitoring()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize online runner", e)
                return null
            }
        }
        return onlineRunner
    }

    /**
     * Switch to a new mode (non-blocking).
     */
    suspend fun switchMode(newMode: ModeManager.InferenceMode): Result<Unit> {
        return withContext(Dispatchers.IO) {
            try {
                Log.i(TAG, "Switching to $newMode mode...")

                when (newMode) {
                    ModeManager.InferenceMode.OFFLINE -> {
                        // Release online runner if switching to offline
                        onlineRunner?.release()
                        onlineRunner = null
                        stopConnectionMonitoring()
                        // Offline runner will be created lazily when needed
                    }
                    ModeManager.InferenceMode.ONLINE -> {
                        // Release offline runner if switching to online
                        offlineRunner?.release()
                        offlineRunner = null
                        // Online runner will be created lazily when needed
                    }
                }

                modeManager.setMode(newMode)
                Log.i(TAG, "Mode switched to $newMode")
                Result.success(Unit)
            } catch (e: Exception) {
                Log.e(TAG, "Mode switch failed", e)
                Result.failure(e)
            }
        }
    }

    /**
     * Start monitoring connection health (only in online mode).
     */
    private fun startConnectionMonitoring() {
        if (connectionMonitor == null) {
            connectionMonitor = ConnectionMonitor(
                onConnectionLost = {
                    Log.w(TAG, "Connection lost - will fallback to offline on next inference")
                    // Don't release runner immediately, let fallback handle it
                }
            )
            connectionMonitor?.start()
            Log.d(TAG, "Connection monitoring started")
        }
    }

    /**
     * Stop connection monitoring.
     */
    private fun stopConnectionMonitoring() {
        connectionMonitor?.release()
        connectionMonitor = null
        Log.d(TAG, "Connection monitoring stopped")
    }

    /**
     * Get metadata from current runner (or offline as fallback).
     */
    suspend fun getMetadata(): CtcModelMetadata? {
        // Try to get metadata from an existing runner
        val current = getCurrentRunner()
        if (current != null) {
            return current.meta
        }

        // If no runner is active, try to load metadata from default offline model
        Log.d(TAG, "No active runner, attempting to load metadata from default offline model.")
        return try {
            ModelMetadataLoader.loadFromAssets(context, "SignTransformerCtc_best.model.json")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load default offline model metadata", e)
            null
        }
    }

    /**
     * Release all resources.
     */
    fun release() {
        managerScope.cancel()
        offlineRunner?.release()
        onlineRunner?.release()
        stopConnectionMonitoring()
        offlineRunner = null
        onlineRunner = null
        Log.i(TAG, "InferenceManager released")
    }
}

/**
 * Monitors connection health for online mode.
 */
private class ConnectionMonitor(
    private val onConnectionLost: () -> Unit
) {
    companion object {
        private const val TAG = "ConnectionMonitor"
        private const val HEALTH_CHECK_INTERVAL_MS = 30_000L // 30 seconds
    }

    private val monitorScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private var monitorJob: Job? = null

    fun start() {
        stop() // Stop existing if any
        monitorJob = monitorScope.launch {
            while (isActive) {
                delay(HEALTH_CHECK_INTERVAL_MS)
                if (!NetworkClient.testConnection()) {
                    Log.w(TAG, "Connection health check failed")
                    onConnectionLost()
                }
            }
        }
    }

    fun stop() {
        monitorJob?.cancel()
        monitorJob = null
    }

    fun release() {
        monitorScope.cancel()
        stop()
    }
}


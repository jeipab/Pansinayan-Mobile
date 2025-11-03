package com.fslr.pansinayan.utils

import android.content.Context
import android.content.SharedPreferences
import android.util.Log

/**
 * Manages online/offline mode switching.
 */
class ModeManager(context: Context) {

    companion object {
        private const val TAG = "ModeManager"
        private const val PREFS_NAME = "pansinayan_mode"
        private const val KEY_MODE = "inference_mode"
        private const val MODE_ONLINE = "online"
        private const val MODE_OFFLINE = "offline"
    }

    enum class InferenceMode {
        ONLINE,   // Use remote server
        OFFLINE   // Use local PyTorch models
    }

    private val prefs: SharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun getCurrentMode(): InferenceMode {
        val mode = prefs.getString(KEY_MODE, MODE_OFFLINE)
        return if (mode == MODE_ONLINE) InferenceMode.ONLINE else InferenceMode.OFFLINE
    }

    fun setMode(mode: InferenceMode) {
        val modeStr = if (mode == InferenceMode.ONLINE) MODE_ONLINE else MODE_OFFLINE
        prefs.edit().putString(KEY_MODE, modeStr).apply()
        Log.i(TAG, "Mode set to: $mode")
    }

    fun isOnlineMode(): Boolean = getCurrentMode() == InferenceMode.ONLINE
    fun isOfflineMode(): Boolean = getCurrentMode() == InferenceMode.OFFLINE
}
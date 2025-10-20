package com.fslr.pansinayan.services

import android.app.*
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.net.Uri
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.Log
import androidx.core.app.NotificationCompat
import com.fslr.pansinayan.R
import com.fslr.pansinayan.activities.MainActivity
import com.fslr.pansinayan.utils.ScreenRecorder

class ScreenRecordService : Service() {
    companion object {
        private const val TAG = "ScreenRecordService"
        private const val NOTIFICATION_ID = 1001
        private const val CHANNEL_ID = "screen_recording_channel"
        private const val CHANNEL_NAME = "Screen Recording"
        
        const val ACTION_START_RECORDING = "com.fslr.pansinayan.START_RECORDING"
        const val ACTION_STOP_RECORDING = "com.fslr.pansinayan.STOP_RECORDING"
        
        const val EXTRA_RESULT_CODE = "result_code"
        const val EXTRA_DATA = "data"
        const val EXTRA_WIDTH = "width"
        const val EXTRA_HEIGHT = "height"
        const val EXTRA_DPI = "dpi"
        
        const val BROADCAST_RECORDING_STOPPED = "com.fslr.pansinayan.RECORDING_STOPPED"
        const val EXTRA_RECORDING_URI = "recording_uri"
        const val EXTRA_ERROR_MESSAGE = "error_message"
        
        @Volatile
        private var isServiceRunning = false
        
        fun isRunning(): Boolean = isServiceRunning
    }

    private lateinit var screenRecorder: ScreenRecorder
    private val handler = Handler(Looper.getMainLooper())
    private var startTime: Long = 0
    private var notificationBuilder: NotificationCompat.Builder? = null

    override fun onCreate() {
        super.onCreate()
        screenRecorder = ScreenRecorder(this)
        createNotificationChannel()
        isServiceRunning = true
        Log.d(TAG, "Service created")
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent == null) {
            Log.e(TAG, "Received null intent")
            stopSelf()
            return START_NOT_STICKY
        }

        when (intent.action) {
            ACTION_START_RECORDING -> {
                val resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, 0)
                val data = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    intent.getParcelableExtra(EXTRA_DATA, Intent::class.java)
                } else {
                    @Suppress("DEPRECATION")
                    intent.getParcelableExtra(EXTRA_DATA) as? Intent
                }
                val width = intent.getIntExtra(EXTRA_WIDTH, 1920)
                val height = intent.getIntExtra(EXTRA_HEIGHT, 1080)
                val dpi = intent.getIntExtra(EXTRA_DPI, 1)
                
                Log.d(TAG, "Received start recording: resultCode=$resultCode, data=$data, size=${width}x${height}, dpi=$dpi")
                
                if (resultCode != android.app.Activity.RESULT_OK || data == null) {
                    Log.e(TAG, "Invalid MediaProjection permission data: resultCode=$resultCode, data=$data")
                    sendErrorBroadcast("Invalid MediaProjection permission")
                    stopSelf()
                } else {
                    startForeground()
                    startRecordingInternal(resultCode, data, width, height, dpi)
                }
            }
            ACTION_STOP_RECORDING -> {
                stopRecordingInternal()
            }
            else -> {
                Log.w(TAG, "Unknown action: ${intent.action}")
            }
        }
        
        return START_NOT_STICKY
    }

    private fun startForeground() {
        val notification = createNotification()
        
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                CHANNEL_NAME,
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Screen recording notifications"
                setShowBadge(false)
                enableVibration(false)
                setSound(null, null)
            }
            
            val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.createNotificationChannel(channel)
        }
    }

    private fun createNotification(): Notification {
        // Create stop recording intent
        val stopIntent = Intent(this, ScreenRecordService::class.java).apply {
            action = ACTION_STOP_RECORDING
        }
        val stopPendingIntent = PendingIntent.getService(
            this,
            0,
            stopIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        // Create intent to open app
        val appIntent = Intent(this, MainActivity::class.java)
        val appPendingIntent = PendingIntent.getActivity(
            this,
            0,
            appIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        // Create notification builder
        notificationBuilder = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Recording Screen")
            .setContentText("Tap to stop recording")
            .setSmallIcon(R.drawable.ic_record)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setOngoing(true)
            .setShowWhen(true)
            .setWhen(System.currentTimeMillis())
            .addAction(
                android.R.drawable.ic_media_pause,
                "Stop",
                stopPendingIntent
            )
            .setContentIntent(appPendingIntent)
        
        return notificationBuilder!!.build()
    }

    private fun updateNotification(message: String) {
        notificationBuilder?.let { builder ->
            builder.setContentText(message)
            val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.notify(NOTIFICATION_ID, builder.build())
        }
    }

    private fun startRecordingInternal(resultCode: Int, data: Intent, width: Int, height: Int, dpi: Int) {
        if (screenRecorder.isRecording()) {
            Log.w(TAG, "Already recording")
            return
        }

        startTime = System.currentTimeMillis()
        
        screenRecorder.startRecording(resultCode, data, width, height, dpi) { success, errorMessage ->
            if (success) {
                Log.i(TAG, "Recording started successfully")
                updateNotification("Recording in progress...")
                startDurationUpdater()
            } else {
                Log.e(TAG, "Failed to start recording: $errorMessage")
                sendErrorBroadcast(errorMessage ?: "Failed to start recording")
                stopSelf()
            }
        }
    }

    private fun stopRecordingInternal() {
        if (!screenRecorder.isRecording()) {
            Log.w(TAG, "Not recording")
            stopSelf()
            return
        }

        updateNotification("Stopping recording...")
        
        screenRecorder.stopRecording { uri ->
            if (uri != null) {
                Log.i(TAG, "Recording saved successfully: $uri")
                sendSuccessBroadcast(uri)
            } else {
                Log.e(TAG, "Failed to save recording")
                sendErrorBroadcast("Failed to save recording")
            }
            stopSelf()
        }
    }

    private fun startDurationUpdater() {
        handler.postDelayed(object : Runnable {
            override fun run() {
                if (screenRecorder.isRecording()) {
                    val duration = System.currentTimeMillis() - startTime
                    val seconds = (duration / 1000) % 60
                    val minutes = (duration / (1000 * 60)) % 60
                    val hours = duration / (1000 * 60 * 60)
                    
                    val timeString = if (hours > 0) {
                        String.format("%02d:%02d:%02d", hours, minutes, seconds)
                    } else {
                        String.format("%02d:%02d", minutes, seconds)
                    }
                    
                    updateNotification("Recording: $timeString")
                    handler.postDelayed(this, 1000)
                }
            }
        }, 1000)
    }

    private fun sendSuccessBroadcast(uri: Uri) {
        val intent = Intent(BROADCAST_RECORDING_STOPPED).apply {
            putExtra(EXTRA_RECORDING_URI, uri.toString())
            setPackage(packageName)
        }
        sendBroadcast(intent)
    }

    private fun sendErrorBroadcast(errorMessage: String) {
        val intent = Intent(BROADCAST_RECORDING_STOPPED).apply {
            putExtra(EXTRA_ERROR_MESSAGE, errorMessage)
            setPackage(packageName)
        }
        sendBroadcast(intent)
    }

    override fun onDestroy() {
        super.onDestroy()
        handler.removeCallbacksAndMessages(null)
        
        // Make sure recording is stopped
        if (screenRecorder.isRecording()) {
            screenRecorder.stopRecording { }
        }
        
        isServiceRunning = false
        Log.d(TAG, "Service destroyed")
    }

    override fun onBind(intent: Intent?): IBinder? = null
}

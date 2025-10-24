package com.fslr.pansinayan.activities

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.graphics.Color
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.DisplayMetrics
import android.util.Log
import android.view.View
import android.widget.RadioGroup
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.google.android.material.switchmaterial.SwitchMaterial
import com.fslr.pansinayan.R
import com.fslr.pansinayan.database.AppDatabase
import com.fslr.pansinayan.database.RecognitionHistory
import com.fslr.pansinayan.recognition.ContinuousRecognitionManager
import com.fslr.pansinayan.recognition.ContinuousTranscript
import com.fslr.pansinayan.services.ScreenRecordService
import com.fslr.pansinayan.views.OverlayView
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.*

/**
 * Main activity for continuous CTC-based sign language recognition.
 * 
 * UI Components:
 * - Camera preview (full screen)
 * - Continuous transcript display (scrollable)
 * - CTC model selection (Transformer CTC / GRU CTC)
 * - Skeleton overlay toggle
 * - Recording controls
 * - Debug information (optional)
 * 
 * Lifecycle:
 * - onCreate: Initialize components, request permissions
 * - onResume: Start continuous recognition pipeline
 * - onPause: Pause recognition
 * - onDestroy: Release all resources
 */
class MainActivity : AppCompatActivity() {
    companion object {
        private const val TAG = "MainActivity"
        private const val CAMERA_PERMISSION_REQUEST = 101
        private const val AUDIO_PERMISSION_REQUEST = 102
        private const val SCREEN_RECORD_REQUEST = 103
    }

    // UI Components (bind these in onCreate after setContentView)
    private lateinit var previewView: PreviewView
    private lateinit var transcriptTextView: TextView
    private lateinit var statsTextView: TextView
    private lateinit var debugInfoTextView: TextView
    private lateinit var statsCard: View
    private lateinit var debugInfoCard: View
    private lateinit var overlayView: OverlayView
    private lateinit var occlusionIndicator: View
    private lateinit var skeletonToggle: SwitchMaterial
    private lateinit var fabRecord: FloatingActionButton
    private lateinit var fabSwitchCamera: FloatingActionButton
    private lateinit var fabBack: FloatingActionButton
    private lateinit var radioModelSelection: RadioGroup
    
    private var debugModeEnabled = false
    private var longPressStartTime = 0L
    private val longPressThreshold = 500L

    // Continuous recognition pipeline
    private lateinit var continuousRecognitionManager: ContinuousRecognitionManager

    // Database
    private lateinit var database: AppDatabase

    // Current CTC model selection
    private var currentModel = "Transformer CTC"

    // Recording state
    private var isRecording = false
    private var recordingStartTime = 0L
    private var pendingRecordingData: Pair<Int, Intent?>? = null

    // Broadcast receiver for recording status
    private val recordingStatusReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            when (intent?.action) {
                ScreenRecordService.BROADCAST_RECORDING_STARTED -> {
                    Log.i(TAG, "Recording started broadcast received")
                    isRecording = true
                    updateRecordingUI(true)
                    
                    // Notify pipeline that recording has started
                    if (::continuousRecognitionManager.isInitialized) {
                        continuousRecognitionManager.onRecordingStarted()
                    }
                }
                ScreenRecordService.BROADCAST_RECORDING_STOPPED -> {
                    Log.i(TAG, "Recording stopped broadcast received")
                    isRecording = false
                    updateRecordingUI(false)
                    
                    val uriString = intent.getStringExtra(ScreenRecordService.EXTRA_RECORDING_URI)
                    val errorMessage = intent.getStringExtra(ScreenRecordService.EXTRA_ERROR_MESSAGE)
                    
                    val duration = (System.currentTimeMillis() - recordingStartTime) / 1000
                    
                    if (uriString != null) {
                        val uri = Uri.parse(uriString)
                        Toast.makeText(
                            this@MainActivity,
                            "Recording saved (${duration}s)\nLocation: Movies/Pansinayan",
                            Toast.LENGTH_LONG
                        ).show()
                        Log.i(TAG, "Recording saved: $uri")
                    } else {
                        Toast.makeText(
                            this@MainActivity,
                            "Failed to save recording${errorMessage?.let { ": $it" } ?: ""}",
                            Toast.LENGTH_LONG
                        ).show()
                        Log.e(TAG, "Failed to save recording: $errorMessage")
                    }
                    
                    // Notify pipeline that recording has stopped
                    if (::continuousRecognitionManager.isInitialized) {
                        continuousRecognitionManager.onRecordingStopped()
                    }
                }
            }
        }
    }

    // Continuous transcript management
    private val transcriptHistory = mutableListOf<String>()
    private val maxTranscriptLength = 20

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        // Bind UI components
        previewView = findViewById(R.id.preview_view)
        transcriptTextView = findViewById(R.id.transcript_text)
        statsTextView = findViewById(R.id.stats_text)
        debugInfoTextView = findViewById(R.id.debug_info_text)
        statsCard = findViewById(R.id.stats_card)
        debugInfoCard = findViewById(R.id.debug_info_card)
        overlayView = findViewById(R.id.overlay_view)
        occlusionIndicator = findViewById(R.id.occlusion_indicator)
        skeletonToggle = findViewById(R.id.toggle_skeleton)
        fabRecord = findViewById(R.id.fab_record)
        fabSwitchCamera = findViewById(R.id.fab_switch_camera)
        fabBack = findViewById(R.id.fab_back)
        radioModelSelection = findViewById(R.id.radio_model_selection)

        // Initialize database
        database = AppDatabase.getDatabase(this)

        // Register broadcast receiver for recording status
        registerRecordingStatusReceiver()

        // Set up UI listeners
        setupUIListeners()

        // Check camera permission
        if (hasCameraPermission()) {
            setupContinuousRecognitionPipeline()
        } else {
            requestCameraPermission()
        }
        
        // Check if recording service is already running
        isRecording = ScreenRecordService.isRunning()
        if (isRecording) {
            updateRecordingUI(true)
        }
    }

    /**
     * Register broadcast receiver for recording status updates.
     */
    @Suppress("UnspecifiedRegisterReceiverFlag")
    private fun registerRecordingStatusReceiver() {
        val filter = IntentFilter().apply {
            addAction(ScreenRecordService.BROADCAST_RECORDING_STARTED)
            addAction(ScreenRecordService.BROADCAST_RECORDING_STOPPED)
        }
        
        // Properly handles Android 13+ requirement with version check
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(recordingStatusReceiver, filter, Context.RECEIVER_NOT_EXPORTED)
        } else {
            registerReceiver(recordingStatusReceiver, filter)
        }
    }

    private fun setupUIListeners() {
        skeletonToggle.isChecked = true
        overlayView.visibility = View.VISIBLE
        
        skeletonToggle.setOnCheckedChangeListener { _, isChecked ->
            overlayView.visibility = if (isChecked) View.VISIBLE else View.GONE
            Log.i(TAG, "Skeleton overlay: ${if (isChecked) "visible" else "hidden"}")
        }
        
        skeletonToggle.setOnTouchListener { _, event ->
            when (event.action) {
                android.view.MotionEvent.ACTION_DOWN -> {
                    longPressStartTime = System.currentTimeMillis()
                    false
                }
                android.view.MotionEvent.ACTION_UP -> {
                    val pressDuration = System.currentTimeMillis() - longPressStartTime
                    if (pressDuration >= longPressThreshold) {
                        toggleDebugMode()
                        true
                    } else {
                        false
                    }
                }
                android.view.MotionEvent.ACTION_CANCEL -> {
                    longPressStartTime = 0L
                    false
                }
                else -> false
            }
        }

        // CTC Model selection
        radioModelSelection.setOnCheckedChangeListener { _, checkedId ->
            when (checkedId) {
                R.id.radio_transformer_ctc -> {
                    currentModel = "Transformer CTC"
                    switchCTCModel("ctc/sign_transformer_ctc_fp16.tflite")
                }
                R.id.radio_gru_ctc -> {
                    currentModel = "GRU CTC"
                    switchCTCModel("ctc/mediapipe_gru_ctc_fp16.tflite")
                }
            }
        }

        // Recording button
        fabRecord.setOnClickListener {
            toggleRecording()
        }

        // Camera switch button
        fabSwitchCamera.setOnClickListener {
            switchCamera()
        }

        // Back button
        fabBack.setOnClickListener {
            onBackPressed()
        }
    }

    private fun toggleDebugMode() {
        debugModeEnabled = !debugModeEnabled
        statsCard.visibility = if (debugModeEnabled) View.VISIBLE else View.GONE
        debugInfoCard.visibility = if (debugModeEnabled) View.VISIBLE else View.GONE
        overlayView.setDebugMode(debugModeEnabled)
        
        val status = if (debugModeEnabled) "enabled" else "disabled"
        Toast.makeText(this, "Debug mode $status", Toast.LENGTH_SHORT).show()
        Log.i(TAG, "Debug mode $status")
    }

    private fun switchCTCModel(modelPath: String) {
        Toast.makeText(this, "Switching to $currentModel model...", Toast.LENGTH_SHORT).show()
        Log.i(TAG, "Switching CTC model to: $modelPath")
        
        // TODO: Implement actual CTC model switching in ContinuousRecognitionManager
        // For now, just show a toast
        lifecycleScope.launch {
            delay(500)
            Toast.makeText(this@MainActivity, "$currentModel model active", Toast.LENGTH_SHORT).show()
        }
    }

    /**
     * Toggle screen recording on/off.
     */
    private fun toggleRecording() {
        if (!isRecording) {
            // Check audio permission before starting recording
            if (!hasAudioPermission()) {
                requestAudioPermission()
                return
            }
            requestScreenRecordPermission()
        } else {
            stopRecording()
        }
    }

    /**
     * Request screen recording permission.
     */
    private fun requestScreenRecordPermission() {
        val mediaProjectionManager = getSystemService(Context.MEDIA_PROJECTION_SERVICE) 
            as android.media.projection.MediaProjectionManager
        val intent = mediaProjectionManager.createScreenCaptureIntent()
        @Suppress("DEPRECATION")
        startActivityForResult(intent, SCREEN_RECORD_REQUEST)
    }

    /**
     * Start screen recording after permission granted.
     */
    private fun startRecording(resultCode: Int, data: Intent?) {
        recordingStartTime = System.currentTimeMillis()
        
        // Store data in case we need to retry
        pendingRecordingData = Pair(resultCode, data)
        
        // Get display metrics from Activity
        val metrics = DisplayMetrics()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            display?.getRealMetrics(metrics)
        } else {
            @Suppress("DEPRECATION")
            windowManager.defaultDisplay.getRealMetrics(metrics)
        }
        
        // Ensure even dimensions (required for H.264 encoding)
        val width = if (metrics.widthPixels % 2 == 0) metrics.widthPixels else metrics.widthPixels - 1
        val height = if (metrics.heightPixels % 2 == 0) metrics.heightPixels else metrics.heightPixels - 1
        
        Log.d(TAG, "Starting recording with: resultCode=$resultCode, dimensions=${width}x${height}, dpi=${metrics.densityDpi}")
        
        // Start the service with all necessary data
        val serviceIntent = Intent(this, ScreenRecordService::class.java).apply {
            action = ScreenRecordService.ACTION_START_RECORDING
            putExtra(ScreenRecordService.EXTRA_RESULT_CODE, resultCode)
            putExtra(ScreenRecordService.EXTRA_DATA, data)
            putExtra(ScreenRecordService.EXTRA_WIDTH, width)
            putExtra(ScreenRecordService.EXTRA_HEIGHT, height)
            putExtra(ScreenRecordService.EXTRA_DPI, metrics.densityDpi)
        }
        
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(serviceIntent)
        } else {
            startService(serviceIntent)
        }
        
        // UI will be updated when we receive the broadcast
        Toast.makeText(this, "Starting recording...", Toast.LENGTH_SHORT).show()
    }

    /**
     * Stop screen recording.
     */
    private fun stopRecording() {
        val serviceIntent = Intent(this, ScreenRecordService::class.java).apply {
            action = ScreenRecordService.ACTION_STOP_RECORDING
        }
        startService(serviceIntent)
        
        Toast.makeText(this, "Stopping recording...", Toast.LENGTH_SHORT).show()
    }

    /**
     * Update UI based on recording state.
     */
    private fun updateRecordingUI(recording: Boolean) {
        if (recording) {
            // Change icon to stop/square
            fabRecord.setImageResource(android.R.drawable.ic_media_pause)
            fabRecord.backgroundTintList = ContextCompat.getColorStateList(this, android.R.color.holo_red_dark)
        } else {
            // Change icon back to record
            fabRecord.setImageResource(R.drawable.ic_record)
            fabRecord.backgroundTintList = ContextCompat.getColorStateList(this, R.color.primary_blue)
        }
    }

    /**
     * Switch between front and back camera.
     */
    private fun switchCamera() {
        if (::continuousRecognitionManager.isInitialized) {
            continuousRecognitionManager.switchCamera()
            val cameraName = if (continuousRecognitionManager.isFrontCamera()) "Front" else "Back"
            Toast.makeText(this, "$cameraName camera", Toast.LENGTH_SHORT).show()
            Log.i(TAG, "Switched to $cameraName camera")
        }
    }

    /**
     * Set up the continuous CTC recognition pipeline with callbacks.
     */
    private fun setupContinuousRecognitionPipeline() {
        try {
            continuousRecognitionManager = ContinuousRecognitionManager(
                context = this,
                lifecycleOwner = this,
                previewView = previewView,
                onTranscriptUpdate = { continuousTranscript ->
                    handleContinuousTranscript(continuousTranscript)
                },
                onFrameUpdate = { keypoints, imageWidth, imageHeight, isOccluded ->
                    handleFrameUpdate(keypoints, imageWidth, imageHeight, isOccluded)
                }
            )

            continuousRecognitionManager.initialize()
            Log.i(TAG, "Continuous CTC recognition pipeline initialized")

            // Start periodic stats update
            startStatsUpdater()

        } catch (e: Exception) {
            Log.e(TAG, "Failed to setup CTC pipeline", e)
            Toast.makeText(this, "Failed to initialize continuous recognition pipeline", Toast.LENGTH_LONG).show()
        }
    }

    private fun handleFrameUpdate(keypoints: FloatArray?, imageWidth: Int, imageHeight: Int, isOccluded: Boolean) {
        overlayView.setKeypoints(keypoints, imageWidth, imageHeight)
        occlusionIndicator.setBackgroundColor(if (isOccluded) Color.RED else Color.GREEN)
        
        if (debugModeEnabled) {
            updateDebugInfo(keypoints)
        }
    }
    
    private fun updateDebugInfo(keypoints: FloatArray?) {
        val validCount = keypoints?.let { countValidKeypoints(it) } ?: 0
        val arraySize = keypoints?.size ?: 0
        debugInfoTextView.text = "Valid keypoints: $validCount/89\nKeypoint array size: $arraySize"
    }
    
    private fun countValidKeypoints(kp: FloatArray): Int {
        var count = 0
        for (i in 0 until minOf(89, kp.size / 2)) {
            val x = kp[i * 2]
            val y = kp[i * 2 + 1]
            if (!(x == 0f && y == 0f) && x in 0.0f..1.0f && y in 0.0f..1.0f) {
                count++
            }
        }
        return count
    }

    /**
     * Handle continuous transcript updates from CTC recognition.
     * Called on main thread.
     */
    private fun handleContinuousTranscript(transcript: ContinuousTranscript) {
        Log.i(TAG, "Continuous transcript: '${transcript.currentPhrase}' (${transcript.phraseCount} phrases)")

        // Update transcript display with full continuous text
        transcriptTextView.text = transcript.fullTranscript

        // Add to transcript history for database storage
        transcriptHistory.add(transcript.currentPhrase)
        if (transcriptHistory.size > maxTranscriptLength) {
            transcriptHistory.removeAt(0)
        }

        // Save to database
        saveTranscriptToDatabase(transcript)

        // Optional: Trigger animation or sound
        animateTranscriptUpdate()
    }

    /**
     * Save continuous transcript to database.
     */
    private fun saveTranscriptToDatabase(transcript: ContinuousTranscript) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val history = RecognitionHistory(
                    timestamp = transcript.timestamp,
                    glossLabel = transcript.currentPhrase,
                    categoryLabel = "CONTINUOUS", // CTC doesn't use categories
                    occlusionStatus = if (occlusionIndicator.solidColor == Color.RED) "Occluded" else "Not Occluded",
                    modelUsed = currentModel
                )
                database.historyDao().insert(history)
                Log.d(TAG, "Saved continuous transcript to database: '${transcript.currentPhrase}'")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to save continuous transcript to database", e)
            }
        }
    }

    /**
     * Animate transcript update (optional).
     */
    private fun animateTranscriptUpdate() {
        // TODO: Add fade-in animation or scale animation
        transcriptTextView.animate()
            .alpha(0f)
            .alpha(1f)
            .setDuration(300)
            .start()
    }

    /**
     * Start periodic stats updater (for debugging).
     */
    private fun startStatsUpdater() {
        lifecycleScope.launch(Dispatchers.Main) {
            while (true) {
                delay(2000)  // Update every 2 seconds
                
                try {
                    val stats = continuousRecognitionManager.getStats()
                    statsTextView.text = stats.toString()
                } catch (e: Exception) {
                    // Ignore if pipeline not ready
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        
        if (hasCameraPermission() && ::continuousRecognitionManager.isInitialized) {
            continuousRecognitionManager.start()
            Log.i(TAG, "Continuous CTC pipeline started")
            
            // If recording is active, notify pipeline
            if (isRecording) {
                continuousRecognitionManager.onRecordingStarted()
            }
        }
    }

    override fun onPause() {
        super.onPause()
        
        if (::continuousRecognitionManager.isInitialized) {
            continuousRecognitionManager.stop()
            Log.i(TAG, "Continuous CTC pipeline stopped")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        
        // Unregister broadcast receiver
        try {
            unregisterReceiver(recordingStatusReceiver)
        } catch (e: Exception) {
            Log.e(TAG, "Error unregistering receiver", e)
        }
        
        if (::continuousRecognitionManager.isInitialized) {
            continuousRecognitionManager.release()
            Log.i(TAG, "Continuous CTC pipeline released")
        }
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        
        if (requestCode == SCREEN_RECORD_REQUEST) {
            if (resultCode == RESULT_OK && data != null) {
                Log.d(TAG, "MediaProjection permission granted")
                startRecording(resultCode, data)
            } else {
                Log.w(TAG, "MediaProjection permission denied or cancelled")
                Toast.makeText(this, "Screen recording permission required", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // ================== Permission Handling ==================

    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestCameraPermission() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.CAMERA),
            CAMERA_PERMISSION_REQUEST
        )
    }

    private fun hasAudioPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestAudioPermission() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.RECORD_AUDIO),
            AUDIO_PERMISSION_REQUEST
        )
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        
        when (requestCode) {
            CAMERA_PERMISSION_REQUEST -> {
                if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    Log.i(TAG, "Camera permission granted")
                    setupContinuousRecognitionPipeline()
                } else {
                    Log.w(TAG, "Camera permission denied")
                    Toast.makeText(
                        this,
                        "Camera permission is required for sign language recognition",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
            AUDIO_PERMISSION_REQUEST -> {
                if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    Log.i(TAG, "Audio permission granted")
                    // Now request screen capture permission
                    requestScreenRecordPermission()
                } else {
                    Log.w(TAG, "Audio permission denied")
                    Toast.makeText(
                        this,
                        "Audio permission required for recording with sound.",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        }
    }

    override fun onBackPressed() {
        super.onBackPressed()
        // Return to home instead of exiting app
        val intent = Intent(this, HomeActivity::class.java)
        intent.flags = Intent.FLAG_ACTIVITY_CLEAR_TOP
        startActivity(intent)
        finish()
    }

    // ================== UI Actions (Optional) ==================

    /**
     * Clear continuous transcript history.
     */
    private fun clearTranscript() {
        transcriptHistory.clear()
        transcriptTextView.text = "Continuous Recognition Ready..."
        Log.i(TAG, "Continuous transcript cleared")
    }

    /**
     * Toggle continuous pipeline on/off.
     */
    private fun togglePipeline() {
        if (::continuousRecognitionManager.isInitialized) {
            // Toggle between start and stop
            // Implementation depends on your UI design
        }
    }
}




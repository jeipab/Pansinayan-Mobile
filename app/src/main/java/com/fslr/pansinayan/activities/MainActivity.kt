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
import com.fslr.pansinayan.recognition.RecognitionPipeline
import com.fslr.pansinayan.recognition.RecognizedSign
import com.fslr.pansinayan.services.ScreenRecordService
import com.fslr.pansinayan.views.OverlayView
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.*

/**
 * Main activity for live sign language recognition.
 * 
 * UI Components:
 * - Camera preview (full screen or 80%)
 * - Recognition result TextView (overlay)
 * - Confidence indicator
 * - Transcript history (scrollable list)
 * - Settings button
 * 
 * Lifecycle:
 * - onCreate: Initialize components, request permissions
 * - onResume: Start recognition pipeline
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
    private lateinit var resultTextView: TextView
    private lateinit var confidenceTextView: TextView
    private lateinit var transcriptTextView: TextView
    private lateinit var statsTextView: TextView
    private lateinit var overlayView: OverlayView
    private lateinit var occlusionIndicator: View
    private lateinit var skeletonToggle: SwitchMaterial
    private lateinit var fabRecord: FloatingActionButton
    private lateinit var fabSwitchCamera: FloatingActionButton
    private lateinit var fabBack: FloatingActionButton
    private lateinit var radioModelSelection: RadioGroup

    // Recognition pipeline
    private lateinit var recognitionPipeline: RecognitionPipeline

    // Database
    private lateinit var database: AppDatabase

    // Current model selection
    private var currentModel = "Transformer"

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
                    if (::recognitionPipeline.isInitialized) {
                        recognitionPipeline.onRecordingStarted()
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
                    if (::recognitionPipeline.isInitialized) {
                        recognitionPipeline.onRecordingStopped()
                    }
                }
            }
        }
    }

    // Transcript history (keep small for on-screen display)
    private val transcript = mutableListOf<RecognizedSign>()
    private val maxTranscriptLength = 5

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        // Bind UI components
        previewView = findViewById(R.id.preview_view)
        resultTextView = findViewById(R.id.result_text)
        confidenceTextView = findViewById(R.id.confidence_text)
        transcriptTextView = findViewById(R.id.transcript_text)
        statsTextView = findViewById(R.id.stats_text)
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
            setupRecognitionPipeline()
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

    /**
     * Set up UI component listeners.
     */
    private fun setupUIListeners() {
        // Skeleton toggle - enable by default
        skeletonToggle.isChecked = true
        overlayView.visibility = View.VISIBLE
        
        skeletonToggle.setOnCheckedChangeListener { _, isChecked ->
            overlayView.visibility = if (isChecked) View.VISIBLE else View.GONE
            Log.i(TAG, "Skeleton overlay: ${if (isChecked) "visible" else "hidden"}")
        }
        
        // Long press on skeleton toggle to enable debug mode
        skeletonToggle.setOnLongClickListener {
            overlayView.setDebugMode(true)
            Toast.makeText(this, "Debug mode enabled", Toast.LENGTH_SHORT).show()
            Log.i(TAG, "Debug mode enabled for overlay")
            true
        }

        // Model selection
        radioModelSelection.setOnCheckedChangeListener { _, checkedId ->
            when (checkedId) {
                R.id.radio_transformer -> {
                    currentModel = "Transformer"
                    switchModel("sign_transformer_quant.tflite")
                }
                R.id.radio_gru -> {
                    currentModel = "GRU"
                    switchModel("sign_mediapipe_gru_quant.tflite")
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

    /**
     * Switch to a different model.
     */
    private fun switchModel(modelPath: String) {
        Toast.makeText(this, "Switching to $currentModel model...", Toast.LENGTH_SHORT).show()
        Log.i(TAG, "Switching model to: $modelPath")
        
        // TODO: Implement actual model switching in RecognitionPipeline
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
        if (::recognitionPipeline.isInitialized) {
            recognitionPipeline.switchCamera()
            val cameraName = if (recognitionPipeline.isFrontCamera()) "Front" else "Back"
            Toast.makeText(this, "$cameraName camera", Toast.LENGTH_SHORT).show()
            Log.i(TAG, "Switched to $cameraName camera")
        }
    }

    /**
     * Set up the recognition pipeline with callbacks.
     */
    private fun setupRecognitionPipeline() {
        try {
            recognitionPipeline = RecognitionPipeline(
                context = this,
                lifecycleOwner = this,
                previewView = previewView,
                onSignRecognized = { recognizedSign ->
                    handleRecognizedSign(recognizedSign)
                },
                onFrameUpdate = { keypoints, imageWidth, imageHeight, isOccluded ->
                    handleFrameUpdate(keypoints, imageWidth, imageHeight, isOccluded)
                }
            )

            recognitionPipeline.initialize()
            Log.i(TAG, "Recognition pipeline initialized")

            // Start periodic stats update
            startStatsUpdater()

        } catch (e: Exception) {
            Log.e(TAG, "Failed to setup pipeline", e)
            Toast.makeText(this, "Failed to initialize recognition pipeline", Toast.LENGTH_LONG).show()
        }
    }

    /**
     * Handle frame updates (keypoints and occlusion status).
     * Called on main thread.
     */
    private fun handleFrameUpdate(keypoints: FloatArray?, imageWidth: Int, imageHeight: Int, isOccluded: Boolean) {
        overlayView.setKeypoints(keypoints, imageWidth, imageHeight)
        occlusionIndicator.setBackgroundColor(if (isOccluded) Color.RED else Color.GREEN)
    }

    /**
     * Handle a newly recognized sign.
     * Called on main thread.
     */
    private fun handleRecognizedSign(sign: RecognizedSign) {
        Log.i(TAG, "Recognized: ${sign.label} (${sign.confidence})")

        // Update main result display
        resultTextView.text = sign.label
        confidenceTextView.text = String.format("%.1f%%", sign.confidence * 100)

        // Update confidence color based on threshold
        val confidenceColor = when {
            sign.confidence >= 0.8f -> Color.GREEN
            sign.confidence >= 0.6f -> Color.YELLOW
            else -> Color.RED
        }
        confidenceTextView.setTextColor(confidenceColor)

        // Add to temporary transcript (for on-screen display)
        transcript.add(sign)
        if (transcript.size > maxTranscriptLength) {
            transcript.removeAt(0)
        }

        // Update transcript display
        updateTranscriptDisplay()

        // Save to database (will be implemented in next step)
        saveToDatabase(sign)

        // Optional: Trigger animation or sound
        animateRecognition()
    }

    /**
     * Save recognized sign to database.
     */
    private fun saveToDatabase(sign: RecognizedSign) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val history = RecognitionHistory(
                    timestamp = sign.timestamp,
                    glossLabel = sign.label,
                    categoryLabel = "Unknown", // TODO: Add category mapping
                    occlusionStatus = if (occlusionIndicator.solidColor == Color.RED) "Occluded" else "Not Occluded",
                    modelUsed = currentModel
                )
                database.historyDao().insert(history)
                Log.d(TAG, "Saved to database: ${sign.label}")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to save to database", e)
            }
        }
    }

    /**
     * Update transcript TextView with recent recognitions.
     */
    private fun updateTranscriptDisplay() {
        val dateFormat = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
        val transcriptText = transcript.joinToString("\n") { sign ->
            val time = dateFormat.format(Date(sign.timestamp))
            "$time - ${sign.label} (${String.format("%.1f%%", sign.confidence * 100)})"
        }
        transcriptTextView.text = transcriptText
    }

    /**
     * Animate recognition result (optional).
     */
    private fun animateRecognition() {
        // TODO: Add fade-in animation or scale animation
        resultTextView.animate()
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
                    val stats = recognitionPipeline.getStats()
                    statsTextView.text = stats.toString()
                } catch (e: Exception) {
                    // Ignore if pipeline not ready
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        
        if (hasCameraPermission() && ::recognitionPipeline.isInitialized) {
            recognitionPipeline.start()
            Log.i(TAG, "Pipeline started")
            
            // If recording is active, notify pipeline
            if (isRecording) {
                recognitionPipeline.onRecordingStarted()
            }
        }
    }

    override fun onPause() {
        super.onPause()
        
        if (::recognitionPipeline.isInitialized) {
            recognitionPipeline.stop()
            Log.i(TAG, "Pipeline stopped")
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
        
        if (::recognitionPipeline.isInitialized) {
            recognitionPipeline.release()
            Log.i(TAG, "Pipeline released")
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
                    setupRecognitionPipeline()
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
     * Clear transcript history.
     */
    private fun clearTranscript() {
        transcript.clear()
        transcriptTextView.text = ""
        Log.i(TAG, "Transcript cleared")
    }

    /**
     * Toggle pipeline on/off.
     */
    private fun togglePipeline() {
        if (::recognitionPipeline.isInitialized) {
            // Toggle between start and stop
            // Implementation depends on your UI design
        }
    }
}


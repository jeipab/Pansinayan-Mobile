package com.fslr.pansinayan.activities
import com.fslr.pansinayan.R

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
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
import com.fslr.pansinayan.database.AppDatabase
import com.fslr.pansinayan.database.RecognitionHistory
import com.fslr.pansinayan.recognition.RecognitionPipeline
import com.fslr.pansinayan.recognition.RecognizedSign
import com.fslr.pansinayan.views.OverlayView
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
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
    private lateinit var radioModelSelection: RadioGroup

    // Recognition pipeline
    private lateinit var recognitionPipeline: RecognitionPipeline

    // Database
    private lateinit var database: AppDatabase

    // Current model selection
    private var currentModel = "Transformer"

    // Recording state
    private var isRecording = false

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
        radioModelSelection = findViewById(R.id.radio_model_selection)

        // Initialize database
        database = AppDatabase.getDatabase(this)

        // Set up UI listeners
        setupUIListeners()

        // Check camera permission
        if (hasCameraPermission()) {
            setupRecognitionPipeline()
        } else {
            requestCameraPermission()
        }
    }

    /**
     * Set up UI component listeners.
     */
    private fun setupUIListeners() {
        // Skeleton toggle
        skeletonToggle.setOnCheckedChangeListener { _, isChecked ->
            overlayView.visibility = if (isChecked) View.VISIBLE else View.GONE
            Log.i(TAG, "Skeleton overlay: ${if (isChecked) "visible" else "hidden"}")
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
        isRecording = !isRecording
        
        if (isRecording) {
            // Start recording
            fabRecord.setImageResource(android.R.drawable.ic_media_pause)
            Toast.makeText(this, "Recording started (feature in progress)", Toast.LENGTH_SHORT).show()
            Log.i(TAG, "Recording started")
            
            // TODO: Implement actual screen recording with MediaProjectionManager
        } else {
            // Stop recording
            fabRecord.setImageResource(R.drawable.ic_record)
            Toast.makeText(this, "Recording stopped", Toast.LENGTH_SHORT).show()
            Log.i(TAG, "Recording stopped")
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
                onFrameUpdate = { keypoints, isOccluded ->
                    handleFrameUpdate(keypoints, isOccluded)
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
    private fun handleFrameUpdate(keypoints: FloatArray?, isOccluded: Boolean) {
        // Update skeleton overlay
        if (skeletonToggle.isChecked) {
            overlayView.setKeypoints(keypoints)
        }

        // Update occlusion indicator
        occlusionIndicator.setBackgroundColor(
            if (isOccluded) Color.RED else Color.GREEN
        )
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
        
        if (::recognitionPipeline.isInitialized) {
            recognitionPipeline.release()
            Log.i(TAG, "Pipeline released")
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

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        
        if (requestCode == CAMERA_PERMISSION_REQUEST) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Log.i(TAG, "Camera permission granted")
                setupRecognitionPipeline()
            } else {
                Log.w(TAG, "Camera permission denied")
                // TODO: Show explanation dialog
            }
        }
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


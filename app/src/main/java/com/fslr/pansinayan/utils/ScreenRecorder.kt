package com.fslr.pansinayan.utils

import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.MediaRecorder
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.util.Log
import java.io.File
import java.io.FileDescriptor
import java.text.SimpleDateFormat
import java.util.*

class ScreenRecorder(private val context: Context) {
    companion object {
        private const val TAG = "ScreenRecorder"
        private const val VIDEO_BITRATE = 8_000_000 // 8 Mbps
        private const val VIDEO_FRAME_RATE = 30
        private const val AUDIO_BITRATE = 128_000 // 128 kbps
        private const val AUDIO_SAMPLE_RATE = 44100
    }

    private var mediaProjection: MediaProjection? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var mediaRecorder: MediaRecorder? = null
    private var recordingFile: File? = null
    private var contentUri: Uri? = null
    private var fileDescriptor: FileDescriptor? = null
    
    private val mediaProjectionManager: MediaProjectionManager by lazy {
        context.getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
    }
    
    private val handler = Handler(Looper.getMainLooper())
    private var isRecording = false

    fun startRecording(
        resultCode: Int,
        data: Intent?,
        width: Int,
        height: Int,
        dpi: Int,
        callback: (Boolean, String?) -> Unit
    ) {
        if (isRecording) {
            Log.w(TAG, "Already recording")
            callback(false, "Already recording")
            return
        }

        if (data == null) {
            Log.e(TAG, "MediaProjection intent data is null")
            callback(false, "MediaProjection permission not granted")
            return
        }

        Log.d(TAG, "Starting recording: resultCode=$resultCode, dimensions=${width}x${height}, dpi=$dpi")

        try {
            // Step 1: Create MediaProjection with callback
            mediaProjection = mediaProjectionManager.getMediaProjection(resultCode, data)
            if (mediaProjection == null) {
                Log.e(TAG, "Failed to create MediaProjection")
                callback(false, "Failed to create MediaProjection")
                return
            }

            // Step 2: Register callback BEFORE creating VirtualDisplay
            mediaProjection?.registerCallback(object : MediaProjection.Callback() {
                override fun onStop() {
                    Log.d(TAG, "MediaProjection stopped by system")
                    stopRecordingInternal(null)
                }
            }, handler)

            // Step 3: Prepare output file/URI
            if (!prepareOutputFile()) {
                Log.e(TAG, "Failed to prepare output file")
                cleanup()
                callback(false, "Failed to prepare output file")
                return
            }

            // Step 4: Setup and prepare MediaRecorder
            if (!setupMediaRecorder(width, height)) {
                Log.e(TAG, "Failed to setup MediaRecorder")
                cleanup()
                callback(false, "Failed to setup MediaRecorder")
                return
            }

            // Step 5: Create VirtualDisplay (AFTER MediaRecorder is prepared)
            virtualDisplay = createVirtualDisplay(width, height, dpi)
            if (virtualDisplay == null) {
                Log.e(TAG, "Failed to create VirtualDisplay")
                cleanup()
                callback(false, "Failed to create VirtualDisplay")
                return
            }

            // Step 6: Start MediaRecorder
            try {
                mediaRecorder?.start()
                isRecording = true
                Log.i(TAG, "Recording started successfully")
                callback(true, null)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start MediaRecorder", e)
                cleanup()
                callback(false, "Failed to start recording: ${e.message}")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error starting recording", e)
            cleanup()
            callback(false, "Error: ${e.message}")
        }
    }

    private fun prepareOutputFile(): Boolean {
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                // Android 10+ - Use MediaStore
                val contentValues = ContentValues().apply {
                    put(MediaStore.Video.Media.DISPLAY_NAME, "Pansinayan_${getTimestamp()}.mp4")
                    put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
                    put(MediaStore.Video.Media.RELATIVE_PATH, Environment.DIRECTORY_MOVIES + "/Pansinayan")
                    put(MediaStore.Video.Media.IS_PENDING, 1)
                }

                val resolver = context.contentResolver
                contentUri = resolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, contentValues)
                
                if (contentUri != null) {
                    val pfd = resolver.openFileDescriptor(contentUri!!, "w")
                    fileDescriptor = pfd?.fileDescriptor
                    fileDescriptor != null
                } else {
                    false
                }
            } else {
                // Android 9 and below - Use File
                val moviesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES)
                val appDir = File(moviesDir, "Pansinayan")
                if (!appDir.exists() && !appDir.mkdirs()) {
                    Log.e(TAG, "Failed to create directory")
                    return false
                }
                
                recordingFile = File(appDir, "Pansinayan_${getTimestamp()}.mp4")
                recordingFile?.let {
                    it.createNewFile()
                    true
                } ?: false
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error preparing output file", e)
            false
        }
    }

    @SuppressLint("NewApi")
    private fun setupMediaRecorder(width: Int, height: Int): Boolean {
        return try {
            mediaRecorder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                MediaRecorder(context)
            } else {
                @Suppress("DEPRECATION")
                MediaRecorder()
            }

            val recorder = mediaRecorder ?: return false
            
            // Set sources FIRST
            recorder.setAudioSource(MediaRecorder.AudioSource.MIC)
            recorder.setVideoSource(MediaRecorder.VideoSource.SURFACE)
            
            // Set output format
            recorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            
            // Set video encoder settings
            recorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264)
            recorder.setVideoSize(width, height)
            recorder.setVideoFrameRate(VIDEO_FRAME_RATE)
            recorder.setVideoEncodingBitRate(VIDEO_BITRATE)
            
            // Set audio encoder settings
            recorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
            recorder.setAudioEncodingBitRate(AUDIO_BITRATE)
            recorder.setAudioSamplingRate(AUDIO_SAMPLE_RATE)
            
            // Set output file
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                fileDescriptor?.let {
                    recorder.setOutputFile(it)
                } ?: run {
                    Log.e(TAG, "FileDescriptor is null")
                    return false
                }
            } else {
                recordingFile?.absolutePath?.let {
                    recorder.setOutputFile(it)
                } ?: run {
                    Log.e(TAG, "Recording file path is null")
                    return false
                }
            }
            
            // Prepare MUST be called last
            recorder.prepare()
            true
        } catch (e: Exception) {
            Log.e(TAG, "Error setting up MediaRecorder", e)
            false
        }
    }

    private fun createVirtualDisplay(width: Int, height: Int, dpi: Int): VirtualDisplay? {
        return try {
            // Create VirtualDisplay AFTER MediaRecorder is prepared
            val surface = mediaRecorder?.surface
            if (surface == null) {
                Log.e(TAG, "MediaRecorder surface is null")
                return null
            }

            mediaProjection?.createVirtualDisplay(
                "ScreenRecorder",
                width, height, dpi,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                surface,
                null,
                handler
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error creating VirtualDisplay", e)
            null
        }
    }

    fun stopRecording(callback: (Uri?) -> Unit) {
        if (!isRecording) {
            Log.w(TAG, "Not recording")
            callback(null)
            return
        }

        stopRecordingInternal(callback)
    }

    private fun stopRecordingInternal(callback: ((Uri?) -> Unit)?) {
        try {
            isRecording = false
            
            // Step 1: Stop MediaRecorder first
            try {
                mediaRecorder?.stop()
                Log.d(TAG, "MediaRecorder stopped")
            } catch (e: Exception) {
                Log.e(TAG, "Error stopping MediaRecorder", e)
            }
            
            // Step 2: Release MediaRecorder
            try {
                mediaRecorder?.release()
                mediaRecorder = null
                Log.d(TAG, "MediaRecorder released")
            } catch (e: Exception) {
                Log.e(TAG, "Error releasing MediaRecorder", e)
            }
            
            // Step 3: Release VirtualDisplay
            try {
                virtualDisplay?.release()
                virtualDisplay = null
                Log.d(TAG, "VirtualDisplay released")
            } catch (e: Exception) {
                Log.e(TAG, "Error releasing VirtualDisplay", e)
            }
            
            // Step 4: Stop MediaProjection
            try {
                mediaProjection?.stop()
                mediaProjection = null
                Log.d(TAG, "MediaProjection stopped")
            } catch (e: Exception) {
                Log.e(TAG, "Error stopping MediaProjection", e)
            }
            
            // Step 5: Finalize the file
            val savedUri = finalizeRecording()
            
            // Step 6: Cleanup
            cleanup()
            
            // Step 7: Invoke callback
            callback?.invoke(savedUri)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping recording", e)
            cleanup()
            callback?.invoke(null)
        }
    }

    private fun finalizeRecording(): Uri? {
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                contentUri?.let { uri ->
                    // Mark file as complete
                    val contentValues = ContentValues().apply {
                        put(MediaStore.Video.Media.IS_PENDING, 0)
                    }
                    context.contentResolver.update(uri, contentValues, null, null)
                    Log.i(TAG, "Recording saved to: $uri")
                    uri
                }
            } else {
                recordingFile?.let { file ->
                    if (file.exists() && file.length() > 0) {
                        val uri = Uri.fromFile(file)
                        Log.i(TAG, "Recording saved to: ${file.absolutePath}")
                        uri
                    } else {
                        Log.e(TAG, "Recording file is empty or doesn't exist")
                        null
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error finalizing recording", e)
            null
        }
    }

    private fun cleanup() {
        try {
            mediaRecorder?.release()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing MediaRecorder during cleanup", e)
        }
        mediaRecorder = null
        
        try {
            virtualDisplay?.release()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing VirtualDisplay during cleanup", e)
        }
        virtualDisplay = null
        
        try {
            mediaProjection?.stop()
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping MediaProjection during cleanup", e)
        }
        mediaProjection = null
        
        recordingFile = null
        contentUri = null
        fileDescriptor = null
        isRecording = false
    }

    private fun getTimestamp(): String {
        return SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
    }

    fun isRecording(): Boolean = isRecording
}

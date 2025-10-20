package com.fslr.pansinayan.mediapipe

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult

class MediaPipeProcessor(
    private val context: Context,
    private val listener: KeypointListener? = null
) {
    companion object {
        private const val TAG = "MediaPipeProcessor"
        
        private val POSE_UPPER_INDICES = listOf(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22,
            23, 24
        )
        
        // Minimal face landmark indices for sign language (Option 2: ~22 points)
        // These capture the most critical facial features for non-manual markers
        // Using actual MediaPipe Face Mesh 468-point model indices
        private val MINIMAL_FACE_INDICES = listOf(
            // Lips (8 points) - Outer lip contour for mouth shapes in sign language
            // Follows natural mouth anatomy: upper lip left→right, lower lip right→left
            61,   // 0: Left mouth corner
            185,  // 1: Upper lip left-outer
            40,   // 2: Upper lip left-center
            39,   // 3: Upper lip right-center
            291,  // 4: Right mouth corner
            321,  // 5: Lower lip right-center
            17,   // 6: Lower lip bottom center
            146,  // 7: Lower lip left-center
            
            // Eyes (6 points) - For eye aperture and gaze
            33,   // 8: Left eye left corner
            133,  // 9: Left eye right corner
            159,  // 10: Left eye top center
            362,  // 11: Right eye left corner
            263,  // 12: Right eye right corner
            386,  // 13: Right eye top center
            
            // Eyebrows (6 points) - Critical for questions and emotions
            70,   // 14: Left eyebrow inner
            107,  // 15: Left eyebrow middle
            46,   // 16: Left eyebrow outer
            300,  // 17: Right eyebrow inner
            336,  // 18: Right eyebrow middle
            276,  // 19: Right eyebrow outer
            
            // Nose + Cheeks (2 points) - For facial orientation
            1,    // 20: Nose tip
            4     // 21: Nose bridge (between eyes)
        )
        // Total: 22 face landmarks × 2 = 44 values
    }

    interface KeypointListener {
        fun onKeypointsExtracted(keypoints: FloatArray?, imageWidth: Int, imageHeight: Int)
        fun onError(error: String)
    }

    private var handLandmarker: HandLandmarker? = null
    private var poseLandmarker: PoseLandmarker? = null
    private var faceLandmarker: FaceLandmarker? = null
    
    private var latestPoseResult: PoseLandmarkerResult? = null
    private var latestHandResult: HandLandmarkerResult? = null
    private var latestFaceResult: FaceLandmarkerResult? = null
    private var latestImageWidth = 0
    private var latestImageHeight = 0
    
    private var successCount = 0
    private var failureCount = 0
    private var frameCount = 0

    init {
        setupLandmarkers()
    }

    private fun setupLandmarkers() {
        try {
            val handOptions = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build())
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setNumHands(2)
                .setMinHandDetectionConfidence(0.3f)
                .setMinHandPresenceConfidence(0.3f)
                .setMinTrackingConfidence(0.3f)
                .setResultListener(this::onHandResult)
                .setErrorListener { error -> listener?.onError(error.message ?: "Hand detection error") }
                .build()

            handLandmarker = HandLandmarker.createFromOptions(context, handOptions)

            val poseOptions = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath("pose_landmarker_full.task").build())
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setMinPoseDetectionConfidence(0.3f)
                .setMinPosePresenceConfidence(0.3f)
                .setMinTrackingConfidence(0.3f)
                .setResultListener(this::onPoseResult)
                .setErrorListener { error -> listener?.onError(error.message ?: "Pose detection error") }
                .build()

            poseLandmarker = PoseLandmarker.createFromOptions(context, poseOptions)
            
            val faceOptions = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath("face_landmarker.task").build())
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setNumFaces(1)
                .setMinFaceDetectionConfidence(0.3f)
                .setMinFacePresenceConfidence(0.3f)
                .setMinTrackingConfidence(0.3f)
                .setResultListener(this::onFaceResult)
                .setErrorListener { error -> listener?.onError(error.message ?: "Face detection error") }
                .build()

            faceLandmarker = FaceLandmarker.createFromOptions(context, faceOptions)
            
            Log.i(TAG, "MediaPipe initialized successfully (Hand, Pose, Face)")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize MediaPipe", e)
            listener?.onError("Initialization failed: ${e.message}")
        }
    }

    fun detectLiveStream(imageProxy: ImageProxy, isFrontCamera: Boolean = true) {
        frameCount++
        
        if (frameCount <= 3) {
            Log.d(TAG, "detectLiveStream called: frame $frameCount, ${imageProxy.width}x${imageProxy.height}, format: ${imageProxy.format}")
        }
        
        try {
            val bitmapBuffer = Bitmap.createBitmap(
                imageProxy.width,
                imageProxy.height,
                Bitmap.Config.ARGB_8888
            )
            
            imageProxy.use { bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer) }
            imageProxy.close()

            val matrix = Matrix().apply {
                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                
                // Flip image if using front camera (matches official MediaPipe examples)
                if (isFrontCamera) {
                    postScale(-1f, 1f, imageProxy.width.toFloat(), imageProxy.height.toFloat())
                }
            }
            
            val rotatedBitmap = Bitmap.createBitmap(
                bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height, matrix, true
            )

            if (frameCount <= 3) {
                Log.d(TAG, "Bitmap created: ${rotatedBitmap.width}x${rotatedBitmap.height}, config: ${rotatedBitmap.config}")
            }

            val mpImage = BitmapImageBuilder(rotatedBitmap).build()
            val frameTime = SystemClock.uptimeMillis()

            if (poseLandmarker != null && handLandmarker != null && faceLandmarker != null) {
                poseLandmarker?.detectAsync(mpImage, frameTime)
                handLandmarker?.detectAsync(mpImage, frameTime)
                faceLandmarker?.detectAsync(mpImage, frameTime)
                
                if (frameCount <= 3) {
                    Log.d(TAG, "Sent to MediaPipe (Pose, Hand, Face) at time: $frameTime")
                }
            } else {
                Log.e(TAG, "Landmarkers are null! pose: ${poseLandmarker != null}, hand: ${handLandmarker != null}, face: ${faceLandmarker != null}")
            }
            
        } catch (e: Exception) {
            failureCount++
            Log.e(TAG, "Detection failed", e)
            listener?.onError("Detection failed: ${e.message}")
        }
    }

    private fun onPoseResult(result: PoseLandmarkerResult, input: MPImage) {
        latestPoseResult = result
        latestImageWidth = input.width
        latestImageHeight = input.height
        
        val hasPose = result.landmarks().isNotEmpty()
        if (frameCount <= 3) {
            Log.d(TAG, "Pose result: $hasPose poses detected")
        }
        
        processResults()
    }

    private fun onHandResult(result: HandLandmarkerResult, input: MPImage) {
        latestHandResult = result
        
        val hasHands = result.landmarks().isNotEmpty()
        if (frameCount <= 3) {
            Log.d(TAG, "Hand result: ${result.landmarks().size} hands detected")
        }
        
        processResults()
    }
    
    private fun onFaceResult(result: FaceLandmarkerResult, input: MPImage) {
        latestFaceResult = result
        
        val hasFaces = result.faceLandmarks().isNotEmpty()
        if (frameCount <= 3) {
            Log.d(TAG, "Face result: ${result.faceLandmarks().size} faces detected")
        }
        
        processResults()
    }

    private fun processResults() {
        val keypoints = extractKeypoints()
        
        successCount++
        listener?.onKeypointsExtracted(keypoints, latestImageWidth, latestImageHeight)
        
        if (successCount % 30 == 0) {
            Log.d(TAG, "Processed $successCount frames")
        }
    }

    private fun extractKeypoints(): FloatArray {
        // New total: 50 (pose) + 42 (left hand) + 42 (right hand) + 44 (face) = 178 values
        val keypoints = FloatArray(178) { 0f }
        var idx = 0

        // Pose landmarks (25 points × 2 = 50 values)
        val poseLandmarks = latestPoseResult?.landmarks()?.firstOrNull()
        if (poseLandmarks != null && poseLandmarks.size >= 33) {
            for (poseIdx in POSE_UPPER_INDICES) {
                val landmark = poseLandmarks[poseIdx]
                keypoints[idx++] = landmark.x()
                keypoints[idx++] = landmark.y()
            }
        } else {
            idx += 50
        }

        // Hand landmarks
        val handLandmarks = latestHandResult?.landmarks() ?: emptyList()
        val handedness = latestHandResult?.handednesses() ?: emptyList()
        
        var leftHandLandmarks: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>? = null
        var rightHandLandmarks: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>? = null

        for (i in handLandmarks.indices) {
            val hand = handLandmarks[i]
            val handLabel = handedness.getOrNull(i)?.firstOrNull()?.categoryName()
            when (handLabel) {
                "Left" -> leftHandLandmarks = hand
                "Right" -> rightHandLandmarks = hand
            }
        }

        // Left hand (21 points × 2 = 42 values)
        if (leftHandLandmarks != null && leftHandLandmarks.size >= 21) {
            for (i in 0 until 21) {
                keypoints[idx++] = leftHandLandmarks[i].x()
                keypoints[idx++] = leftHandLandmarks[i].y()
            }
        } else {
            idx += 42
        }

        // Right hand (21 points × 2 = 42 values)
        if (rightHandLandmarks != null && rightHandLandmarks.size >= 21) {
            for (i in 0 until 21) {
                keypoints[idx++] = rightHandLandmarks[i].x()
                keypoints[idx++] = rightHandLandmarks[i].y()
            }
        } else {
            idx += 42
        }

        // Face landmarks - MINIMAL SET (22 points × 2 = 44 values)
        val faceLandmarks = latestFaceResult?.faceLandmarks()?.firstOrNull()
        if (faceLandmarks != null && faceLandmarks.size >= 468) {
            // Extract only the minimal face landmark indices
            for (faceIdx in MINIMAL_FACE_INDICES) {
                val landmark = faceLandmarks[faceIdx]
                keypoints[idx++] = landmark.x()
                keypoints[idx++] = landmark.y()
            }
        } else {
            idx += 44
        }

        return keypoints
    }

    fun detectOcclusion(keypoints: FloatArray): Boolean {
        if (keypoints.size < 178) return false
        
        val criticalIndices = listOf(11, 12, 13, 14, 15, 16, 25, 26, 46, 47)
        
        var occludedCount = 0
        for (i in criticalIndices) {
            if (i * 2 + 1 >= keypoints.size) continue
            val x = keypoints[i * 2]
            val y = keypoints[i * 2 + 1]
            if (x == 0f && y == 0f) occludedCount++
        }
        
        return occludedCount > 5
    }

    fun release() {
        handLandmarker?.close()
        poseLandmarker?.close()
        faceLandmarker?.close()
        handLandmarker = null
        poseLandmarker = null
        faceLandmarker = null
        Log.i(TAG, "MediaPipe released (Hand, Pose, Face)")
    }

    fun getStats(): Pair<Int, Int> = Pair(successCount, failureCount)
}

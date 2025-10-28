package com.fslr.pansinayan.mediapipe

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.PointF
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
import kotlin.math.sqrt

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
            // Lips (8 points) - User specified landmarks for proper arc distribution
            // Upper lip: left → center → right
            // Lower lip: left → center → right
            // Corners: clean connection points
            81,   // 0: Upper lip left
            13,   // 1: Upper lip center (cupid's bow)
            311,  // 2: Upper lip right
            61,   // 3: Left corner
            178,  // 4: Lower lip left
            14,   // 5: Lower lip center
            402,  // 6: Lower lip right
            291,  // 7: Right corner
            
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
        
        // Hand-face occlusion detection constants (aligned with preprocessing pipeline)
        private const val PROXIMITY_MULTIPLIER = 1.5f
        private const val MIN_FINGERTIPS_INSIDE = 1
        private const val MIN_FACE_POINTS = 3
        private const val TEMPORAL_WINDOW_SIZE = 5
        private const val TEMPORAL_THRESHOLD = 3
        
        // Hand keypoint structure (relative indices within hand)
        private val MCP_JOINTS = listOf(5, 9, 13, 17)  // For palm center calculation
        private val FINGERTIP_INDICES = listOf(4, 8, 12, 16, 20)  // All 5 fingertips
    }
    
    /**
     * Hand features extracted for occlusion detection.
     */
    data class HandFeatures(
        val palmCenter: PointF?,
        val fingertips: List<PointF>
    )
    
    /**
     * Face features extracted for occlusion detection.
     */
    data class FaceFeatures(
        val center: PointF,
        val radiusX: Float,
        val radiusY: Float
    )

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
    
    // Temporal filtering buffer for hand-face occlusion detection
    private val occlusionHistory = ArrayDeque<Boolean>(TEMPORAL_WINDOW_SIZE)
    private val occlusionLock = Any()

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
            
            imageProxy.use { img ->
                bitmapBuffer.copyPixelsFromBuffer(img.planes[0].buffer)
            }

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
        
        if (successCount % 100 == 0) {
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

    /**
     * Detect hand-face occlusion (aligned with preprocessing pipeline).
     * 
     * This function detects when hands are covering the face, which affects
     * sign language recognition quality. Uses spatial proximity detection
     * between hand fingertips and face landmarks.
     * 
     * @param keypoints FloatArray of 178 values (89 keypoints × 2)
     * @return true if hands are covering face, false otherwise
     */
    fun detectHandFaceOcclusion(keypoints: FloatArray): Boolean {
        if (keypoints.size < 178) return false
        
        // 1. Extract hand features
        val leftHand = extractHandFeatures(keypoints, handStart = 25, handLen = 21)
        val rightHand = extractHandFeatures(keypoints, handStart = 46, handLen = 21)
        
        // 2. Extract face features
        val face = extractFaceFeatures(keypoints, faceStart = 67, faceLen = 22) ?: return false
        
        // 3. Check if fingertips are inside/near face region
        val occlusionDetected = checkHandFaceProximity(leftHand, face) || 
                                checkHandFaceProximity(rightHand, face)
        
        // 4. Apply temporal filtering (need consistent detection across frames)
        return applyTemporalFilter(occlusionDetected)
    }
    
    /**
     * Extract hand features (palm center and fingertips) from keypoints.
     */
    private fun extractHandFeatures(keypoints: FloatArray, handStart: Int, handLen: Int): HandFeatures {
        // Extract MCP joints for palm center calculation
        val mcpPoints = mutableListOf<PointF>()
        
        for (relIdx in MCP_JOINTS) {
            val absIdx = handStart + relIdx
            if (absIdx < 89) {
                val x = keypoints[absIdx * 2]
                val y = keypoints[absIdx * 2 + 1]
                if (x != 0f || y != 0f) {
                    mcpPoints.add(PointF(x, y))
                }
            }
        }
        
        // Calculate palm center (average of MCP joints)
        val palmCenter = if (mcpPoints.size >= 2) {
            val avgX = mcpPoints.map { it.x }.average().toFloat()
            val avgY = mcpPoints.map { it.y }.average().toFloat()
            PointF(avgX, avgY)
        } else null
        
        // Extract fingertips
        val fingertips = mutableListOf<PointF>()
        
        for (relIdx in FINGERTIP_INDICES) {
            val absIdx = handStart + relIdx
            if (absIdx < 89) {
                val x = keypoints[absIdx * 2]
                val y = keypoints[absIdx * 2 + 1]
                if (x != 0f || y != 0f) {
                    fingertips.add(PointF(x, y))
                }
            }
        }
        
        return HandFeatures(palmCenter, fingertips)
    }
    
    /**
     * Extract face features (center and bounding ellipse) from keypoints.
     */
    private fun extractFaceFeatures(keypoints: FloatArray, faceStart: Int, faceLen: Int): FaceFeatures? {
        val facePoints = mutableListOf<PointF>()
        
        for (i in 0 until faceLen) {
            val absIdx = faceStart + i
            if (absIdx < 89) {
                val x = keypoints[absIdx * 2]
                val y = keypoints[absIdx * 2 + 1]
                if (x != 0f || y != 0f) {
                    facePoints.add(PointF(x, y))
                }
            }
        }
        
        // Need at least MIN_FACE_POINTS landmarks to define face region
        if (facePoints.size < MIN_FACE_POINTS) return null
        
        // Calculate face center
        val centerX = facePoints.map { it.x }.average().toFloat()
        val centerY = facePoints.map { it.y }.average().toFloat()
        
        // Calculate ellipse radii (using standard deviation)
        val radiusX = facePoints.map { kotlin.math.abs(it.x - centerX) }.average().toFloat()
        val radiusY = facePoints.map { kotlin.math.abs(it.y - centerY) }.average().toFloat()
        
        return FaceFeatures(PointF(centerX, centerY), radiusX, radiusY)
    }
    
    /**
     * Check if hand fingertips are within face proximity threshold.
     */
    private fun checkHandFaceProximity(hand: HandFeatures, face: FaceFeatures): Boolean {
        var fingertipsInside = 0
        
        for (tip in hand.fingertips) {
            // Calculate normalized distance from face ellipse center
            val dx = (tip.x - face.center.x) / face.radiusX
            val dy = (tip.y - face.center.y) / face.radiusY
            val distance = sqrt(dx * dx + dy * dy)
            
            // Check if within proximity threshold
            if (distance < PROXIMITY_MULTIPLIER) {
                fingertipsInside++
            }
        }
        
        // Occlusion detected if at least MIN_FINGERTIPS_INSIDE are near face
        return fingertipsInside >= MIN_FINGERTIPS_INSIDE
    }
    
    /**
     * Apply temporal filtering to reduce flickering false positives.
     * Requires TEMPORAL_THRESHOLD out of TEMPORAL_WINDOW_SIZE frames to confirm.
     */
    private fun applyTemporalFilter(currentDetection: Boolean): Boolean {
        synchronized(occlusionLock) {
            occlusionHistory.addLast(currentDetection)
            if (occlusionHistory.size > TEMPORAL_WINDOW_SIZE) {
                occlusionHistory.removeFirst()
            }

            // Need full window for reliable detection
            if (occlusionHistory.size < TEMPORAL_WINDOW_SIZE) return false

            // Count detections in window without creating an external iterator
            var detectionCount = 0
            for (value in occlusionHistory) if (value) detectionCount++
            return detectionCount >= TEMPORAL_THRESHOLD
        }
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
    
    /**
     * Restart MediaPipe processors - useful when they get frozen/deadlocked.
     * This fully recreates all landmarkers to clear any internal deadlocks.
     */
    fun restart() {
        Log.i(TAG, "Restarting MediaPipe processors...")
        try {
            // Release old instances
            handLandmarker?.close()
            poseLandmarker?.close()
            faceLandmarker?.close()
            
            // Small delay to ensure cleanup
            Thread.sleep(200)
            
            // Recreate all landmarkers
            setupLandmarkers()
            
            Log.i(TAG, "MediaPipe processors restarted successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to restart MediaPipe processors", e)
        }
    }

    fun getStats(): Pair<Int, Int> = Pair(successCount, failureCount)
}

package com.fslr.pansinayan.mediapipe

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker.HandLandmarkerOptions
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker.PoseLandmarkerOptions

/**
 * Processes frames with MediaPipe to extract keypoints.
 * 
 * Extracts 78 keypoints total:
 * - Pose: 25 upper body points (50 values)
 * - Left Hand: 21 points (42 values)
 * - Right Hand: 21 points (42 values)
 * - Face: 11 points (22 values)
 * 
 * Output: FloatArray of 156 values (78 keypoints × 2 coordinates: x, y)
 * 
 * Usage:
 *   val processor = MediaPipeProcessor(context)
 *   val keypoints = processor.extractKeypoints(bitmap)
 *   // keypoints is FloatArray[156] or null if detection failed
 */
class MediaPipeProcessor(private val context: Context) {
    companion object {
        private const val TAG = "MediaPipeProcessor"
        
        // Upper body pose indices (25 points out of 33 total pose landmarks)
        // Indices: face (0-10), arms/shoulders (11-16), torso (17-22), hips (23-24)
        private val POSE_UPPER_INDICES = listOf(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // Face landmarks
            11, 12, 13, 14, 15, 16,             // Arms and shoulders
            17, 18, 19, 20, 21, 22,             // Torso
            23, 24                              // Hips
        )
    }

    private var handLandmarker: HandLandmarker? = null
    private var poseLandmarker: PoseLandmarker? = null
    
    private var successCount = 0
    private var failureCount = 0

    init {
        initializeMediaPipe()
    }

    /**
     * Initialize MediaPipe hand and pose landmarkers.
     */
    private fun initializeMediaPipe() {
        try {
            // Initialize Hand Landmarker
            val handOptions = HandLandmarkerOptions.builder()
                .setBaseOptions(
                    BaseOptions.builder()
                        .setModelAssetPath("hand_landmarker.task")
                        .build()
                )
                .setRunningMode(RunningMode.IMAGE)
                .setNumHands(2)  // Detect both hands
                .setMinHandDetectionConfidence(0.5f)
                .setMinHandPresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .build()

            handLandmarker = HandLandmarker.createFromOptions(context, handOptions)
            Log.i(TAG, "Hand landmarker initialized")

            // Initialize Pose Landmarker
            val poseOptions = PoseLandmarkerOptions.builder()
                .setBaseOptions(
                    BaseOptions.builder()
                        .setModelAssetPath("pose_landmarker_full.task")
                        .build()
                )
                .setRunningMode(RunningMode.IMAGE)
                .setMinPoseDetectionConfidence(0.5f)
                .setMinPosePresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .build()

            poseLandmarker = PoseLandmarker.createFromOptions(context, poseOptions)
            Log.i(TAG, "Pose landmarker initialized")

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize MediaPipe", e)
        }
    }

    /**
     * Extract keypoints from a bitmap frame.
     * 
     * Returns FloatArray of 156 values structured as:
     * - [0-49]: Pose (25 points × 2 coords)
     * - [50-91]: Left hand (21 points × 2 coords)
     * - [92-133]: Right hand (21 points × 2 coords)
     * - [134-155]: Face (11 points × 2 coords)
     * 
     * @param bitmap Input frame
     * @return FloatArray[156] or null if extraction fails
     */
    fun extractKeypoints(bitmap: Bitmap): FloatArray? {
        try {
            val mpImage = BitmapImageBuilder(bitmap).build()

            // Extract pose landmarks
            val poseResult = poseLandmarker?.detect(mpImage)
            val poseLandmarks = poseResult?.landmarks()?.firstOrNull()

            // Extract hand landmarks
            val handResult = handLandmarker?.detect(mpImage)
            val handLandmarks = handResult?.landmarks() ?: emptyList()
            val handedness = handResult?.handednesses() ?: emptyList()

            // Separate left and right hands
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

            // Build 156D keypoint array
            val keypoints = FloatArray(156) { 0f }
            var idx = 0

            // 1. Pose landmarks (25 points × 2 = 50 values)
            if (poseLandmarks != null && poseLandmarks.size >= 33) {
                for (poseIdx in POSE_UPPER_INDICES) {
                    if (poseIdx < poseLandmarks.size) {
                        val landmark = poseLandmarks[poseIdx]
                        keypoints[idx++] = landmark.x()
                        keypoints[idx++] = landmark.y()
                    } else {
                        keypoints[idx++] = 0f
                        keypoints[idx++] = 0f
                    }
                }
            } else {
                // No pose detected - fill with zeros
                idx += 50
            }

            // 2. Left hand landmarks (21 points × 2 = 42 values)
            if (leftHandLandmarks != null && leftHandLandmarks.size >= 21) {
                for (i in 0 until 21) {
                    val landmark = leftHandLandmarks[i]
                    keypoints[idx++] = landmark.x()
                    keypoints[idx++] = landmark.y()
                }
            } else {
                // No left hand detected - fill with zeros
                idx += 42
            }

            // 3. Right hand landmarks (21 points × 2 = 42 values)
            if (rightHandLandmarks != null && rightHandLandmarks.size >= 21) {
                for (i in 0 until 21) {
                    val landmark = rightHandLandmarks[i]
                    keypoints[idx++] = landmark.x()
                    keypoints[idx++] = landmark.y()
                }
            } else {
                // No right hand detected - fill with zeros
                idx += 42
            }

            // 4. Face landmarks (11 points × 2 = 22 values)
            // Note: For simplicity, using zeros. In production, you can:
            // - Use FaceLandmarker from MediaPipe
            // - Extract face points from pose landmarks (first 11 pose points)
            // - Or leave as zeros if face not critical for your signs
            idx += 22

            successCount++
            
            // Log periodically
            if (successCount % 100 == 0) {
                Log.d(TAG, "Keypoint extraction stats: $successCount success / $failureCount failures")
            }

            return keypoints

        } catch (e: Exception) {
            failureCount++
            Log.e(TAG, "Keypoint extraction failed", e)
            return null
        }
    }

    /**
     * Release MediaPipe resources.
     * Call this when done with the processor.
     */
    fun release() {
        handLandmarker?.close()
        poseLandmarker?.close()
        Log.i(TAG, "MediaPipe resources released")
    }

    /**
     * Detect if hand is occluding face (simple bounding box overlap check).
     * 
     * @param keypoints FloatArray[156] containing normalized coordinates
     * @return true if occlusion detected, false otherwise
     */
    fun detectOcclusion(keypoints: FloatArray): Boolean {
        if (keypoints.size < 156) return false

        // Face region: Use pose landmarks 0-10 (first 11 pose points)
        val faceMinX = (0 until 11).mapNotNull { i ->
            val x = keypoints[i * 2]
            if (x > 0) x else null
        }.minOrNull() ?: return false

        val faceMaxX = (0 until 11).mapNotNull { i ->
            val x = keypoints[i * 2]
            if (x > 0) x else null
        }.maxOrNull() ?: return false

        val faceMinY = (0 until 11).mapNotNull { i ->
            val y = keypoints[i * 2 + 1]
            if (y > 0) y else null
        }.minOrNull() ?: return false

        val faceMaxY = (0 until 11).mapNotNull { i ->
            val y = keypoints[i * 2 + 1]
            if (y > 0) y else null
        }.maxOrNull() ?: return false

        // Check left hand (indices 50-91, representing 21 points)
        val leftHandOccludes = checkHandOcclusion(
            keypoints, 50, faceMinX, faceMaxX, faceMinY, faceMaxY
        )

        // Check right hand (indices 92-133, representing 21 points)
        val rightHandOccludes = checkHandOcclusion(
            keypoints, 92, faceMinX, faceMaxX, faceMinY, faceMaxY
        )

        return leftHandOccludes || rightHandOccludes
    }

    /**
     * Check if a specific hand is occluding the face bounding box.
     */
    private fun checkHandOcclusion(
        keypoints: FloatArray,
        startIdx: Int,
        faceMinX: Float,
        faceMaxX: Float,
        faceMinY: Float,
        faceMaxY: Float
    ): Boolean {
        // Get hand bounding box
        val handMinX = (0 until 21).mapNotNull { i ->
            val x = keypoints[startIdx + i * 2]
            if (x > 0) x else null
        }.minOrNull() ?: return false

        val handMaxX = (0 until 21).mapNotNull { i ->
            val x = keypoints[startIdx + i * 2]
            if (x > 0) x else null
        }.maxOrNull() ?: return false

        val handMinY = (0 until 21).mapNotNull { i ->
            val y = keypoints[startIdx + i * 2 + 1]
            if (y > 0) y else null
        }.minOrNull() ?: return false

        val handMaxY = (0 until 21).mapNotNull { i ->
            val y = keypoints[startIdx + i * 2 + 1]
            if (y > 0) y else null
        }.maxOrNull() ?: return false

        // Check bounding box overlap
        val xOverlap = handMinX < faceMaxX && handMaxX > faceMinX
        val yOverlap = handMinY < faceMaxY && handMaxY > faceMinY

        return xOverlap && yOverlap
    }

    /**
     * Get extraction statistics.
     * @return Pair(successCount, failureCount)
     */
    fun getStats(): Pair<Int, Int> {
        return Pair(successCount, failureCount)
    }
}


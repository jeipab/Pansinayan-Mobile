# CTC Model Placeholder Files

This directory contains CTC-based TensorFlow Lite models for continuous sign language recognition.

## Expected Model Files

- `sign_transformer_ctc_fp16.tflite` - Transformer-based CTC model
- `mediapipe_gru_ctc_fp16.tflite` - GRU-based CTC model

## Model Specifications

- **Input Shape**: [1, 300, 178] (batch_size=1, sequence_length=300, features=178)
- **Output Shape**: [1, 300, 106] (batch_size=1, sequence_length=300, classes=105 glosses + 1 blank)
- **Quantization**: FP16 for mobile optimization
- **Keypoint Structure**: 89 keypoints × 2 coordinates = 178 features
  - Pose: 25 points × 2 = 50 values
  - Left hand: 21 points × 2 = 42 values
  - Right hand: 21 points × 2 = 42 values
  - Face: 22 points × 2 = 44 values

## Usage

The models are loaded by `CTCModelInterpreter` and used for continuous recognition via `ContinuousRecognitionManager`.

## Note

These are placeholder files. Replace with actual CTC models trained for continuous sign language recognition.

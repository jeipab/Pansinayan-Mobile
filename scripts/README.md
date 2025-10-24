# CTC Model Export Pipeline

## Quick Start

1. **Clone and setup**:

   ```bash
   git clone https://github.com/jeipab/Pansinayan-Mobile.git
   cd Pansinayan-Mobile
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   pip install -r scripts/requirements.txt
   ```

2. **Place CTC models** in project root:

   ```
   SignTransformer_best.pt    # Transformer CTC model
   MediaPipeGRU_best.pt       # GRU CTC model
   ```

3. **Export CTC models**:
   ```bash
   python scripts/export_ctc_models.py
   ```

## CTC Model Specifications

- **Input**: `[1, 300, 178]` - Keypoint sequences (89 keypoints × 2)
- **Output**: `[1, 300, 106]` - CTC logits (105 glosses + 1 blank token)
- **Blank Token**: Index 105

## Usage Options

```bash
# Basic export
python scripts/export_ctc_models.py

# Custom model paths
python scripts/export_ctc_models.py --transformer path/to/model.pt --gru path/to/model.pt

# Custom sequence length
python scripts/export_ctc_models.py --sequence-length 500

# Skip verification
python scripts/export_ctc_models.py --no-verify
```

## Output

Generated files in `exports/ctc/`:

- `sign_transformer_ctc_fp16.tflite` (~2-4 MB)
- `sign_transformer_ctc_fp32.tflite` (~4-8 MB)
- `mediapipe_gru_ctc_fp16.tflite` (~1-2 MB)
- `mediapipe_gru_ctc_fp32.tflite` (~2-4 MB)

## Android Integration

Copy CTC models to Android assets:

```bash
cp exports/ctc/*.tflite app/src/main/assets/ctc/
```

Use in Android:

```kotlin
val interpreter = CTCModelInterpreter(context)
val logits = interpreter.runInference(sequence)
val decoded = ctcDecoder.decode(logits)
```

## Pipeline Flow

```
PyTorch CTC → ONNX → TensorFlow → TFLite (FP16/FP32)
```

- **Dynamic sequences**: Variable-length input with padding
- **FP16 quantization**: Mobile optimization
- **Verification**: Shape validation and dummy inference
- **Error handling**: Multiple fallback strategies

## Troubleshooting

- **Model loading**: Ensure CTC models have correct output shape `[1, 300, 106]`
- **Dependencies**: Install all requirements from `scripts/requirements.txt`
- **Memory issues**: Use `--no-verify` for faster export
- **Export fails**: Check ONNX/TensorFlow compatibility

## Note

This repository focuses on CTC-based continuous sign language recognition. The previous classification model export pipeline has been removed.

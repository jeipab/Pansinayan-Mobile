# Model Export Pipeline Documentation

## Overview

The `export_models.py` script provides a complete pipeline for converting PyTorch classification models to TensorFlow Lite format for Android deployment. It supports both Transformer and GRU models with FP16 quantization for optimal performance.

## Features

- **Complete Conversion Pipeline**: PyTorch → ONNX → TensorFlow → TensorFlow Lite
- **Dual Quantization**: Both FP16 (optimized) and FP32 (fallback) outputs
- **Model Verification**: Cosine similarity comparison between PyTorch and TFLite outputs
- **Robust Error Handling**: Multiple fallback strategies for conversion issues
- **Detailed Logging**: Comprehensive progress tracking and file size reporting
- **Flexible Configuration**: Command-line arguments for customization

## Prerequisites

### Required Files

Place your PyTorch models in the project root directory:

```
Transformer.pt    # Your Transformer model checkpoint
GRU.pt           # Your GRU model checkpoint
```

### Environment Setup

1. **Create virtual environment**:

   ```bash
   python -m venv venv
   ```

2. **Activate virtual environment**:

   ```bash
   # Windows
   venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r scripts/requirements.txt
   ```

## Usage

### Basic Usage

```bash
python scripts/export_models.py
```

This will:

- Look for `Transformer.pt` and `GRU.pt` in the current directory
- Convert both models through the full pipeline
- Save outputs to `exports/` directory
- Perform verification on both FP16 and FP32 versions

### Advanced Usage

#### Custom Model Paths

```bash
python scripts/export_models.py --transformer path/to/your/transformer.pt --gru path/to/your/gru.pt
```

#### Custom Sequence Length

```bash
python scripts/export_models.py --sequence-length 500
```

#### Skip Verification (Faster Export)

```bash
python scripts/export_models.py --no-verify
```

#### Custom Output Directory

```bash
python scripts/export_models.py --output-dir my_exports
```

#### Export Only One Model

```bash
# Export only Transformer
python scripts/export_models.py --gru nonexistent.pt

# Export only GRU
python scripts/export_models.py --transformer nonexistent.pt
```

## Output Structure

The script creates the following directory structure:

```
exports/
├── sign_transformer_fp16.tflite      # FP16 quantized Transformer model
├── sign_transformer_fp32.tflite      # FP32 Transformer model
├── mediapipe_gru_fp16.tflite         # FP16 quantized GRU model
├── mediapipe_gru_fp32.tflite         # FP32 GRU model
├── sign_transformer.onnx             # Intermediate ONNX files
├── mediapipe_gru.onnx
├── sign_transformer_tf/              # Intermediate TensorFlow models
├── mediapipe_gru_tf/
└── export_results.json               # Detailed export results
```

## Model Specifications

### Input Format

- **Shape**: `[Batch, Sequence_Length, 178]`
- **Type**: Float32
- **Description**: MediaPipe keypoints data
- **Default Sequence Length**: 300 frames

### Output Format

- **Shape**: `[Batch, 105]`
- **Type**: Float32 (logits)
- **Description**: Classification logits for 105 gloss classes

## Conversion Pipeline Details

### 1. PyTorch Loading

- Supports multiple checkpoint formats
- Handles both state dict and direct model objects
- Automatic model evaluation mode

### 2. ONNX Export

- **Opset Version**: 17 (latest stable)
- **Dynamic Axes**: Batch size and sequence length
- **Optimizations**: Constant folding enabled

### 3. TensorFlow Conversion

- **Primary Method**: `onnx-tf` backend
- **Fallback Method**: `tf2onnx` (if onnx-tf fails)
- **Format**: SavedModel for maximum compatibility

### 4. TensorFlow Lite Conversion

- **FP16 Quantization**: Optimized for mobile deployment
- **FP32 Fallback**: Full precision for accuracy verification
- **Optimizations**: Default TensorFlow Lite optimizations

## Verification Process

The verification step:

1. Generates random test input with correct shape
2. Runs inference on both PyTorch and TFLite models
3. Calculates cosine similarity between outputs
4. Reports similarity scores (closer to 1.0 is better)

**Expected Results**:

- FP32 similarity: > 0.99 (near-perfect match)
- FP16 similarity: > 0.95 (acceptable quantization loss)

## Troubleshooting

### Common Issues

#### 1. Model Loading Errors

```
Error: Failed to load model Transformer.pt
```

**Solutions**:

- Verify model file exists and is not corrupted
- Check if model was saved with `torch.save(model, path)` vs `torch.save(model.state_dict(), path)`
- Ensure model is compatible with current PyTorch version

#### 2. ONNX Export Failures

```
Error: ONNX export failed: Unsupported operator
```

**Solutions**:

- Update PyTorch to latest version
- Check for custom operations in your model
- Consider simplifying model architecture

#### 3. TensorFlow Conversion Issues

```
Error: TensorFlow conversion failed
```

**Solutions**:

- Install both `onnx-tf` and `tf2onnx` for fallback
- Update TensorFlow to latest version
- Check ONNX model validity with `onnx.checker.check_model()`

#### 4. TFLite Conversion Problems

```
Error: TFLite conversion failed: Unsupported operations
```

**Solutions**:

- Some operations may not be supported in TFLite
- Try FP32 version first to isolate quantization issues
- Consider model architecture modifications

### Dependency Conflicts

If you encounter dependency conflicts:

```bash
# Install packages individually
pip install torch>=2.1.0
pip install onnx>=1.15.0
pip install tensorflow>=2.15.0
pip install onnx-tf>=1.10.0
pip install tf2onnx>=1.15.0
```

### Memory Issues

For large models or limited memory:

```bash
# Skip verification to reduce memory usage
python scripts/export_models.py --no-verify

# Export models one at a time
python scripts/export_models.py --gru nonexistent.pt  # Only Transformer
python scripts/export_models.py --transformer nonexistent.pt  # Only GRU
```

## Performance Expectations

### File Sizes (Approximate)

- **Transformer FP16**: ~1-2 MB
- **Transformer FP32**: ~2-4 MB
- **GRU FP16**: ~500 KB - 1 MB
- **GRU FP32**: ~1-2 MB

### Export Time

- **Per Model**: 2-5 minutes (depending on hardware)
- **Verification**: Additional 30-60 seconds per model
- **Total Pipeline**: 5-10 minutes for both models

## Integration with Android Project

### File Placement

Copy the generated `.tflite` files to your Android project:

```
app/src/main/assets/classification/
├── sign_transformer_fp16.tflite
└── mediapipe_gru_fp16.tflite
```

### Android Integration

The exported models are compatible with:

- TensorFlow Lite Android API
- MediaPipe integration
- Your existing Android inference pipeline

## Advanced Configuration

### Custom Model Architectures

If your models have different architectures, modify the `load_pytorch_model()` function in `export_models.py` to handle your specific checkpoint format.

### Different Input Shapes

Modify the `input_shape` parameter in the script or use the `--sequence-length` argument to match your model's expected input.

### Additional Quantization Options

For more quantization options, modify the `convert_tensorflow_to_tflite()` function to include:

- INT8 quantization
- Dynamic range quantization
- Custom quantization parameters

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Verify all dependencies are correctly installed
3. Ensure model compatibility with the conversion pipeline
4. Check the `export_results.json` file for detailed error information

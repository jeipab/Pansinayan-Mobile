# VAST AI Guide

## Setup Instructions

1. **Clone repo**

   ```bash
   git clone https://github.com/jeipab/Pansinayan-Mobile.git
   ```

2. **Move to that repo**

   ```bash
   cd Pansinayan-Mobile
   ```

3. **Create virtual env**

   ```bash
   python -m venv venv
   ```

4. **Activate venv**

   ```bash
   # On Windows
   venv\Scripts\activate

   # On Linux/Mac
   source venv/bin/activate
   ```

5. **Install requirements**

   ```bash
   pip install -r scripts/requirements.txt
   ```

   **Troubleshooting:**

   If you encounter `tensorflow-lite` installation errors, this is normal - TensorFlow Lite is included with TensorFlow. The requirements.txt has been updated to remove the redundant `tensorflow-lite` package.

   The requirements.txt now uses `tf2onnx` instead of the problematic `onnx-tf` package, which resolves dependency conflicts and is actively maintained.

   For GPU instances, you may need to install TensorFlow with GPU support:

   ```bash
   pip install tensorflow[and-cuda]>=2.8.0
   ```

   If you still have issues, try installing packages individually:

   ```bash
   pip install torch>=1.9.0
   pip install onnx>=1.12.0
   pip install tensorflow>=2.8.0
   pip install numpy>=1.21.0
   pip install pandas>=1.3.0
   pip install tf2onnx>=1.15.0
   ```

   **Alternative approach for stubborn dependency conflicts:**

   ```bash
   # Install without strict version constraints
   pip install torch onnx tensorflow numpy pandas tf2onnx
   ```

6. **Export pt models with the script**

   **Place your PyTorch checkpoints first:**

   ```
   models/checkpoints/transformer/SignTransformerCtc_best.pt
   models/checkpoints/gru/MediaPipeGRUCtc_best.pt
   ```

   **Run the export script:**

   ```bash
   # Export both CTC models (recommended)
   python scripts/model_export_and_setup.py --model both_ctc

   # Or export individual CTC models:
   python scripts/model_export_and_setup.py --model transformer_ctc
   python scripts/model_export_and_setup.py --model mediapipe_gru_ctc

   # Use custom checkpoint path:
   python scripts/model_export_and_setup.py --model both_ctc --checkpoint path/to/model.pt

   # Skip quantization (faster but larger files):
   python scripts/model_export_and_setup.py --model both_ctc --skip-quantization
   ```

   **Generated files will be in `models/converted/` (default output directory):**

   - `sign_transformer_ctc_quant.tflite` (~1-2 MB) ⭐ **USE THIS**
   - `sign_mediapipe_gru_ctc_quant.tflite` (~500 KB)
   - `label_mapping.json`

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

6. **Export pt models with the script**

   **Place your PyTorch models in the project root:**

   ```
   Transformer.pt    # Your Transformer model
   GRU.pt           # Your GRU model
   ```

   **Run the new export script:**

   ```bash
   # Export both models with full pipeline (PyTorch → ONNX → TF → TFLite)
   python scripts/export_models.py

   # Custom sequence length
   python scripts/export_models.py --sequence-length 500

   # Skip verification for faster export
   python scripts/export_models.py --no-verify

   # Custom output directory
   python scripts/export_models.py --output-dir my_exports
   ```

   **Generated files will be in `exports/` directory:**

   - `sign_transformer_fp16.tflite` (~1-2 MB) ⭐ **USE THIS**
   - `sign_transformer_fp32.tflite` (~2-4 MB) (fallback)
   - `mediapipe_gru_fp16.tflite` (~500 KB - 1 MB) ⭐ **USE THIS**
   - `mediapipe_gru_fp32.tflite` (~1-2 MB) (fallback)
   - `export_results.json` (detailed results)

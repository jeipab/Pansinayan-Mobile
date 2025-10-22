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

   **Place your PyTorch checkpoints first:**

   ```
   models/checkpoints/transformer/best_model.pt
   models/checkpoints/gru/best_model.pt
   ```

   **Run the export script:**

   ```bash
   # Export both models (recommended)
   python scripts/model_export_and_setup.py --model both

   # Or export individual models:
   python scripts/model_export_and_setup.py --model transformer
   python scripts/model_export_and_setup.py --model mediapipe_gru

   # Use custom checkpoint path:
   python scripts/model_export_and_setup.py --model both --checkpoint path/to/model.pt

   # Skip quantization (faster but larger files):
   python scripts/model_export_and_setup.py --model both --skip-quantization
   ```

   **Generated files will be in `models/converted/`:**

   - `sign_transformer_quant.tflite` (~1-2 MB) ⭐ **USE THIS**
   - `sign_mediapipe_gru_quant.tflite` (~500 KB)
   - `label_mapping.json`

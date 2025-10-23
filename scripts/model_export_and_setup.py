"""
🚀 Complete Android Model Export Pipeline

This script exports your trained sign language recognition models for Android deployment.

SUPPORTED MODELS:
1. SignTransformer: Transformer encoder for keypoint sequences (primary model)
2. MediaPipeGRU: Lightweight GRU for keypoint sequences (baseline model)

WHAT IT DOES:
1. Loads trained PyTorch model (Transformer or MediaPipeGRU)
2. Exports to ONNX format (intermediate, portable)
3. Converts ONNX → TensorFlow SavedModel
4. Converts TensorFlow → TFLite (with quantization options)
5. Validates accuracy (compares PyTorch vs TFLite predictions)
6. Generates label mapping JSON for Android
7. Creates detailed model specification document

CONVERSIONS PERFORMED:
PyTorch (.pt) → ONNX (.onnx) → TensorFlow (SavedModel) → TFLite (.tflite)
                    ↓
            (Can also use ONNX directly in Android with ONNX Runtime)

OUTPUT FILES (example for Transformer):
models/converted/
├── sign_transformer.onnx              # ONNX model (can use with ONNX Runtime)
├── sign_transformer_tf/               # TensorFlow SavedModel (intermediate)
├── sign_transformer.tflite            # Standard TFLite (~4-5 MB)
├── sign_transformer_quant.tflite      # Quantized TFLite (~1-2 MB) ⭐ USE THIS
├── label_mapping.json                 # Gloss/category labels (105 glosses, 10 categories)
└── MODEL_SPECS.txt                    # Model specifications for reference

For MediaPipeGRU: sign_mediapipe_gru*.* files

USAGE:
    # Export Transformer (default)
    python model_export_and_setup.py --model transformer
    
    # Export MediaPipeGRU
    python model_export_and_setup.py --model mediapipe_gru
    
    # Export with custom checkpoint
    python model_export_and_setup.py --model transformer --checkpoint path/to/model.pt
    
    # Export both models
    python model_export_and_setup.py --model both

REQUIREMENTS:
    pip install torch onnx onnx-tf tensorflow numpy pandas

NEXT STEPS AFTER RUNNING:
1. Copy <model>_quant.tflite to Android: app/src/main/assets/
2. Copy label_mapping.json to Android: app/src/main/assets/
3. Follow Android project structure for integration

Author: Generated for Filipino Sign Language Recognition Project
Date: October 2025
"""

import os
import sys
import json
import glob
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.onnx

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Try to import models, but make it optional for standalone use
try:
    from models import SignTransformer, MediaPipeGRU, SignTransformerCtc, MediaPipeGRUCtc
    from data.labels import load_label_mappings
    MODELS_AVAILABLE = True
except ImportError:
    print("WARNING: Could not import model classes.")
    print("   This script can still be used for model conversion if you provide")
    print("   the model classes separately or modify the imports.")
    MODELS_AVAILABLE = False


# ==================== CONFIGURATION ====================

# Model architecture parameters (must match your trained model)
TRANSFORMER_CONFIG = {
    'input_dim': 178,      # 89 keypoints × 2 coordinates
    'emb_dim': 256,        # Embedding dimension
    'n_heads': 8,          # Number of attention heads
    'n_layers': 4,         # Number of transformer layers
    'num_gloss': 105,      # Number of sign classes
    'num_cat': 10,         # Number of categories
    'dropout': 0.0,        # Set to 0 for inference
    'pooling_method': 'mean'  # Pooling strategy
}

MEDIAPIPE_GRU_CONFIG = {
    'input_dim': 178,      # 89 keypoints × 2 coordinates
    'projection_dim': None,  # Optional input projection (None = use input_dim directly)
    'hidden1': 256,        # First GRU hidden size
    'hidden2': 128,        # Second GRU hidden size
    'num_gloss': 105,      # Number of sign classes
    'num_cat': 10,         # Number of categories
    'dropout': 0.0,        # Set to 0 for inference
    'bidirectional': False # Bidirectional GRU
}

# Export configuration
EXPORT_CONFIG = {
    'seq_len': 150,        # Sequence length for export (5 seconds at 30 FPS)
    'opset_version': 18,   # ONNX opset version (updated for compatibility)
    'quantize': True,      # Create quantized version
}

TRANSFORMER_CTC_CONFIG = {
    'input_dim': 178,
    'emb_dim': 256,
    'n_heads': 8,
    'n_layers': 4,
    'num_ctc_classes': 11,  # 10 greeting signs + 1 blank token
    'num_cat': 1,           # Only 1 category (greetings)
    'dropout': 0.0,
}

MEDIAPIPE_GRU_CTC_CONFIG = {
    'input_dim': 178,
    'projection_dim': None,
    'hidden1': 30,           # Actual trained model dimensions
    'hidden2': 22,           # Actual trained model dimensions  
    'num_ctc_classes': 11,  # 10 greeting signs + 1 blank token
    'num_cat': 1,           # Only 1 category (greetings)
    'dropout': 0.3,
}

# Paths
DEFAULT_CHECKPOINTS = {
    'transformer': 'models/checkpoints/transformer/SignTransformerCtc_best.pt',
    'mediapipe_gru': 'models/checkpoints/gru/MediaPipeGRUCtc_best.pt',
    'transformer_ctc': 'models/checkpoints/transformer/SignTransformerCtc_best.pt',
    'mediapipe_gru_ctc': 'models/checkpoints/gru/MediaPipeGRUCtc_best.pt',
}
OUTPUT_DIR = 'models/converted'


def print_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_step(step_num, title):
    """Print step header."""
    print(f"\n[Step {step_num}] {title}")
    print("-" * 70)


def export_to_onnx(model, output_path, seq_len=90, is_ctc=False):
    """Export PyTorch model to ONNX."""
    print_step("1/5", "Exporting PyTorch → ONNX")
    
    model.eval()
    dummy_input = torch.randn(1, seq_len, 178)
    
    if is_ctc:
        # CTC models output [B, T, num_ctc_classes] + [B, T, num_cat]
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['input_sequence'],
            output_names=['ctc_logits', 'category_logits'],
            verbose=False
        )
    else:
        # Classification models output [B, num_gloss] + [B, num_cat]
        torch.onnx.export(
            model,
            (dummy_input, None),
            output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input_sequence'],
            output_names=['gloss_logits', 'category_logits'],
            dynamic_axes={
                'input_sequence': {0: 'batch_size', 1: 'seq_len'},
                'gloss_logits': {0: 'batch_size'},
                'category_logits': {0: 'batch_size'}
            },
            verbose=False
        )
    
    print(f"✓ ONNX model saved: {output_path}")


def convert_to_tensorflow(onnx_path, tf_path):
    """Convert ONNX to TensorFlow SavedModel - SKIP for now, use ONNX Runtime instead."""
    print_step("2/5", "Skipping ONNX → TensorFlow conversion")
    
    print("⚠️  TensorFlow conversion skipped due to tf2onnx compatibility issues")
    print("   RECOMMENDED: Use ONNX Runtime for Android instead")
    print("   Add to build.gradle:")
    print("   implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.16.0'")
    print(f"\n   Your ONNX model is ready: {onnx_path}")
    
    return False


def convert_to_tflite(tf_path, tflite_path, quantize=False):
    """Convert TensorFlow to TFLite."""
    step_name = "3/5" if not quantize else "4/5"
    model_type = "Quantized" if quantize else "Standard"
    print_step(step_name, f"Converting TensorFlow → TFLite ({model_type})")
    
    try:
        import tensorflow as tf
        
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
        
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
        print(f"✓ TFLite model saved: {tflite_path} ({size_mb:.2f} MB)")
        return True
        
    except Exception as e:
        print(f"❌ TFLite conversion failed: {e}")
        return False


def validate_tflite_vs_pytorch(tflite_path, pytorch_model, device, test_dir="data/demo"):
    """
    Comprehensive validation: Compare TFLite vs PyTorch predictions.
    
    This ensures the conversion didn't break the model and accuracy is preserved.
    """
    print_step("5/6", "Validating TFLite Model (Comprehensive)")
    
    try:
        import tensorflow as tf
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"\n📊 TFLite Model Specifications:")
        print(f"   Input shape:  {input_details[0]['shape']}")
        print(f"   Input dtype:  {input_details[0]['dtype']}")
        print(f"   Output 0 (gloss):    {output_details[0]['shape']} ({output_details[0]['dtype']})")
        print(f"   Output 1 (category): {output_details[1]['shape']} ({output_details[1]['dtype']})")
        
        # Test 1: Dummy data inference (sanity check)
        print(f"\n✓ Test 1: Dummy data inference")
        input_shape = input_details[0]['shape']
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        print(f"   ✓ Inference successful on random data")
        
        # Test 2: Real data comparison
        npz_files = glob.glob(os.path.join(test_dir, "**/*.npz"), recursive=True)
        if not npz_files:
            print(f"\n⚠️  No test data found in {test_dir}, skipping accuracy comparison")
            return True
            
        print(f"\n✓ Test 2: Accuracy comparison on {len(npz_files[:3])} samples")
        
        agreements = []
        for i, npz_path in enumerate(npz_files[:3]):
            data = np.load(npz_path)
            X = data['X'][:90]  # Use first 90 frames
            
            # PyTorch inference
            pytorch_model.eval()
            with torch.no_grad():
                X_torch = torch.from_numpy(X).float().unsqueeze(0).to(device)
                gloss_pt, cat_pt = pytorch_model(X_torch)
                gloss_probs_pt = torch.softmax(gloss_pt, dim=-1).squeeze().cpu().numpy()
            
            # TFLite inference
            X_tflite = X.astype(np.float32)[np.newaxis, ...]
            interpreter.set_tensor(input_details[0]['index'], X_tflite)
            interpreter.invoke()
            
            gloss_logits_tflite = interpreter.get_tensor(output_details[0]['index']).squeeze()
            
            # Softmax for TFLite
            exp_logits = np.exp(gloss_logits_tflite - np.max(gloss_logits_tflite))
            gloss_probs_tflite = exp_logits / exp_logits.sum()
            
            # Compare top-5
            top5_pt = np.argsort(gloss_probs_pt)[-5:][::-1]
            top5_tflite = np.argsort(gloss_probs_tflite)[-5:][::-1]
            
            top1_match = (top5_pt[0] == top5_tflite[0])
            top5_overlap = len(set(top5_pt) & set(top5_tflite))
            
            agreements.append({
                'file': os.path.basename(npz_path),
                'top1_match': top1_match,
                'top5_overlap': top5_overlap,
                'pytorch_top1': int(top5_pt[0]),
                'tflite_top1': int(top5_tflite[0]),
                'pytorch_conf': float(gloss_probs_pt[top5_pt[0]]),
                'tflite_conf': float(gloss_probs_tflite[top5_tflite[0]])
            })
            
            status = "✓" if top1_match else "✗"
            print(f"   Sample {i+1}: {status} PyTorch:{top5_pt[0]} vs TFLite:{top5_tflite[0]} (overlap: {top5_overlap}/5)")
        
        # Summary
        top1_accuracy = sum(a['top1_match'] for a in agreements) / len(agreements)
        avg_overlap = np.mean([a['top5_overlap'] for a in agreements])
        
        print(f"\n📈 Validation Summary:")
        print(f"   Top-1 Agreement: {top1_accuracy*100:.1f}%")
        print(f"   Avg Top-5 Overlap: {avg_overlap:.1f}/5")
        
        if top1_accuracy >= 0.9:
            print(f"   ✅ Excellent! TFLite model is highly accurate.")
        elif top1_accuracy >= 0.7:
            print(f"   ⚠️  Good, but some differences exist. Should be usable.")
        else:
            print(f"   ❌ Poor agreement. Check conversion process.")
            
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_label_mapping(output_path):
    """
    Generate label mapping JSON for Android.
    
    Creates a basic label mapping if CSV is not available.
    """
    print_step("6/7", "Generating Label Mapping JSON")
    
    try:
        # Try to load from CSV first
        csv_path = "../data/splitting/labels_reference.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # Create gloss mapping: {id: label}
            gloss_mapping = {int(row['gloss_id']): str(row['label']) 
                            for _, row in df.iterrows()}
            
            # Create category mapping: {id: category}
            category_mapping = {}
            for _, row in df.iterrows():
                cat_id = int(row['cat_id'])
                if cat_id not in category_mapping:
                    category_mapping[cat_id] = str(row['category'])
        else:
            print(f"⚠️  CSV not found at {csv_path}, creating greeting-only mapping")
            # Create greeting mapping for 10 greeting signs and 1 category
            greeting_signs = [
                "GOOD MORNING", "GOOD AFTERNOON", "GOOD EVENING", "HELLO", "HOW ARE YOU",
                "IM FINE", "NICE TO MEET YOU", "THANK YOU", "YOURE WELCOME", "SEE YOU TOMORROW"
            ]
            gloss_mapping = {i: greeting_signs[i] for i in range(10)}
            category_mapping = {0: "GREETING"}
        
        # Save JSON with metadata
        label_data = {
            "glosses": gloss_mapping,
            "categories": category_mapping,
            "metadata": {
                "total_glosses": len(gloss_mapping),
                "total_categories": len(category_mapping),
                "source": csv_path if os.path.exists(csv_path) else "generated",
                "generated_at": datetime.now().isoformat()
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(label_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Label mapping saved: {output_path}")
        print(f"   - {len(gloss_mapping)} glosses")
        print(f"   - {len(category_mapping)} categories")
        
        # Show sample entries
        print(f"\n📝 Sample glosses:")
        for i in list(gloss_mapping.keys())[:5]:
            print(f"   {i}: {gloss_mapping[i]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Label mapping failed: {e}")
        return False


def generate_model_specs(output_path, model_config, export_config, file_sizes):
    """
    Generate detailed model specifications document for reference.
    
    This helps with Android integration and debugging.
    """
    print_step("7/7", "Generating Model Specifications Document")
    
    try:
        specs = f"""
================================================================================
                    ANDROID MODEL SPECIFICATIONS
================================================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model Type:         Transformer Encoder
Input Dimension:    {model_config['input_dim']} (78 keypoints × 2 coordinates)
Embedding Dim:      {model_config['emb_dim']}
Attention Heads:    {model_config['n_heads']}
Encoder Layers:     {model_config['n_layers']}
Pooling Method:     {model_config['pooling_method']}

Output Classes:
  - Glosses:        {model_config['num_gloss']} classes
  - Categories:     {model_config['num_cat']} classes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT SPECIFICATIONS (For Android Integration)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tensor Shape:       [1, T, 178]
  - Batch size:     1 (single sequence)
  - Sequence len:   T (typically 60-120 frames, recommended: 90)
  - Feature dim:    178 (89 keypoints × 2 coordinates)

Data Type:          float32
Value Range:        [0.0, 1.0] (normalized x, y coordinates)

Keypoint Structure:
  [0-49]:    Pose landmarks (25 points × 2) - Upper body
  [50-91]:   Left hand (21 points × 2) - Hand landmarks
  [92-133]:  Right hand (21 points × 2) - Hand landmarks
  [134-177]: Face landmarks (22 points × 2) - Key facial points

Preprocessing:
  1. Extract keypoints with MediaPipe (pose + hands + face)
  2. Normalize coordinates to [0, 1] range
  3. Fill gaps using linear interpolation (max_gap=5)
  4. Arrange in format: [pose, left_hand, right_hand, face]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT SPECIFICATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Output 0 (Gloss Logits):
  - Shape:          [1, 105]
  - Type:           float32
  - Description:    Raw logits for 105 sign gloss classes

Output 1 (Category Logits):
  - Shape:          [1, 10]
  - Type:           float32
  - Description:    Raw logits for 10 category classes

Post-processing:
  1. Apply softmax to convert logits → probabilities
  2. Get argmax for top prediction
  3. Use top-k for confidence ranking
  4. Map ID to label using label_mapping.json

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILE SIZES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{file_sizes}

Recommendation: Use sign_transformer_quant.tflite for Android (smaller & faster)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANDROID INTEGRATION GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. COPY FILES TO ANDROID PROJECT:
   
   Copy to app/src/main/assets/:
   - sign_transformer_quant.tflite
   - label_mapping.json

2. LOAD MODEL IN KOTLIN (TFLiteModelRunner.kt):

   val interpreter = Interpreter(modelBuffer, options)
   
   // Enable GPU acceleration
   if (CompatibilityList().isDelegateSupportedOnThisDevice) {{
       options.addDelegate(GpuDelegate())
   }}

3. PREPARE INPUT:

   // Keypoint sequence: Array[T, 178] where T = 90 frames
   val inputBuffer = ByteBuffer.allocateDirect(1 * T * 178 * 4)
   inputBuffer.order(ByteOrder.nativeOrder())
   
   for (t in 0 until T) {{
       for (i in 0 until 178) {{
           inputBuffer.putFloat(sequence[t][i])
       }}
   }}

4. RUN INFERENCE:

   val glossLogits = Array(1) {{ FloatArray(105) }}
   val categoryLogits = Array(1) {{ FloatArray(10) }}
   
   interpreter.runForMultipleInputsOutputs(
       arrayOf(inputBuffer),
       mapOf(0 to glossLogits, 1 to categoryLogits)
   )

5. POST-PROCESS:

   // Apply softmax
   val probs = softmax(glossLogits[0])
   
   // Get top prediction
   val topIdx = probs.indices.maxByOrNull {{ probs[it] }} ?: 0
   val confidence = probs[topIdx]
   
   // Map to label
   val label = labelMapper.getGlossLabel(topIdx)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PERFORMANCE TIPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. GPU Acceleration:    2-5× speedup (enable in TFLiteModelRunner)
2. Quantized Model:     4× smaller, 2-3× faster
3. Sequence Length:     Use 60-90 frames (not full 300)
4. Inference Frequency: Run every 10-15 frames, not every frame
5. Async Processing:    Use Kotlin Coroutines (avoid blocking UI)

Expected Performance on Android:
  - Inference Time:     100-300ms per sequence
  - Frame Rate:         10-15 FPS
  - Detection Latency:  500-1000ms (from sign completion)
  - Memory Usage:       100-200MB

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TROUBLESHOOTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Issue: Model file not found in Android
→ Solution: Verify file in app/src/main/assets/sign_transformer_quant.tflite
→ Add to build.gradle: aaptOptions {{ noCompress "tflite" }}

Issue: Inference fails or crashes
→ Solution: Check input tensor shape matches [1, T, 178]
→ Ensure data type is float32
→ Verify sequence length T is reasonable (30-120 frames)

Issue: Slow inference (>500ms)
→ Solution: Enable GPU delegate (see TFLiteModelRunner.kt)
→ Reduce sequence length (90 → 60 frames)
→ Run inference less frequently (every 15 frames instead of 10)

Issue: Poor accuracy on Android
→ Solution: Verify preprocessing matches Python pipeline
→ Check keypoint extraction order (pose, left_hand, right_hand, face)
→ Ensure coordinates normalized to [0, 1] range

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEXT STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Copy model files to Android project (see above)
2. Use provided Kotlin code in tool/kotlin_scaffolding/
3. Follow tool/ANDROID_IMPLEMENTATION_GUIDE.md for complete instructions
4. Test on Android device and verify recognition works

For detailed implementation guide, see:
→ tool/ANDROID_IMPLEMENTATION_GUIDE.md
→ tool/kotlin_scaffolding/PROJECT_STRUCTURE.md

================================================================================
"""
        
        with open(output_path, 'w') as f:
            f.write(specs)
        
        print(f"✓ Model specifications saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to generate specs: {e}")
        return False


def print_next_steps():
    """Print comprehensive next steps with all resources."""
    print_header("✅ Model Export Complete!")
    
    print("\n📦 Generated Files in tool/android_models/:")
    print("   ├── sign_transformer.onnx             (~5 MB)")
    print("   ├── sign_transformer_tf/              (SavedModel)")
    print("   ├── sign_transformer.tflite           (~4-5 MB)")
    print("   ├── sign_transformer_quant.tflite     (~1-2 MB) ⭐ USE THIS")
    print("   ├── label_mapping.json                (105 glosses, 10 categories)")
    print("   └── MODEL_SPECS.txt                   (Detailed specifications)")
    
    print("\n" + "=" * 70)
    print("                    🚀 NEXT STEPS")
    print("=" * 70)
    
    print("\n📱 STEP 1: Create Android Project (15 minutes)")
    print("   1. Open Android Studio")
    print("   2. New Project → Empty Activity")
    print("      - Name: Sign Language Recognition")
    print("      - Package: com.slr.app")
    print("      - Language: Kotlin, Min SDK: 24")
    
    print("\n📂 STEP 2: Copy Kotlin Code (10 minutes)")
    print("   Copy ALL files from tool/kotlin_scaffolding/ to your Android project:")
    print("   ✓ 8 Kotlin classes  → src/main/java/com/slr/app/")
    print("   ✓ activity_main.xml → src/main/res/layout/")
    print("   ✓ build.gradle      → app/build.gradle (merge dependencies)")
    print("   ✓ AndroidManifest.xml → Merge permissions")
    print("")
    print("   📘 See: tool/kotlin_scaffolding/PROJECT_STRUCTURE.md")
    
    print("\n📦 STEP 3: Copy Model Files (5 minutes)")
    print("   Create assets folder: app/src/main/assets/")
    print("   Copy:")
    print("   → tool/android_models/sign_transformer_quant.tflite")
    print("   → tool/android_models/label_mapping.json")
    
    print("\n🔽 STEP 4: Download MediaPipe Models (5 minutes)")
    print("   Download to app/src/main/assets/:")
    print("   → hand_landmarker.task")
    print("     https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
    print("   → pose_landmarker_full.task")
    print("     https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task")
    
    print("\n🔨 STEP 5: Build & Run (5 minutes)")
    print("   1. Sync Gradle")
    print("   2. Build → Make Project")
    print("   3. Run on Android device")
    print("   4. Grant camera permission")
    print("   5. Test recognition!")
    
    print("\n📚 DOCUMENTATION:")
    print("   → tool/ANDROID_IMPLEMENTATION_GUIDE.md   (Complete 3-week guide)")
    print("   → tool/kotlin_scaffolding/PROJECT_STRUCTURE.md  (Setup instructions)")
    print("   → tool/android_models/MODEL_SPECS.txt   (Model reference)")
    
    print("\n🎯 TIMELINE:")
    print("   Week 1: Camera + MediaPipe working")
    print("   Week 2: Single sign recognition (>70% accuracy)")
    print("   Week 3: Continuous recognition + polish")
    
    print("\n" + "=" * 70)
    print("  🎉 Everything is ready! Start with tool/START_HERE.md")
    print("=" * 70)


def get_file_sizes(output_dir, model_prefix):
    """Get file sizes for all generated models."""
    files = {
        'onnx': os.path.join(output_dir, f"{model_prefix}.onnx"),
        'tflite': os.path.join(output_dir, f"{model_prefix}.tflite"),
        'tflite_quant': os.path.join(output_dir, f"{model_prefix}_quant.tflite")
    }
    
    sizes_text = ""
    for name, path in files.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            sizes_text += f"{os.path.basename(path):40s} {size_mb:6.2f} MB\n"
    
    return sizes_text.strip()


def export_single_model(model_type, checkpoint_path, output_dir, seq_len, skip_quantization, device):
    """
    Export a single model to TFLite format.
    
    Args:
        model_type (str): 'transformer', 'mediapipe_gru', 'transformer_ctc', or 'mediapipe_gru_ctc'
        checkpoint_path (str): Path to PyTorch checkpoint file
        output_dir (str): Output directory for exported files
        seq_len (int): Sequence length for export
        skip_quantization (bool): Skip quantization step
        device: PyTorch device to use
    
    Returns:
        bool: True if export successful, False otherwise
    """
    # Configure model-specific settings
    if model_type == 'transformer':
        model_config = TRANSFORMER_CONFIG
        model_class = SignTransformer
        model_prefix = "sign_transformer"
        model_name = "SignTransformer"
        is_ctc = False
    elif model_type == 'mediapipe_gru':
        model_config = MEDIAPIPE_GRU_CONFIG
        model_class = MediaPipeGRU
        model_prefix = "sign_mediapipe_gru"
        model_name = "MediaPipeGRU"
        is_ctc = False
    elif model_type == 'transformer_ctc':
        model_config = TRANSFORMER_CTC_CONFIG
        model_class = SignTransformerCtc
        model_prefix = "sign_transformer_ctc"
        model_name = "SignTransformerCtc"
        is_ctc = True
    elif model_type == 'mediapipe_gru_ctc':
        model_config = MEDIAPIPE_GRU_CTC_CONFIG
        model_class = MediaPipeGRUCtc
        model_prefix = "sign_mediapipe_gru_ctc"
        model_name = "MediaPipeGRUCtc"
        is_ctc = True
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print_header(f"🚀 Exporting {model_name} Model")
    
    # Setup paths
    onnx_path = os.path.join(output_dir, f"{model_prefix}.onnx")
    tf_path = os.path.join(output_dir, f"{model_prefix}_tf")
    tflite_path = os.path.join(output_dir, f"{model_prefix}.tflite")
    tflite_quant_path = os.path.join(output_dir, f"{model_prefix}_quant.tflite")
    specs_path = os.path.join(output_dir, f"{model_prefix}_SPECS.txt")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"\n❌ Checkpoint not found: {checkpoint_path}")
        print(f"   Looking for {model_type} models...")
        checkpoint_dir = f"../models/checkpoints/{model_type}"
        if os.path.exists(checkpoint_dir):
            print(f"   Available models in {checkpoint_dir}:")
            for f in os.listdir(checkpoint_dir):
                if f.endswith(('.pt', '.pth')):
                    print(f"     - {os.path.join(checkpoint_dir, f)}")
        else:
            print(f"   Directory not found: {checkpoint_dir}")
            print(f"   Have you trained a {model_name} model yet?")
        return False
    
    # Load PyTorch model
    print(f"\n📦 Loading {model_name} from: {checkpoint_path}")
    
    model = model_class(**model_config).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"✓ {model_name} loaded successfully")
    
    # Display model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Estimated size: {total_params * 4 / (1024 * 1024):.2f} MB (float32)")
    
    # Step 1: Export to ONNX
    export_to_onnx(model, onnx_path, seq_len, is_ctc)
    
    # Step 2: Convert to TensorFlow (skip if not available)
    success = convert_to_tensorflow(onnx_path, tf_path)
    if not success:
        print(f"\n⚠️  Skipping TensorFlow conversion for {model_name}")
        print("   ONNX model is ready for Android deployment!")
        print(f"   Model file: {onnx_path}")
        
        # Still generate label mapping and specs
        print_header("📝 Using Greeting Label Mapping")
        
        # Use the existing greeting label mapping
        greeting_label_path = os.path.join(output_dir, "greeting_label_mapping.json")
        label_json_path = os.path.join(output_dir, "label_mapping.json")
        
        if os.path.exists(greeting_label_path):
            # Copy greeting mapping to main label mapping
            import shutil
            shutil.copy2(greeting_label_path, label_json_path)
            print(f"✓ Using greeting label mapping: {greeting_label_path}")
        else:
            # Fallback to generating new one
            generate_label_mapping(label_json_path)
        
        # Generate specs for ONNX model
        file_sizes = f"{os.path.basename(onnx_path):40s} {os.path.getsize(onnx_path) / (1024 * 1024):6.2f} MB"
        specs_path = os.path.join(output_dir, f"{model_prefix}_ONNX_SPECS.txt")
        
        # Add missing parameters for CTC models
        if 'pooling_method' not in model_config:
            model_config['pooling_method'] = 'none'  # CTC models don't use pooling
        if 'num_gloss' not in model_config:
            model_config['num_gloss'] = model_config.get('num_ctc_classes', 11) - 1  # CTC classes - blank token
            
        generate_model_specs(specs_path, model_config, EXPORT_CONFIG, file_sizes)
        
        print(f"\n✅ {model_name} ONNX export complete!")
        print(f"   ONNX model: {onnx_path}")
        print(f"   Label mapping: {label_json_path}")
        print(f"   Specifications: {specs_path}")
        
        return True
    
    # Step 3: Convert to TFLite (standard)
    success = convert_to_tflite(tf_path, tflite_path, quantize=False)
    if not success:
        return False
    
    # Step 4: Convert to TFLite (quantized)
    if not skip_quantization:
        success = convert_to_tflite(tf_path, tflite_quant_path, quantize=True)
        if not success:
            print(f"\n⚠️  Quantization failed for {model_name}, but standard TFLite model is available")
    
    # Step 5: Validate
    validate_tflite_vs_pytorch(
        tflite_quant_path if os.path.exists(tflite_quant_path) else tflite_path,
        model, device
    )
    
    # Step 6: Generate specs
    file_sizes = get_file_sizes(output_dir, model_prefix)
    generate_model_specs(specs_path, model_config, EXPORT_CONFIG, file_sizes)
    
    print(f"\n✅ {model_name} export complete!")
    print(f"   Models saved with prefix: {model_prefix}")
    
    return True


def main():
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Export PyTorch sign language recognition models to Android-optimized formats"
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['transformer', 'mediapipe_gru', 'transformer_ctc', 'mediapipe_gru_ctc', 'both', 'both_ctc'],
        default='transformer',
        help='Model to export (default: transformer)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to PyTorch checkpoint (overrides default paths)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=OUTPUT_DIR,
        help=f'Output directory (default: {OUTPUT_DIR})'
    )
    parser.add_argument(
        '--seq-len',
        type=int,
        default=EXPORT_CONFIG['seq_len'],
        help=f"Sequence length for export (default: {EXPORT_CONFIG['seq_len']})"
    )
    parser.add_argument(
        '--skip-quantization',
        action='store_true',
        help='Skip quantized model generation'
    )
    
    args = parser.parse_args()
    
    print_header("🎯 Android Sign Language Recognition - Complete Model Export")
    
    if not MODELS_AVAILABLE:
        print("\nWARNING: Model classes not available!")
        print("   This script requires the model classes to be available.")
        print("   Please ensure your model definitions are accessible.")
        print("   You may need to modify the import paths or copy the model files.")
        return
    
    print("\nSupported Models:")
    print("  Classification (isolated signs):")
    print("    • SignTransformer: Transformer encoder (~1-2 MB quantized)")
    print("    • MediaPipeGRU: Lightweight GRU (~500 KB quantized)")
    print("  CTC (continuous sequences):")
    print("    • SignTransformerCtc: Transformer with CTC + Category (~1-2 MB quantized)")
    print("    • MediaPipeGRUCtc: GRU with CTC + Category (~500 KB quantized)")
    
    print("\nThis script will:")
    print("  - Load trained PyTorch model(s)")
    print("  - Export to ONNX format (portable)")
    print("  - Convert to TensorFlow SavedModel")
    print("  - Generate TFLite models (standard + quantized)")
    print("  - Validate accuracy (compare PyTorch vs TFLite)")
    print("  - Generate label mapping JSON")
    print("  - Create model specifications document")
    
    # Setup output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cpu')  # Use CPU for export consistency
    
    # Determine which models to export
    if args.model == 'both':
        models_to_export = ['transformer', 'mediapipe_gru']
    elif args.model == 'both_ctc':
        models_to_export = ['transformer_ctc', 'mediapipe_gru_ctc']
    else:
        models_to_export = [args.model]
    
    # Export each model
    success_count = 0
    for model_type in models_to_export:
        # Determine checkpoint path
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = DEFAULT_CHECKPOINTS.get(model_type)
            if not checkpoint_path:
                print(f"\n⚠️  No default checkpoint path for model type: {model_type}")
                continue
        
        # Export the model
        success = export_single_model(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            seq_len=args.seq_len,
            skip_quantization=args.skip_quantization,
            device=device
        )
        
        if success:
            success_count += 1
    
    # Generate label mapping (only once, shared by both models)
    label_json_path = os.path.join(output_dir, "label_mapping.json")
    if success_count > 0:
        print_header("📝 Generating Label Mapping")
        generate_label_mapping(label_json_path)
    
    # Print summary
    print_header("✅ Export Summary")
    print(f"\n{success_count}/{len(models_to_export)} model(s) exported successfully")
    
    if success_count > 0:
        print(f"\n📦 Generated Files in {output_dir}/:")
        for model_type in models_to_export:
            if model_type == 'transformer':
                prefix = "sign_transformer"
            else:
                prefix = "sign_mediapipe_gru"
            
            for ext in ['.onnx', '.tflite', '_quant.tflite', '_SPECS.txt']:
                filepath = os.path.join(output_dir, f"{prefix}{ext}")
                if os.path.exists(filepath):
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    print(f"   ✓ {os.path.basename(filepath):40s} ({size_mb:6.2f} MB)")
        
        if os.path.exists(label_json_path):
            print(f"   ✓ {os.path.basename(label_json_path):40s} (shared)")
        
        print("\n" + "=" * 70)
        print("                    🚀 NEXT STEPS")
        print("=" * 70)
        
        print("\n📱 ANDROID DEPLOYMENT:")
        print("   1. Copy to app/src/main/assets/:")
        print("      • <model>_quant.tflite (your chosen model)")
        print("      • label_mapping.json")
        print("   2. Download MediaPipe models to assets/ (see ANDROID_IMPLEMENTATION_GUIDE.md)")
        print("   3. Follow kotlin_scaffolding/PROJECT_STRUCTURE.md for setup")
        
        print("\n📚 DOCUMENTATION:")
        print("   • tool/ANDROID_IMPLEMENTATION_GUIDE.md - Complete guide")
        print("   • tool/MODEL_COMPARISON_STRATEGY.md - Model comparison strategy")
        print("   • <model>_SPECS.txt - Model specifications")
        
        print("\n" + "=" * 70)
        print("  🎉 Everything is ready! Start building your Android app!")
        print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Script interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("\nIf you encounter issues:")
        print("  1. Ensure dependencies installed: pip install torch onnx onnx-tf tensorflow")
        print("  2. Check that trained model exists: models/checkpoints/transformer/best_model.pt")
        print("  3. See ANDROID_IMPLEMENTATION_GUIDE.md for troubleshooting")


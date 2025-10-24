#!/usr/bin/env python3
"""
Model Export Pipeline: PyTorch → ONNX → TensorFlow → TensorFlow Lite

This script converts PyTorch classification models to TFLite format with FP16 quantization
for Android deployment. Supports both Transformer and GRU models.

Usage:
    python export_models.py

Requirements:
    - Transformer.pt and GRU.pt models in the current directory
    - All dependencies from requirements.txt installed

Output:
    - exports/sign_transformer_fp16.tflite
    - exports/sign_transformer_fp32.tflite  
    - exports/mediapipe_gru_fp16.tflite
    - exports/mediapipe_gru_fp32.tflite
"""

import os
import sys
import torch
import onnx
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Optional
import argparse
import json
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def setup_directories() -> Path:
    """Create exports directory if it doesn't exist."""
    exports_dir = Path("exports")
    exports_dir.mkdir(exist_ok=True)
    return exports_dir

def load_pytorch_model(model_path: str) -> torch.nn.Module:
    """
    Load PyTorch model from checkpoint.
    
    Args:
        model_path: Path to the .pt file
        
    Returns:
        Loaded PyTorch model
    """
    print(f"Loading PyTorch model from {model_path}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # Load the model checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                model_state_dict = checkpoint['state_dict']
            else:
                # Assume the dict itself is the state dict
                model_state_dict = checkpoint
        else:
            # Direct model object
            return checkpoint
        
        # Create a dummy model structure (you may need to adjust this based on your actual model)
        # This is a generic approach - you might need to modify based on your specific model architecture
        model = torch.nn.Module()
        model.load_state_dict(model_state_dict)
        model.eval()
        
        return model
        
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        print("Attempting to load as direct model object...")
        try:
            model = torch.load(model_path, map_location='cpu')
            if hasattr(model, 'eval'):
                model.eval()
            return model
        except Exception as e2:
            raise RuntimeError(f"Failed to load model {model_path}: {e2}")

def export_to_onnx(model: torch.nn.Module, model_name: str, input_shape: Tuple[int, int, int], 
                   exports_dir: Path) -> str:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        model_name: Name for the exported file
        input_shape: Input tensor shape (B, T, 178)
        exports_dir: Directory to save exports
        
    Returns:
        Path to the exported ONNX file
    """
    print(f"Exporting {model_name} to ONNX...")
    
    batch_size, seq_len, features = input_shape
    dummy_input = torch.randn(batch_size, seq_len, features)
    
    onnx_path = exports_dir / f"{model_name}.onnx"
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"✓ ONNX export successful: {onnx_path}")
        return str(onnx_path)
        
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        raise

def convert_onnx_to_tensorflow(onnx_path: str, exports_dir: Path) -> str:
    """
    Convert ONNX model to TensorFlow SavedModel format.
    
    Args:
        onnx_path: Path to ONNX model
        exports_dir: Directory to save exports
        
    Returns:
        Path to the TensorFlow SavedModel directory
    """
    print(f"Converting ONNX to TensorFlow: {onnx_path}")
    
    try:
        import onnx_tf
        from onnx_tf.backend import prepare
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)
        
        # Save as SavedModel
        model_name = Path(onnx_path).stem
        tf_path = exports_dir / f"{model_name}_tf"
        tf_rep.export_graph(str(tf_path))
        
        print(f"✓ TensorFlow conversion successful: {tf_path}")
        return str(tf_path)
        
    except ImportError:
        print("onnx-tf not available, trying tf2onnx...")
        return convert_onnx_to_tensorflow_tf2onnx(onnx_path, exports_dir)
    except Exception as e:
        print(f"✗ TensorFlow conversion failed: {e}")
        raise

def convert_onnx_to_tensorflow_tf2onnx(onnx_path: str, exports_dir: Path) -> str:
    """
    Alternative ONNX to TensorFlow conversion using tf2onnx.
    
    Args:
        onnx_path: Path to ONNX model
        exports_dir: Directory to save exports
        
    Returns:
        Path to the TensorFlow SavedModel directory
    """
    print(f"Converting ONNX to TensorFlow using tf2onnx: {onnx_path}")
    
    try:
        import tf2onnx
        from tf2onnx import convert
        
        model_name = Path(onnx_path).stem
        tf_path = exports_dir / f"{model_name}_tf"
        
        # Convert ONNX to TensorFlow using tf2onnx
        convert.from_onnx(onnx_path, output_path=str(tf_path))
        
        print(f"✓ TensorFlow conversion successful: {tf_path}")
        return str(tf_path)
        
    except Exception as e:
        print(f"✗ TensorFlow conversion failed: {e}")
        raise

def convert_tensorflow_to_tflite(tf_path: str, exports_dir: Path, 
                                quantize_fp16: bool = True) -> str:
    """
    Convert TensorFlow SavedModel to TensorFlow Lite format.
    
    Args:
        tf_path: Path to TensorFlow SavedModel
        exports_dir: Directory to save exports
        quantize_fp16: Whether to apply FP16 quantization
        
    Returns:
        Path to the TFLite model file
    """
    print(f"Converting TensorFlow to TFLite: {tf_path}")
    
    try:
        # Load the SavedModel
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
        
        # Configure optimization
        if quantize_fp16:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            suffix = "fp16"
        else:
            suffix = "fp32"
        
        # Convert
        tflite_model = converter.convert()
        
        # Save
        model_name = Path(tf_path).stem.replace("_tf", "")
        tflite_path = exports_dir / f"{model_name}_{suffix}.tflite"
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✓ TFLite conversion successful: {tflite_path}")
        return str(tflite_path)
        
    except Exception as e:
        print(f"✗ TFLite conversion failed: {e}")
        raise

def verify_tflite_output(pytorch_model: torch.nn.Module, tflite_path: str, 
                        input_shape: Tuple[int, int, int]) -> float:
    """
    Verify TFLite model output by comparing with PyTorch model.
    
    Args:
        pytorch_model: Original PyTorch model
        tflite_path: Path to TFLite model
        input_shape: Input tensor shape
        
    Returns:
        Cosine similarity between outputs
    """
    print(f"Verifying TFLite model: {tflite_path}")
    
    try:
        # Generate test input
        batch_size, seq_len, features = input_shape
        test_input = torch.randn(batch_size, seq_len, features)
        
        # PyTorch inference
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input).numpy()
        
        # TFLite inference
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], test_input.numpy())
        interpreter.invoke()
        
        tflite_output = interpreter.get_tensor(output_details[0]['index'])
        
        # Calculate cosine similarity
        pytorch_flat = pytorch_output.flatten()
        tflite_flat = tflite_output.flatten()
        
        cosine_sim = np.dot(pytorch_flat, tflite_flat) / (
            np.linalg.norm(pytorch_flat) * np.linalg.norm(tflite_flat)
        )
        
        print(f"✓ Verification complete - Cosine similarity: {cosine_sim:.6f}")
        return cosine_sim
        
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return 0.0

def get_model_size(file_path: str) -> str:
    """Get human-readable file size."""
    size_bytes = os.path.getsize(file_path)
    
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/(1024**2):.1f} MB"
    else:
        return f"{size_bytes/(1024**3):.1f} GB"

def export_single_model(model_path: str, model_name: str, input_shape: Tuple[int, int, int], 
                        exports_dir: Path, verify: bool = True) -> dict:
    """
    Complete export pipeline for a single model.
    
    Args:
        model_path: Path to PyTorch model
        model_name: Name for the model
        input_shape: Input tensor shape
        exports_dir: Directory to save exports
        verify: Whether to verify the output
        
    Returns:
        Dictionary with export results
    """
    results = {
        'model_name': model_name,
        'input_shape': input_shape,
        'files': {},
        'sizes': {},
        'verification': {}
    }
    
    try:
        # Step 1: Load PyTorch model
        pytorch_model = load_pytorch_model(model_path)
        
        # Step 2: Export to ONNX
        onnx_path = export_to_onnx(pytorch_model, model_name, input_shape, exports_dir)
        results['files']['onnx'] = onnx_path
        results['sizes']['onnx'] = get_model_size(onnx_path)
        
        # Step 3: Convert ONNX to TensorFlow
        tf_path = convert_onnx_to_tensorflow(onnx_path, exports_dir)
        results['files']['tensorflow'] = tf_path
        results['sizes']['tensorflow'] = get_model_size(tf_path)
        
        # Step 4: Convert to TFLite (FP16)
        tflite_fp16_path = convert_tensorflow_to_tflite(tf_path, exports_dir, quantize_fp16=True)
        results['files']['tflite_fp16'] = tflite_fp16_path
        results['sizes']['tflite_fp16'] = get_model_size(tflite_fp16_path)
        
        # Step 5: Convert to TFLite (FP32)
        tflite_fp32_path = convert_tensorflow_to_tflite(tf_path, exports_dir, quantize_fp16=False)
        results['files']['tflite_fp32'] = tflite_fp32_path
        results['sizes']['tflite_fp32'] = get_model_size(tflite_fp32_path)
        
        # Step 6: Verification
        if verify:
            fp16_sim = verify_tflite_output(pytorch_model, tflite_fp16_path, input_shape)
            fp32_sim = verify_tflite_output(pytorch_model, tflite_fp32_path, input_shape)
            results['verification'] = {
                'fp16_cosine_similarity': fp16_sim,
                'fp32_cosine_similarity': fp32_sim
            }
        
        print(f"✓ {model_name} export completed successfully!")
        
    except Exception as e:
        print(f"✗ {model_name} export failed: {e}")
        results['error'] = str(e)
    
    return results

def main():
    """Main export pipeline."""
    parser = argparse.ArgumentParser(description='Export PyTorch models to TFLite format')
    parser.add_argument('--transformer', type=str, default='Transformer.pt',
                       help='Path to Transformer model')
    parser.add_argument('--gru', type=str, default='GRU.pt',
                       help='Path to GRU model')
    parser.add_argument('--sequence-length', type=int, default=300,
                       help='Sequence length for dummy input')
    parser.add_argument('--no-verify', action='store_true',
                       help='Skip verification step')
    parser.add_argument('--output-dir', type=str, default='exports',
                       help='Output directory for exported models')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PyTorch → ONNX → TensorFlow → TFLite Export Pipeline")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Setup
    exports_dir = Path(args.output_dir)
    exports_dir.mkdir(exist_ok=True)
    
    input_shape = (1, args.sequence_length, 178)  # B, T, 178
    print(f"Input shape: {input_shape}")
    print(f"Output directory: {exports_dir}")
    print()
    
    # Export results
    all_results = {}
    
    # Export Transformer model
    if os.path.exists(args.transformer):
        print("🔄 Exporting Transformer model...")
        transformer_results = export_single_model(
            args.transformer, 'sign_transformer', input_shape, exports_dir, 
            verify=not args.no_verify
        )
        all_results['transformer'] = transformer_results
        print()
    else:
        print(f"⚠️  Transformer model not found: {args.transformer}")
    
    # Export GRU model
    if os.path.exists(args.gru):
        print("🔄 Exporting GRU model...")
        gru_results = export_single_model(
            args.gru, 'mediapipe_gru', input_shape, exports_dir,
            verify=not args.no_verify
        )
        all_results['gru'] = gru_results
        print()
    else:
        print(f"⚠️  GRU model not found: {args.gru}")
    
    # Summary
    print("=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)
    
    for model_type, results in all_results.items():
        if 'error' in results:
            print(f"❌ {model_type.upper()}: FAILED - {results['error']}")
            continue
            
        print(f"✅ {model_type.upper()}: SUCCESS")
        print(f"   Files created:")
        for format_type, file_path in results['files'].items():
            size = results['sizes'].get(format_type, 'Unknown')
            print(f"     {format_type}: {Path(file_path).name} ({size})")
        
        if results['verification']:
            fp16_sim = results['verification'].get('fp16_cosine_similarity', 0)
            fp32_sim = results['verification'].get('fp32_cosine_similarity', 0)
            print(f"   Verification:")
            print(f"     FP16 similarity: {fp16_sim:.6f}")
            print(f"     FP32 similarity: {fp32_sim:.6f}")
        print()
    
    # Save results to JSON
    results_file = exports_dir / "export_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"📄 Detailed results saved to: {results_file}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()

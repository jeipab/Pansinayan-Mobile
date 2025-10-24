#!/usr/bin/env python3
"""
CTC Model Export Pipeline: PyTorch → ONNX → TensorFlow → TensorFlow Lite

This script converts PyTorch CTC models to TFLite format for continuous sign language recognition.
Supports both Transformer and GRU CTC models with FP16 quantization for Android deployment.

Usage:
    python scripts/export_ctc_models.py

Requirements:
    - CTC-trained PyTorch models (.pt files) in the current directory
    - All dependencies from requirements.txt installed

Output:
    - exports/ctc/sign_transformer_ctc_fp16.tflite
    - exports/ctc/sign_transformer_ctc_fp32.tflite
    - exports/ctc/mediapipe_gru_ctc_fp16.tflite
    - exports/ctc/mediapipe_gru_ctc_fp32.tflite

CTC Model Specifications:
    - Input: [1, 300, 178] (batch_size=1, sequence_length=300, features=178)
    - Output: [1, 300, 106] (batch_size=1, sequence_length=300, classes=105 glosses + 1 blank)
    - Keypoint Structure: 89 keypoints × 2 coordinates = 178 features
"""

import os
import sys
import torch
import onnx
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import argparse
import json
from datetime import datetime
import logging

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# CTC Model Constants
CTC_INPUT_SHAPE = (1, 300, 178)  # batch_size=1, sequence_length=300, features=178
CTC_OUTPUT_CLASSES = 106  # 105 glosses + 1 blank token
CTC_BLANK_TOKEN_IDX = 105  # Blank token index
CTC_MIN_SEQUENCE_LENGTH = 30  # Minimum frames for CTC inference


def setup_directories() -> Path:
    """Create exports directory structure for CTC models."""
    exports_dir = Path("exports")
    ctc_dir = exports_dir / "ctc"
    ctc_dir.mkdir(parents=True, exist_ok=True)
    return ctc_dir


def load_pytorch_ctc_model(model_path: str) -> torch.nn.Module:
    """
    Load PyTorch CTC model from checkpoint.
    
    Args:
        model_path: Path to the .pt file
        
    Returns:
        Loaded PyTorch CTC model in evaluation mode
    """
    logger.info(f"Loading PyTorch CTC model from {model_path}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CTC model file not found: {model_path}")
    
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
            model = checkpoint
            model.eval()
            return model
        
        # Create a dummy CTC model structure
        # Note: This is a generic approach - modify based on your specific CTC model architecture
        model = torch.nn.Module()
        model.load_state_dict(model_state_dict)
        model.eval()
        
        logger.info(f"✓ CTC model loaded successfully: {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading CTC model {model_path}: {e}")
        logger.info("Attempting to load as direct model object...")
        try:
            model = torch.load(model_path, map_location='cpu')
            if hasattr(model, 'eval'):
                model.eval()
            return model
        except Exception as e2:
            raise RuntimeError(f"Failed to load CTC model {model_path}: {e2}")


def export_to_onnx(model: torch.nn.Module, model_name: str, 
                   input_shape: Tuple[int, int, int], exports_dir: Path) -> str:
    """
    Export PyTorch CTC model to ONNX format with dynamic sequence support.
    
    Args:
        model: PyTorch CTC model
        model_name: Name for the exported file
        input_shape: Input tensor shape (B, T, 178)
        exports_dir: Directory to save exports
        
    Returns:
        Path to the exported ONNX file
    """
    logger.info(f"Exporting {model_name} CTC model to ONNX...")
    
    batch_size, seq_len, features = input_shape
    dummy_input = torch.randn(batch_size, seq_len, features)
    
    onnx_path = exports_dir / f"{model_name}_ctc.onnx"
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=17,  # Latest stable ONNX opset
            do_constant_folding=True,
            input_names=['ctc_input'],
            output_names=['ctc_logits'],
            dynamic_axes={
                'ctc_input': {0: 'batch_size', 1: 'sequence_length'},
                'ctc_logits': {0: 'batch_size', 1: 'sequence_length'}
            },
            verbose=False
        )
        
        logger.info(f"✓ ONNX CTC export successful: {onnx_path}")
        return str(onnx_path)
        
    except Exception as e:
        logger.error(f"✗ ONNX CTC export failed: {e}")
        raise


def convert_to_tf(onnx_path: str, exports_dir: Path) -> str:
    """
    Convert ONNX CTC model to TensorFlow SavedModel format.
    
    Args:
        onnx_path: Path to ONNX model
        exports_dir: Directory to save exports
        
    Returns:
        Path to the TensorFlow SavedModel directory
    """
    logger.info(f"Converting ONNX CTC model to TensorFlow: {onnx_path}")
    
    try:
        import onnx_tf
        from onnx_tf.backend import prepare
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)
        
        # Save as SavedModel
        model_name = Path(onnx_path).stem.replace("_ctc", "")
        tf_path = exports_dir / f"{model_name}_ctc_tf"
        tf_rep.export_graph(str(tf_path))
        
        logger.info(f"✓ TensorFlow CTC conversion successful: {tf_path}")
        return str(tf_path)
        
    except ImportError:
        logger.warning("onnx-tf not available, trying tf2onnx...")
        return convert_to_tf_tf2onnx(onnx_path, exports_dir)
    except Exception as e:
        logger.error(f"✗ TensorFlow CTC conversion failed: {e}")
        raise


def convert_to_tf_tf2onnx(onnx_path: str, exports_dir: Path) -> str:
    """
    Alternative ONNX to TensorFlow conversion using tf2onnx.
    
    Args:
        onnx_path: Path to ONNX model
        exports_dir: Directory to save exports
        
    Returns:
        Path to the TensorFlow SavedModel directory
    """
    logger.info(f"Converting ONNX CTC model to TensorFlow using tf2onnx: {onnx_path}")
    
    try:
        import tf2onnx
        from tf2onnx import convert
        
        model_name = Path(onnx_path).stem.replace("_ctc", "")
        tf_path = exports_dir / f"{model_name}_ctc_tf"
        
        # Convert ONNX to TensorFlow using tf2onnx
        convert.from_onnx(onnx_path, output_path=str(tf_path))
        
        logger.info(f"✓ TensorFlow CTC conversion successful: {tf_path}")
        return str(tf_path)
        
    except Exception as e:
        logger.error(f"✗ TensorFlow CTC conversion failed: {e}")
        raise


def convert_to_tflite(tf_path: str, exports_dir: Path, 
                     quantize_fp16: bool = True) -> str:
    """
    Convert TensorFlow SavedModel to TensorFlow Lite format for CTC models.
    
    Args:
        tf_path: Path to TensorFlow SavedModel
        exports_dir: Directory to save exports
        quantize_fp16: Whether to apply FP16 quantization
        
    Returns:
        Path to the TFLite CTC model file
    """
    logger.info(f"Converting TensorFlow CTC model to TFLite: {tf_path}")
    
    try:
        # Load the SavedModel
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
        
        # Configure optimization for CTC models
        if quantize_fp16:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            suffix = "fp16"
        else:
            suffix = "fp32"
        
        # Convert
        tflite_model = converter.convert()
        
        # Save
        model_name = Path(tf_path).stem.replace("_ctc_tf", "")
        tflite_path = exports_dir / f"{model_name}_ctc_{suffix}.tflite"
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info(f"✓ TFLite CTC conversion successful: {tflite_path}")
        return str(tflite_path)
        
    except Exception as e:
        logger.error(f"✗ TFLite CTC conversion failed: {e}")
        raise


def verify_tflite_model(file_path: str, input_shape: Tuple[int, int, int]) -> Dict[str, Any]:
    """
    Verify TFLite CTC model by performing dummy inference and checking tensor shapes.
    
    Args:
        file_path: Path to TFLite model
        input_shape: Expected input shape
        
    Returns:
        Dictionary with verification results
    """
    logger.info(f"Verifying CTC TFLite model: {file_path}")
    
    try:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=file_path)
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Verify input shape
        actual_input_shape = input_details[0]['shape']
        expected_input_shape = list(input_shape)
        
        # Verify output shape
        actual_output_shape = output_details[0]['shape']
        expected_output_shape = [input_shape[0], input_shape[1], CTC_OUTPUT_CLASSES]
        
        # Prepare test input
        batch_size, seq_len, features = input_shape
        test_input = np.random.randn(batch_size, seq_len, features).astype(np.float32)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Verify shapes
        input_shape_match = actual_input_shape == expected_input_shape
        output_shape_match = actual_output_shape == expected_output_shape
        
        # Check output properties
        output_is_logits = np.all(np.isfinite(output_data))
        output_range_valid = np.all(output_data >= -10) and np.all(output_data <= 10)
        
        verification_results = {
            'file_path': file_path,
            'input_shape_match': input_shape_match,
            'output_shape_match': output_shape_match,
            'actual_input_shape': actual_input_shape,
            'expected_input_shape': expected_input_shape,
            'actual_output_shape': actual_output_shape,
            'expected_output_shape': expected_output_shape,
            'output_is_logits': output_is_logits,
            'output_range_valid': output_range_valid,
            'model_size_mb': os.path.getsize(file_path) / (1024 * 1024),
            'verification_passed': input_shape_match and output_shape_match and output_is_logits and output_range_valid
        }
        
        if verification_results['verification_passed']:
            logger.info(f"✓ CTC model verification passed: {file_path}")
        else:
            logger.error(f"✗ CTC model verification failed: {file_path}")
            
        return verification_results
        
    except Exception as e:
        logger.error(f"✗ CTC model verification failed: {e}")
        return {
            'file_path': file_path,
            'verification_passed': False,
            'error': str(e)
        }


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


def export_ctc_model(model_path: str, model_name: str, 
                    input_shape: Tuple[int, int, int], 
                    exports_dir: Path, verify: bool = True) -> Dict[str, Any]:
    """
    Complete CTC export pipeline for a single model.
    
    Args:
        model_path: Path to PyTorch CTC model
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
        # Step 1: Load PyTorch CTC model
        pytorch_model = load_pytorch_ctc_model(model_path)
        
        # Step 2: Export to ONNX
        onnx_path = export_to_onnx(pytorch_model, model_name, input_shape, exports_dir)
        results['files']['onnx'] = onnx_path
        results['sizes']['onnx'] = get_model_size(onnx_path)
        
        # Step 3: Convert ONNX to TensorFlow
        tf_path = convert_to_tf(onnx_path, exports_dir)
        results['files']['tensorflow'] = tf_path
        results['sizes']['tensorflow'] = get_model_size(tf_path)
        
        # Step 4: Convert to TFLite (FP16)
        tflite_fp16_path = convert_to_tflite(tf_path, exports_dir, quantize_fp16=True)
        results['files']['tflite_fp16'] = tflite_fp16_path
        results['sizes']['tflite_fp16'] = get_model_size(tflite_fp16_path)
        
        # Step 5: Convert to TFLite (FP32)
        tflite_fp32_path = convert_to_tflite(tf_path, exports_dir, quantize_fp16=False)
        results['files']['tflite_fp32'] = tflite_fp32_path
        results['sizes']['tflite_fp32'] = get_model_size(tflite_fp32_path)
        
        # Step 6: Verification
        if verify:
            fp16_verification = verify_tflite_model(tflite_fp16_path, input_shape)
            fp32_verification = verify_tflite_model(tflite_fp32_path, input_shape)
            results['verification'] = {
                'fp16': fp16_verification,
                'fp32': fp32_verification
            }
        
        logger.info(f"✓ {model_name} CTC export completed successfully!")
        
    except Exception as e:
        logger.error(f"✗ {model_name} CTC export failed: {e}")
        results['error'] = str(e)
    
    return results


def main():
    """Main CTC export pipeline."""
    parser = argparse.ArgumentParser(description='Export PyTorch CTC models to TFLite format')
    parser.add_argument('--transformer', type=str, default='SignTransformer_best.pt',
                       help='Path to Transformer CTC model')
    parser.add_argument('--gru', type=str, default='MediaPipeGRU_best.pt',
                       help='Path to GRU CTC model')
    parser.add_argument('--sequence-length', type=int, default=300,
                       help='Sequence length for CTC models')
    parser.add_argument('--no-verify', action='store_true',
                       help='Skip verification step')
    parser.add_argument('--output-dir', type=str, default='exports/ctc',
                       help='Output directory for exported CTC models')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CTC Model Export Pipeline: PyTorch → ONNX → TensorFlow → TFLite")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Setup
    exports_dir = Path(args.output_dir)
    exports_dir.mkdir(parents=True, exist_ok=True)
    
    input_shape = (1, args.sequence_length, 178)  # B, T, 178
    print(f"CTC Input shape: {input_shape}")
    print(f"CTC Output shape: [1, {args.sequence_length}, {CTC_OUTPUT_CLASSES}]")
    print(f"Output directory: {exports_dir}")
    print()
    
    # Export results
    all_results = {}
    
    # Export Transformer CTC model
    if os.path.exists(args.transformer):
        print("🔄 Exporting Transformer CTC model...")
        transformer_results = export_ctc_model(
            args.transformer, 'sign_transformer', input_shape, exports_dir, 
            verify=not args.no_verify
        )
        all_results['transformer_ctc'] = transformer_results
        print()
    else:
        print(f"⚠️  Transformer CTC model not found: {args.transformer}")
    
    # Export GRU CTC model
    if os.path.exists(args.gru):
        print("🔄 Exporting GRU CTC model...")
        gru_results = export_ctc_model(
            args.gru, 'mediapipe_gru', input_shape, exports_dir,
            verify=not args.no_verify
        )
        all_results['gru_ctc'] = gru_results
        print()
    else:
        print(f"⚠️  GRU CTC model not found: {args.gru}")
    
    # Summary
    print("=" * 70)
    print("CTC EXPORT SUMMARY")
    print("=" * 70)
    
    for model_type, results in all_results.items():
        if 'error' in results:
            print(f"❌ {model_type.upper()}: FAILED - {results['error']}")
            continue
            
        print(f"✅ {model_type.upper()}: SUCCESS")
        print(f"   CTC Files created:")
        for format_type, file_path in results['files'].items():
            size = results['sizes'].get(format_type, 'Unknown')
            print(f"     {format_type}: {Path(file_path).name} ({size})")
        
        if results['verification']:
            fp16_verification = results['verification'].get('fp16', {})
            fp32_verification = results['verification'].get('fp32', {})
            print(f"   CTC Verification:")
            print(f"     FP16: {'✓ PASSED' if fp16_verification.get('verification_passed', False) else '✗ FAILED'}")
            print(f"     FP32: {'✓ PASSED' if fp32_verification.get('verification_passed', False) else '✗ FAILED'}")
        print()
    
    # Save results to JSON
    results_file = exports_dir / "ctc_export_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"📄 Detailed CTC results saved to: {results_file}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("🎯 Next Steps:")
    print("   1. Copy CTC models to app/src/main/assets/ctc/")
    print("   2. Test continuous recognition in Android app")
    print("   3. Verify CTC decoding performance")


if __name__ == "__main__":
    main()

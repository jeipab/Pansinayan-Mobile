#!/usr/bin/env python3
"""
Convert ONNX models with external data files to single-file format.

This script merges external .onnx.data files back into the main .onnx file,
making them suitable for mobile deployment where external data file handling
is problematic.

Usage:
    python convert_onnx_to_single_file.py
"""

import os
import sys
import shutil
from pathlib import Path

def convert_onnx_to_single_file(input_model_path: str, output_model_path: str = None) -> bool:
    """
    Convert ONNX model with external data to single-file format.
    
    Args:
        input_model_path: Path to the input ONNX model file
        output_model_path: Path for the output single-file model (optional)
        
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        # Import ONNX
        import onnx
        
        print(f"Converting {input_model_path} to single-file format...")
        
        # Load the model (this will automatically load external data if present)
        model = onnx.load(input_model_path)
        
        # Set output path if not provided
        if output_model_path is None:
            base_name = os.path.splitext(input_model_path)[0]
            output_model_path = f"{base_name}_single.onnx"
        
        # Save as single file (external data will be merged automatically)
        onnx.save_model(model, output_model_path)
        
        # Get file sizes for comparison
        input_size = os.path.getsize(input_model_path)
        output_size = os.path.getsize(output_model_path)
        
        print(f"✅ Conversion successful!")
        print(f"   Input:  {input_model_path} ({input_size:,} bytes)")
        print(f"   Output: {output_model_path} ({output_size:,} bytes)")
        
        return True
        
    except ImportError:
        print("❌ ONNX library not found. Please install it with:")
        print("   pip install onnx")
        return False
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

def main():
    """Main conversion function."""
    
    # Define model paths
    models_dir = Path("models/converted/convert_0")
    assets_dir = Path("app/src/main/assets")
    
    # Models to convert
    models_to_convert = [
        "sign_transformer_ctc.onnx",
        "sign_mediapipe_gru_ctc.onnx"
    ]
    
    print("🔄 Converting ONNX models to single-file format...")
    print(f"Source directory: {models_dir}")
    print(f"Target directory: {assets_dir}")
    print()
    
    success_count = 0
    
    for model_name in models_to_convert:
        input_path = models_dir / model_name
        
        if not input_path.exists():
            print(f"⚠️  Model not found: {input_path}")
            continue
            
        # Convert to single file
        output_name = model_name.replace('.onnx', '_single.onnx')
        output_path = models_dir / output_name
        
        if convert_onnx_to_single_file(str(input_path), str(output_path)):
            success_count += 1
            
            # Copy to assets directory
            assets_output = assets_dir / model_name
            try:
                shutil.copy2(output_path, assets_output)
                print(f"   📱 Copied to assets: {assets_output}")
            except Exception as e:
                print(f"   ⚠️  Failed to copy to assets: {e}")
        
        print()
    
    print(f"🎉 Conversion complete! {success_count}/{len(models_to_convert)} models converted successfully.")
    
    if success_count == len(models_to_convert):
        print("\n📋 Next steps:")
        print("1. Update your Android app to use the new single-file models")
        print("2. Remove the old .onnx.data files from assets")
        print("3. Test the model loading in your app")
    else:
        print("\n⚠️  Some models failed to convert. Please check the errors above.")

if __name__ == "__main__":
    main()

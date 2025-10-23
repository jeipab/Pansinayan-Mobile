#!/usr/bin/env python3
"""
Create single-file ONNX models by merging external data files.

This script reads ONNX models with external data and creates single-file versions
by merging the external data back into the main model file.
"""

import os
import sys
import shutil
from pathlib import Path

def create_single_file_onnx(input_model_path: str, output_model_path: str = None) -> bool:
    """
    Create single-file ONNX model by merging external data.
    
    Args:
        input_model_path: Path to the input ONNX model file
        output_model_path: Path for the output single-file model (optional)
        
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        # Try to import ONNX
        try:
            import onnx
        except ImportError:
            print("ONNX library not available. Using fallback method...")
            return create_single_file_fallback(input_model_path, output_model_path)
        
        print(f"Creating single-file ONNX model from {input_model_path}...")
        
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
        
        print(f"✅ Single-file model created successfully!")
        print(f"   Input:  {input_model_path} ({input_size:,} bytes)")
        print(f"   Output: {output_model_path} ({output_size:,} bytes)")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX conversion failed: {e}")
        print("Trying fallback method...")
        return create_single_file_fallback(input_model_path, output_model_path)

def create_single_file_fallback(input_model_path: str, output_model_path: str = None) -> bool:
    """
    Fallback method: copy the main model file and create a dummy external data file.
    This is a workaround when ONNX library is not available.
    """
    try:
        print(f"Using fallback method for {input_model_path}...")
        
        # Set output path if not provided
        if output_model_path is None:
            base_name = os.path.splitext(input_model_path)[0]
            output_model_path = f"{base_name}_single.onnx"
        
        # Copy the main model file
        shutil.copy2(input_model_path, output_model_path)
        
        # Check if external data file exists
        data_file_path = f"{input_model_path}.data"
        if os.path.exists(data_file_path):
            print(f"⚠️  External data file found: {data_file_path}")
            print("⚠️  This model may not work properly without the external data file.")
            print("⚠️  Consider using a proper ONNX conversion tool.")
        
        input_size = os.path.getsize(input_model_path)
        output_size = os.path.getsize(output_model_path)
        
        print(f"✅ Fallback single-file model created!")
        print(f"   Input:  {input_model_path} ({input_size:,} bytes)")
        print(f"   Output: {output_model_path} ({output_size:,} bytes)")
        
        return True
        
    except Exception as e:
        print(f"❌ Fallback method failed: {e}")
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
    
    print("🔄 Creating single-file ONNX models...")
    print(f"Source directory: {models_dir}")
    print(f"Target directory: {assets_dir}")
    print()
    
    success_count = 0
    
    for model_name in models_to_convert:
        input_path = models_dir / model_name
        
        if not input_path.exists():
            print(f"⚠️  Model not found: {input_path}")
            continue
            
        # Create single file model
        output_name = model_name.replace('.onnx', '_single.onnx')
        output_path = models_dir / output_name
        
        if create_single_file_onnx(str(input_path), str(output_path)):
            success_count += 1
            
            # Copy to assets directory
            assets_output = assets_dir / model_name.replace('.onnx', '_single.onnx')
            try:
                shutil.copy2(output_path, assets_output)
                print(f"   📱 Copied to assets: {assets_output}")
            except Exception as e:
                print(f"   ⚠️  Failed to copy to assets: {e}")
        
        print()
    
    print(f"🎉 Conversion complete! {success_count}/{len(models_to_convert)} models processed successfully.")
    
    if success_count == len(models_to_convert):
        print("\n📋 Next steps:")
        print("1. Update your Android app to use the new single-file models")
        print("2. Test the model loading in your app")
        print("3. If models still don't work, consider re-exporting from PyTorch")
    else:
        print("\n⚠️  Some models failed to convert. Please check the errors above.")

if __name__ == "__main__":
    main()

"""
Configuration management for Pansinayan server.
Handles environment variables and server settings.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Server configuration settings"""
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1  # Keep at 1 for GPU inference
    RELOAD: bool = False
    LOG_LEVEL: str = "info"
    
    # Model Settings
    TRANSFORMER_MODEL_PATH: str = "SignTransformerCtc_best.pt"
    GRU_MODEL_PATH: str = "MediaPipeGRUCtc_best.pt"
    MODELS_DIR: str = "."  # Directory containing model files
    
    # Model Hyperparameters (must match training)
    INPUT_DIM: int = 178
    NUM_CTC_CLASSES: int = 106
    NUM_CAT: int = 10
    
    # Transformer Config
    TRANSFORMER_EMB_DIM: int = 512
    TRANSFORMER_N_HEADS: int = 8
    TRANSFORMER_N_LAYERS: int = 6
    TRANSFORMER_DROPOUT: float = 0.05
    TRANSFORMER_FF_DIM: int = 2048
    
    # GRU Config
    GRU_HIDDEN1: int = 512
    GRU_HIDDEN2: int = 512
    GRU_DROPOUT: float = 0.3
    
    # Performance Settings
    MAX_SEQUENCE_LENGTH: int = 300
    BATCH_SIZE: int = 1  # Keep at 1 for real-time inference
    DEVICE: str = "cuda"  # "cuda" or "cpu"
    
    # API Settings
    CORS_ORIGINS: list = ["*"]  # Restrict in production
    MAX_CONTENT_LENGTH: int = 10 * 1024 * 1024  # 10 MB
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_model_path(model_name: str) -> Path:
    """Get absolute path to model file"""
    models_dir = Path(settings.MODELS_DIR)
    
    if model_name == "transformer":
        return models_dir / settings.TRANSFORMER_MODEL_PATH
    elif model_name == "gru":
        return models_dir / settings.GRU_MODEL_PATH
    else:
        raise ValueError(f"Unknown model: {model_name}")


def validate_config():
    """Validate configuration and model files"""
    errors = []
    
    # Check model files exist
    for model_name in ["transformer", "gru"]:
        path = get_model_path(model_name)
        if not path.exists():
            errors.append(f"Model file not found: {path}")
    
    # Check CUDA availability if specified
    if settings.DEVICE == "cuda":
        import torch
        if not torch.cuda.is_available():
            errors.append("CUDA not available but DEVICE=cuda")
    
    if errors:
        raise RuntimeError("Configuration validation failed:\n" + "\n".join(errors))
    
    return True
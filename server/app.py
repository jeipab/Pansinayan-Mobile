"""
FastAPI server for Pansinayan sign language recognition.
Handles model inference requests from Android clients.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from collections import defaultdict
import torch
import time
import logging

from config import settings, validate_config, get_model_path
from logger_config import setup_logging
from utils import (
    get_system_stats, 
    validate_keypoints, 
    preprocess_keypoints,
    InferenceTimer
)
from transformer import SignTransformerCtc
from mediapipe_gru import MediaPipeGRUCtc

# Setup logging
logger = setup_logging(settings.LOG_LEVEL)

# Initialize FastAPI app
app = FastAPI(
    title="Pansinayan Server",
    description="Sign Language Recognition Inference Server",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage
models: Dict[str, torch.nn.Module] = {}
device: Optional[torch.device] = None

# Request metrics tracking
request_metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "model_usage": defaultdict(int),
    "total_inference_time_ms": 0.0,
    "total_request_time_ms": 0.0,
}

# ==================== Request/Response Models ====================

class InferenceRequest(BaseModel):
    """Request body for inference endpoint"""
    keypoints: List[List[float]] = Field(
        ..., 
        description="Keypoint sequence [T, 178]",
        min_items=1,
        max_items=300
    )
    model_type: str = Field(
        "transformer",
        description="Model to use: 'transformer' or 'gru'"
    )
    
    @validator('model_type')
    def validate_model_type(cls, v):
        if v.lower() not in ['transformer', 'gru']:
            raise ValueError("model_type must be 'transformer' or 'gru'")
        return v.lower()
    
    @validator('keypoints')
    def validate_keypoints_structure(cls, v):
        if not v:
            raise ValueError("keypoints cannot be empty")
        if len(v[0]) != 178:
            raise ValueError(f"Each frame must have 178 features, got {len(v[0])}")
        # Validate all frames have same dimension
        for i, frame in enumerate(v):
            if len(frame) != 178:
                raise ValueError(f"Frame {i} has {len(frame)} features, expected 178")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "keypoints": [[0.5] * 178] * 150,
                "model_type": "transformer"
            }
        }


class InferenceResponse(BaseModel):
    """Response body for inference endpoint"""
    ctc_log_probs: List[List[float]] = Field(
        ..., 
        description="CTC log probabilities [T, num_ctc]"
    )
    cat_logits: Optional[List[List[float]]] = Field(
        None, 
        description="Category logits [T, num_cat]"
    )
    sequence_length: int = Field(..., description="Sequence length T")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    model_used: str = Field(..., description="Model that was used")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: List[str]
    device: str
    gpu_available: bool
    gpu_name: Optional[str] = None
    system_stats: Optional[dict] = None


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None


# ==================== Model Loading Helpers ====================

def load_model_checkpoint(model_path: str, device: torch.device) -> Dict:
    """
    Load model checkpoint from file.
    
    Args:
        model_path: Path to model checkpoint file
        device: Target device for loading
        
    Returns:
        Checkpoint dictionary or state dict
    """
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        return checkpoint['model']
    return checkpoint


def load_transformer_model(device: torch.device) -> Optional[torch.nn.Module]:
    """
    Load and initialize Transformer model.
    
    Args:
        device: Target device for model
        
    Returns:
        Loaded model or None if loading fails
    """
    try:
        logger.info("Loading SignTransformerCtc...")
        model = SignTransformerCtc(
            input_dim=settings.INPUT_DIM,
            num_ctc_classes=settings.NUM_CTC_CLASSES,
            num_cat=settings.NUM_CAT,
            emb_dim=settings.TRANSFORMER_EMB_DIM,
            n_heads=settings.TRANSFORMER_N_HEADS,
            n_layers=settings.TRANSFORMER_N_LAYERS,
            dropout=settings.TRANSFORMER_DROPOUT,
            ff_dim=settings.TRANSFORMER_FF_DIM
        )
        
        model_path = get_model_path("transformer")
        state_dict = load_model_checkpoint(model_path, device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"✓ Transformer loaded from {model_path}")
        logger.info(f"  Parameters: {param_count:,}")
        
        return model
    except Exception as e:
        logger.error(f"Failed to load Transformer model: {e}", exc_info=True)
        return None


def load_gru_model(device: torch.device) -> Optional[torch.nn.Module]:
    """
    Load and initialize GRU model.
    
    Args:
        device: Target device for model
        
    Returns:
        Loaded model or None if loading fails
    """
    try:
        logger.info("Loading MediaPipeGRUCtc...")
        model = MediaPipeGRUCtc(
            input_dim=settings.INPUT_DIM,
            num_ctc_classes=settings.NUM_CTC_CLASSES,
            num_cat=settings.NUM_CAT,
            hidden1=settings.GRU_HIDDEN1,
            hidden2=settings.GRU_HIDDEN2,
            dropout=settings.GRU_DROPOUT
        )
        
        model_path = get_model_path("gru")
        state_dict = load_model_checkpoint(model_path, device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"✓ GRU loaded from {model_path}")
        logger.info(f"  Parameters: {param_count:,}")
        
        return model
    except Exception as e:
        logger.error(f"Failed to load GRU model: {e}", exc_info=True)
        return None


def warmup_model(model: torch.nn.Module, device: torch.device, model_name: str) -> None:
    """
    Warm up model with dummy inference to optimize first real request.
    
    Args:
        model: Model to warm up
        device: Device model is on
        model_name: Name of model for logging
    """
    try:
        logger.info(f"Warming up {model_name} model...")
        dummy_input = torch.randn(1, 50, settings.INPUT_DIM, device=device)
        with torch.no_grad():
            _ = model(dummy_input)
        logger.info(f"✓ {model_name} model warmed up")
    except Exception as e:
        logger.warning(f"Model warmup failed for {model_name}: {e}")


# ==================== Startup/Shutdown ====================

@app.on_event("startup")
async def startup_event() -> None:
    """
    Load models and initialize server.
    
    Models are loaded independently - if one fails, the other can still be used.
    After loading, models are warmed up with dummy inference.
    """
    global device, models
    
    logger.info("=" * 60)
    logger.info("Starting Pansinayan Server")
    logger.info("=" * 60)
    
    try:
        # Validate configuration
        logger.info("Validating configuration...")
        validate_config()
        logger.info("✓ Configuration valid")
        
        # Set device
        device = torch.device(settings.DEVICE if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Load models independently (allow partial availability)
        transformer = load_transformer_model(device)
        if transformer is not None:
            models["transformer"] = transformer
            warmup_model(transformer, device, "Transformer")
        
        gru = load_gru_model(device)
        if gru is not None:
            models["gru"] = gru
            warmup_model(gru, device, "GRU")
        
        # Check if at least one model loaded
        if not models:
            raise RuntimeError("Failed to load any models. Server cannot start.")
        
        logger.info("=" * 60)
        logger.info(f"Server ready to accept requests ({len(models)} model(s) loaded)")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """
    Cleanup on shutdown.
    
    Clears models from memory and frees GPU cache.
    """
    logger.info("Shutting down server...")
    models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Server shutdown complete")


# ==================== Exception Handlers ====================

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


# ==================== API Endpoints ====================

@app.get("/")
async def root() -> Dict:
    """
    Root endpoint with API information.
    
    Returns:
        Dictionary with server information and available endpoints
    """
    return {
        "message": "Pansinayan Sign Language Recognition Server",
        "version": "1.0.0",
        "status": "running",
        "models_available": list(models.keys()),
        "endpoints": {
            "health": "GET /health - Server health check",
            "predict": "POST /predict - Run inference on keypoints",
            "stats": "GET /stats - Get system statistics"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Check server health and model status.
    
    Returns information about loaded models and system resources.
    Server is considered healthy if at least one model is loaded.
    
    Returns:
        HealthResponse with server status and model information
    """
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    
    return HealthResponse(
        status="healthy" if models else "unhealthy",
        models_loaded=list(models.keys()),
        device=str(device) if device else "unknown",
        gpu_available=torch.cuda.is_available(),
        gpu_name=gpu_name,
        system_stats=get_system_stats()
    )


@app.get("/stats")
async def get_stats() -> Dict:
    """
    Get detailed system statistics including request metrics.
    
    Returns:
        Dictionary containing system stats, GPU info, and request metrics
    """
    stats = get_system_stats()
    
    if torch.cuda.is_available():
        stats["gpu_name"] = torch.cuda.get_device_name(0)
        stats["gpu_count"] = torch.cuda.device_count()
    
    stats["models_loaded"] = list(models.keys())
    stats["device"] = str(device)
    
    # Add request metrics
    total = request_metrics["total_requests"]
    if total > 0:
        stats["request_metrics"] = {
            "total_requests": total,
            "successful_requests": request_metrics["successful_requests"],
            "failed_requests": request_metrics["failed_requests"],
            "success_rate": request_metrics["successful_requests"] / total,
            "avg_inference_time_ms": request_metrics["total_inference_time_ms"] / total,
            "avg_request_time_ms": request_metrics["total_request_time_ms"] / total,
            "model_usage": dict(request_metrics["model_usage"]),
        }
    else:
        stats["request_metrics"] = {
            "total_requests": 0,
            "message": "No requests processed yet"
        }
    
    return stats


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest) -> InferenceResponse:
    """
    Run inference on keypoint sequence.
    
    This endpoint accepts a sequence of MediaPipe keypoints and returns
    CTC log probabilities for sign language recognition.
    
    The client uses activity-based inference, sending sign-aligned windows
    (variable length) when sign boundaries are detected. The server processes
    whatever sequence length is received (1-300 frames).
    
    Args:
        request: InferenceRequest containing keypoints and model selection
                 - keypoints: Variable-length sequence [T, 178] where T is 
                   determined by sign duration (1-300 frames)
                 - model_type: "transformer" or "gru"
        
    Returns:
        InferenceResponse with CTC predictions and timing information
        
    Raises:
        HTTPException: If model not found or inference fails
    """
    request_metrics["total_requests"] += 1
    
    try:
        with InferenceTimer("total_request") as total_timer:
            # Validate model availability
            model_type = request.model_type
            if model_type not in models:
                request_metrics["failed_requests"] += 1
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{model_type}' not loaded. Available: {list(models.keys())}"
                )
            
            # Validate keypoints (Pydantic already validates structure, but double-check)
            is_valid, error_msg = validate_keypoints(
                request.keypoints, 
                settings.MAX_SEQUENCE_LENGTH
            )
            if not is_valid:
                request_metrics["failed_requests"] += 1
                raise HTTPException(status_code=400, detail=error_msg)
            
            # Preprocess keypoints
            with InferenceTimer("preprocessing"):
                keypoints_np = preprocess_keypoints(request.keypoints)
                T = keypoints_np.shape[0]
                
                # Convert to tensor [1, T, 178]
                keypoints_tensor = torch.from_numpy(keypoints_np).unsqueeze(0).to(device)
            
            logger.debug(f"Running {model_type} inference on sequence length {T}")
            
            # Run inference
            model = models[model_type]
            with InferenceTimer("inference") as inf_timer:
                with torch.no_grad():
                    output = model(keypoints_tensor)
            
            # Process outputs
            with InferenceTimer("postprocessing"):
                if isinstance(output, tuple):
                    ctc_log_probs, cat_logits = output
                    cat_logits_list = cat_logits[0].cpu().numpy().tolist()
                else:
                    ctc_log_probs = output
                    cat_logits_list = None
                
                ctc_log_probs_list = ctc_log_probs[0].cpu().numpy().tolist()
            
            # Update metrics
            request_metrics["successful_requests"] += 1
            request_metrics["model_usage"][model_type] += 1
            request_metrics["total_inference_time_ms"] += inf_timer.elapsed_ms
            request_metrics["total_request_time_ms"] += total_timer.elapsed_ms
            
            logger.info(
                f"Inference complete: {inf_timer.elapsed_ms:.2f}ms inference, "
                f"{total_timer.elapsed_ms:.2f}ms total"
            )
            
            # Cleanup GPU memory if needed
            if torch.cuda.is_available() and torch.cuda.memory_allocated() / 1e9 > 8.0:
                torch.cuda.empty_cache()
            
            return InferenceResponse(
                ctc_log_probs=ctc_log_probs_list,
                cat_logits=cat_logits_list,
                sequence_length=T,
                inference_time_ms=inf_timer.elapsed_ms,
                model_used=model_type
            )
    
    except HTTPException:
        request_metrics["failed_requests"] += 1
        raise
    except Exception as e:
        request_metrics["failed_requests"] += 1
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Inference failed: {str(e)}"
        )


# ==================== Development Endpoints ====================

if settings.RELOAD:  # Only in development mode
    
    @app.post("/reload-models")
    async def reload_models():
        """Reload models from disk (development only)"""
        try:
            await shutdown_event()
            await startup_event()
            return {"status": "success", "message": "Models reloaded"}
        except Exception as e:
            logger.error(f"Failed to reload models: {e}")
            return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )
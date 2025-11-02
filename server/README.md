# Pansinayan Server

FastAPI-based inference server for Filipino Sign Language recognition.

## Features

- ✅ PyTorch model inference (Transformer & GRU)
- ✅ RESTful API with automatic documentation
- ✅ GPU acceleration support
- ✅ Health monitoring and statistics
- ✅ Comprehensive error handling
- ✅ Request validation with Pydantic
- ✅ Logging and monitoring

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Copy Model Files

Place your trained PyTorch models in the server directory:

- `SignTransformerCtc_best.pt`
- `MediaPipeGRUCtc_best.pt`

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env if needed
```

### 4. Start Server

```bash
./start_server.sh
```

Or manually:

```bash
python app.py
```

Server will start on `http://0.0.0.0:8000`

## API Endpoints

### Health Check

```bash
GET /health
```

Returns server status and loaded models.

### Inference

```bash
POST /predict
Content-Type: application/json

{
  "keypoints": [[...], [...], ...],  // [T, 178]
  "model_type": "transformer"         // or "gru"
}
```

Returns CTC log probabilities and timing information.

### System Stats

```bash
GET /stats
```

Returns CPU/GPU usage and memory statistics.

## Testing

Run the test suite:

```bash
python test_server.py
```

## API Documentation

Once running, visit:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Configuration

Edit `.env` file to configure:

- Server host/port
- Model paths
- Device (cuda/cpu)
- Model hyperparameters

## Monitoring

Logs are written to:

- Console (stdout)
- `logs/server.log`

## Deployment

### Vast AI

1. Rent GPU instance
2. Upload files:

```bash
   scp -r server/ root@<vast-ip>:/app/
```

3. SSH and start:

```bash
   ssh root@<vast-ip>
   cd /app/server
   ./start_server.sh
```

### Performance Tips

- Use GPU for inference (DEVICE=cuda)
- Keep WORKERS=1 (GPU serialization)
- Monitor GPU memory usage
- Use quantized models for faster inference

## Troubleshooting

**Import Errors**

- Ensure all dependencies installed: `pip install -r requirements.txt`

**CUDA Out of Memory**

- Reduce MAX_SEQUENCE_LENGTH in config
- Use smaller model (GRU instead of Transformer)

**Model Not Found**

- Check model paths in `.env`
- Ensure .pt files are in correct directory

**Connection Refused**

- Check firewall allows port 8000
- Verify HOST=0.0.0.0 in .env

## License

Part of the Pansinayan project.

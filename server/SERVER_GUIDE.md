# Pansinayan Server Setup Guide

FastAPI inference server for Filipino Sign Language recognition.

---

## Table of Contents

- [Quick Start (Local Testing)](#quick-start-local-testing)
- [Vast.AI Deployment](#vastai-deployment)
- [API Endpoints](#api-endpoints)
- [Troubleshooting](#troubleshooting)
- [Monitoring](#monitoring)
- [Quick Reference](#quick-reference)

---

## Quick Start (Local Testing)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Place Model Files

Ensure these files are in the server directory:

- `SignTransformerCtc_best.pt` (~220MB)
- `MediaPipeGRUCtc_best.pt` (~30MB)

### 3. Configure Environment

Create a `.env` file with the following settings:

```env
HOST=0.0.0.0         # Required for Vast.AI
PORT=8000            # Default port
DEVICE=cpu           # Use CPU
CORS_ORIGINS=*       # Must be JSON array format
```

### 4. Start Server

```bash
./start_server.sh
```

Server starts on `http://0.0.0.0:8000`

---

## Vast.AI Deployment

### Prerequisites: Create Server Package

**When to do this:**

- If `pansinayan_server.tar.gz` doesn't exist
- If server code has been updated and needs to be repackaged

**Create the package:**

```bash
cd server
tar -czf ../pansinayan_server.tar.gz *
```

---

### Step 1: Rent Instance

1. Go to [vast.ai/console/instances](https://vast.ai/console/instances)
2. Filter: RTX 3060/3090, ≥8GB VRAM
3. Template: `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime`
4. Click "RENT"
5. **Save your instance IP:** `xxx.xxx.xxx.xxx`

### Step 2: Access Jupyter Notebook

1. In Vast.AI console, click "Jupyter" button on your instance
2. Wait for Jupyter to launch
3. Open a terminal: **New → Terminal**

### Step 3: Download & Setup

Run these commands in the Jupyter terminal:

```bash
# Install gdown
pip install gdown

# Create workspace
mkdir -p pansinayan
cd pansinayan

# Download server package from Google Drive
gdown 1DQz7cjMXBQExDbVJpI7KIKZLuLbO2MG6

# Extract
tar -xzf pansinayan_server.tar.gz

# Check Python version (Python 3.12 is default on Vast/Ubuntu 24.04)
python3 --version
# You will see: Python 3.12.x (Torch 2.1.0 is NOT compatible)

# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install Miniconda (press ENTER, accept defaults)
bash Miniconda3-latest-Linux-x86_64.sh

# Reload environment so conda works
source ~/.bashrc

# Create Python 3.11 environment (Torch 2.1.0 requires <= 3.11)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n py311 python=3.11 -y
conda activate py311

# Verify Python version
python --version   # Should now show: Python 3.11.x

cd pansinayan

# Install dependencies
pip install --no-cache-dir -r requirements.txt
pip install "numpy<2"

# Configure - Create .env file with required settings
cat > .env << 'EOF'
HOST=0.0.0.0
PORT=8000
DEVICE=cpu
CORS_ORIGINS=*
EOF

# Start server
chmod +x start_server.sh
sed -i 's/\r$//' start_server.sh
./start_server.sh
```

### Step 4: Keep Server Running (Optional)

**Why?** If you close the Jupyter terminal, the server stops. Use this step to keep the server running in the background.

**Option A: Foreground (Testing)**

- Just run `./start_server.sh` from Step 3
- Server stops when you close the terminal

**Option B: Background (Production)**

1. Install screen:

```bash
apt-get update && apt-get install -y screen
```

2. Start server in background:

```bash
screen -dmS pansinayan bash -c 'cd pansinayan && ./start_server.sh'
```

3. Server now runs in background - you can close terminal!

4. To view logs later:

```bash
screen -r pansinayan
# Press Ctrl+A then D to detach (keeps server running)
```

### Step 5: Expose Port via Tunnels

**After server is running, expose it publicly:**

Run this command in a new terminal (or in a screen session):

```bash
cloudflared tunnel --url http://localhost:8000 --protocol http2
```

This will create a public tunnel URL (e.g., `https://xxxx.trycloudflare.com`). **Copy this URL** - this is your public server URL to use instead of the IP address!

**Note:** Keep this terminal/session running to maintain the tunnel. If you close it, the tunnel will stop.

### Step 6: Test Server

Visit these URLs in your browser:

1. **Health Check:**

   ```
   https://xxxx.trycloudflare.com/health
   ```

   Should return:

   ```json
   {
     "status": "healthy",
     "models_loaded": ["transformer", "gru"],
     "device": "cuda"
   }
   ```

2. **Interactive API Docs (Swagger UI):**

   ```
   https://xxxx.trycloudflare.com/docs
   ```

   This shows all available endpoints and lets you test them directly!

3. **Root Endpoint:**
   ```
   https://xxxx.trycloudflare.com/
   ```
   Shows server info and available endpoints

**✅ If you see these pages, your server is working correctly!**

**Note:** Use the tunnel URL (`https://xxxx.trycloudflare.com`) in your Android app instead of the IP address!

---

## API Endpoints

### Health Check

```http
GET /health
```

### Inference

```http
POST /predict
Content-Type: application/json

{
  "keypoints": [[...], [...], ...],  // [T, 178] - Variable length (1-300 frames)
                                      // T is determined by sign-aligned window from activity-based detection
  "model_type": "transformer"         // or "gru"
}
```

### Statistics

```http
GET /stats
```

**Interactive API Documentation:** `http://<INSTANCE_IP>:8000/docs`

---

## Troubleshooting

### Connection Refused

```bash
# Check if server is running
ps aux | grep uvicorn

# Verify .env configuration
grep HOST .env  # Should be: 0.0.0.0
```

### CUDA Out of Memory

```bash
# Edit .env
nano .env
# Set: MAX_SEQUENCE_LENGTH=200

# Restart server
pkill -f uvicorn && ./start_server.sh
```

### Model Not Found

```bash
# Verify model files exist
ls -lh *.pt
```

### Slow Inference

```bash
# Check GPU usage
nvidia-smi

# Try GRU model (faster than Transformer)
```

---

## Monitoring

```bash
# View logs
tail -f logs/server.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check server statistics
curl http://<INSTANCE_IP>:8000/stats
```

---

## Quick Reference

```bash
# Start server
cd server && ./start_server.sh

# Restart server
pkill -f uvicorn && ./start_server.sh

# Check server status
curl http://localhost:8000/health
```

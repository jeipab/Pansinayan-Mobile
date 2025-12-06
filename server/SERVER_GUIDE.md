# Pansinayan Server Setup Guide

FastAPI inference server for Filipino Sign Language recognition.

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

```bash
cp .env.example .env
```

Key settings (create `.env` file):

```
HOST=0.0.0.0         # Required for Vast.AI
PORT=8000            # Default port
DEVICE=cuda          # Use GPU
CORS_ORIGINS=["*"]   # Must be JSON array format
```

### 4. Start Server

```bash
./start_server.sh
```

Server starts on `http://0.0.0.0:8000`

---

## Vast.AI Deployment

### Step 0: Create/Update Server Package (If Needed)

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
5. **Save your instance IP:** xxx.xxx.xxx.xxx

### Step 2: Access Jupyter Notebook

1. In Vast.AI console, click "Jupyter" button on your instance
2. Wait for Jupyter to launch
3. Open a terminal: **New → Terminal**

### Step 3: Download & Setup via Jupyter Terminal

Run these commands in Jupyter terminal:

```bash
# Install gdown
pip install gdown

# Create workspace
mkdir -p pansinayan
cd pansinayan

# Download server package from Google Drive
gdown <file id>

# Extract
tar -xzf pansinayan_server.tar.gz

# Remove archive to save space
rm pansinayan_server.tar.gz

cd server

# Setup virtual environment
python3 -m venv ../venv
source ../venv/bin/activate

# Install dependencies
pip install --no-cache-dir -r requirements.txt

# Configure - Create .env file with required settings
cat > .env << 'EOF'
HOST=0.0.0.0
PORT=8000
DEVICE=cuda
CORS_ORIGINS=["*"]
EOF

# Start server
chmod +x start_server.sh
./start_server.sh
```

### Step 4: Keep Server Running (Background) - Optional

**Why?** If you close the Jupyter terminal in Step 3, the server stops. Use this step to keep the server running even after closing/disconnecting.

**Choose one:**

**Option A:** Server stops when terminal closes (simpler, good for testing)

- Just run `./start_server.sh` from Step 3
- Server stops when you close the terminal

**Option B:** Server keeps running in background (recommended for production)

- Install screen:

```bash
apt-get update && apt-get install -y screen
```

- Start server in background:

```bash
screen -dmS pansinayan bash -c 'cd pansinayan/server && source ../venv/bin/activate && ./start_server.sh'
```

- Server now runs in background - you can close terminal!
- To view logs later:

```bash
screen -r pansinayan
# Press Ctrl+A then D to detach (keeps server running)
```

### Step 5: Expose Port 8000 via Tunnels

**After server is running, expose it publicly:**

1. In Vast.AI console, go to your instance
2. Click **"Tunnels (Open New Ports)"** in the sidebar
3. In "Enter target URL" field, type: `http://localhost:8000`
4. Click **"+ Create New Tunnel"**
5. Wait for tunnel to be created
6. **Copy the Tunnel URL** (e.g., `https://xxxx.trycloudflare.com`)
7. This is your public server URL - use this instead of the IP address!

### Step 6: Test Server

**Visit these URLs in your browser:**

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

```bash
# Health check
GET /health

# Inference
POST /predict
{
  "keypoints": [[...], [...], ...],  // [T, 178] - Variable length (1-300 frames)
                                      // T is determined by sign-aligned window from activity-based detection
  "model_type": "transformer"         // or "gru"
}

# Stats
GET /stats
```

Interactive docs: `http://<INSTANCE_IP>:8000/docs`

---

## Troubleshooting

**Connection Refused**

```bash
# Check server running
ps aux | grep uvicorn

# Verify .env
grep HOST .env  # Should be: 0.0.0.0
```

**CUDA Out of Memory**

```bash
# Edit .env
nano .env
# Set: MAX_SEQUENCE_LENGTH=200

# Restart server
pkill -f uvicorn && ./start_server.sh
```

**Model Not Found**

```bash
# Verify files
ls -lh *.pt
```

**Slow Inference**

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

# Monitor GPU
watch -n 1 nvidia-smi

# Check stats
curl http://<INSTANCE_IP>:8000/stats
```

---

## Quick Reference

```bash
# Activate venv
source pansinayan/venv/bin/activate

# Start server
cd pansinayan/server && ./start_server.sh

# Restart
pkill -f uvicorn && ./start_server.sh
```

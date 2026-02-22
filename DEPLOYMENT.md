# Deployment Guide: Aphrodite Backend Options

This guide provides detailed deployment scenarios for the Aphrodite inference engine backend to work with the SillyTavern frontend.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Local Development](#local-development)
3. [Cloud VM Deployment](#cloud-vm-deployment)
4. [Container Orchestration](#container-orchestration)
5. [Serverless GPU](#serverless-gpu)
6. [Reverse Proxy Setup](#reverse-proxy-setup)
7. [SSL/TLS Configuration](#ssltls-configuration)
8. [Scaling Strategies](#scaling-strategies)
9. [Cost Optimization](#cost-optimization)
10. [Monitoring & Observability](#monitoring--observability)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Layer                            │
│  ┌────────────────────┐                                      │
│  │  GitHub Pages      │  or  │ Custom CDN │ or │ Vercel │   │
│  │  (SillyTavern UI)  │                                      │
│  └──────────┬─────────┘                                      │
│             │ HTTPS                                          │
└─────────────┼──────────────────────────────────────────────-─┘
              │
              │ API Requests
              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Load Balancer / Reverse Proxy              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │   nginx    │  │  traefik   │  │   Caddy    │            │
│  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘            │
│         │                │                │                  │
│         └────────────────┴────────────────┘                  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend Layer                             │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │  Aphrodite       │  │  Aphrodite       │  (optional)     │
│  │  Instance 1      │  │  Instance 2      │                 │
│  │  Port 2242       │  │  Port 2243       │                 │
│  └──────────────────┘  └──────────────────┘                │
│         │                       │                            │
│         └───────────┬───────────┘                            │
│                     ▼                                        │
│              ┌──────────────┐                                │
│              │  GPU(s)      │                                │
│              │  VRAM Pool   │                                │
│              └──────────────┘                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Local Development

### Quick Setup

```bash
# Install Aphrodite
pip install aphrodite-engine --extra-index-url https://downloads.pygmalion.chat/whl

# Start with a small model for testing
aphrodite run Qwen/Qwen3-0.6B
```

### With CORS for Local Frontend

```bash
aphrodite run Qwen/Qwen3-0.6B \
  --cors-allowed-origins "http://localhost:8000" \
  --cors-allowed-origins "http://127.0.0.1:8000"
```

### For Local Development with Tunneling

Using ngrok:
```bash
# Start Aphrodite
aphrodite run Qwen/Qwen3-0.6B

# In another terminal, create tunnel
ngrok http 2242

# Use the ngrok URL in your frontend
```

Using Cloudflare Tunnel:
```bash
# Install cloudflared
# Start Aphrodite
aphrodite run Qwen/Qwen3-0.6B

# Create tunnel
cloudflared tunnel --url http://localhost:2242
```

---

## 2. Cloud VM Deployment

### AWS EC2

#### Instance Selection

| Model Size | Instance Type | VRAM | Cost/Hour | Recommended For |
|------------|---------------|------|-----------|-----------------|
| < 7B       | g4dn.xlarge   | 16GB | ~$0.53    | Development     |
| 7B-13B     | g4dn.2xlarge  | 16GB | ~$0.75    | Small production|
| 13B-30B    | g5.xlarge     | 24GB | ~$1.01    | Medium prod     |
| 30B-70B    | g5.12xlarge   | 96GB | ~$5.67    | Large models    |

#### Setup Script

```bash
#!/bin/bash
# setup-aphrodite-aws.sh

# Update system
sudo apt-get update
sudo apt-get install -y python3-pip nvidia-driver-535 nvidia-cuda-toolkit

# Install Aphrodite
pip3 install aphrodite-engine --extra-index-url https://downloads.pygmalion.chat/whl

# Create systemd service
sudo tee /etc/systemd/system/aphrodite.service > /dev/null <<EOF
[Unit]
Description=Aphrodite Inference Engine
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu
Environment="PATH=/home/ubuntu/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=/home/ubuntu/.local/bin/aphrodite run mistralai/Mistral-7B-Instruct-v0.3 \
  --host 0.0.0.0 \
  --port 2242 \
  --cors-allowed-origins https://org-echo-opencog.github.io
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable aphrodite
sudo systemctl start aphrodite

# Check status
sudo systemctl status aphrodite
```

#### Security Group Configuration

```json
{
  "IpPermissions": [
    {
      "IpProtocol": "tcp",
      "FromPort": 443,
      "ToPort": 443,
      "IpRanges": [{"CidrIp": "0.0.0.0/0"}]
    },
    {
      "IpProtocol": "tcp",
      "FromPort": 22,
      "ToPort": 22,
      "IpRanges": [{"CidrIp": "YOUR_IP/32"}]
    }
  ]
}
```

### Google Cloud Platform

#### Create Instance

```bash
gcloud compute instances create aphrodite-server \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE \
  --metadata startup-script='#!/bin/bash
    apt-get update
    apt-get install -y python3-pip
    pip3 install aphrodite-engine --extra-index-url https://downloads.pygmalion.chat/whl
  '
```

### Azure

```bash
az vm create \
  --resource-group aphrodite-rg \
  --name aphrodite-vm \
  --image UbuntuLTS \
  --size Standard_NC6s_v3 \
  --admin-username azureuser \
  --generate-ssh-keys

# Install GPU drivers
az vm extension set \
  --resource-group aphrodite-rg \
  --vm-name aphrodite-vm \
  --name NvidiaGpuDriverLinux \
  --publisher Microsoft.HpcCompute \
  --version 1.6
```

---

## 3. Container Orchestration

### Docker Compose

```yaml
version: '3.8'

services:
  aphrodite:
    image: alpindale/aphrodite-openai:latest
    ports:
      - "2242:2242"
    environment:
      - APHRODITE_CORS_ORIGINS=https://org-echo-opencog.github.io
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
    volumes:
      - hf-cache:/root/.cache/huggingface
    command: >
      aphrodite run mistralai/Mistral-7B-Instruct-v0.3
      --host 0.0.0.0
      --port 2242
      --max-model-len 4096
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - aphrodite
    restart: unless-stopped

volumes:
  hf-cache:
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aphrodite-deployment
spec:
  replicas: 2
  selector:
    matchLabels:
      app: aphrodite
  template:
    metadata:
      labels:
        app: aphrodite
    spec:
      containers:
      - name: aphrodite
        image: alpindale/aphrodite-openai:latest
        command: ["aphrodite", "run", "mistralai/Mistral-7B-Instruct-v0.3"]
        args:
          - "--host"
          - "0.0.0.0"
          - "--port"
          - "2242"
        ports:
        - containerPort: 2242
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
        env:
        - name: APHRODITE_CORS_ORIGINS
          value: "https://org-echo-opencog.github.io"
        volumeMounts:
        - name: hf-cache
          mountPath: /root/.cache/huggingface
      volumes:
      - name: hf-cache
        persistentVolumeClaim:
          claimName: hf-cache-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: aphrodite-service
spec:
  selector:
    app: aphrodite
  ports:
  - protocol: TCP
    port: 80
    targetPort: 2242
  type: LoadBalancer
```

---

## 4. Serverless GPU

### Modal.com

```python
import modal

stub = modal.Stub("aphrodite-engine")

image = modal.Image.debian_slim().pip_install(
    "aphrodite-engine",
    extra_index_url="https://downloads.pygmalion.chat/whl"
)

@stub.function(
    gpu="A100",
    image=image,
    timeout=3600,
    container_idle_timeout=300,
)
@modal.web_endpoint(method="POST")
def generate(item: dict):
    import subprocess
    # Ephemeral endpoint - start on demand
    subprocess.run([
        "aphrodite", "run", "mistralai/Mistral-7B-Instruct-v0.3",
        "--host", "0.0.0.0",
        "--port", "8000"
    ])
```

### RunPod

```python
# runpod_handler.py
import runpod
import subprocess
import os

def handler(event):
    model = event["input"].get("model", "Qwen/Qwen3-0.6B")
    
    # Start Aphrodite
    process = subprocess.Popen([
        "aphrodite", "run", model,
        "--host", "0.0.0.0",
        "--port", "8000"
    ])
    
    return {"status": "running", "endpoint": "http://0.0.0.0:8000/v1"}

runpod.serverless.start({"handler": handler})
```

---

## 5. Reverse Proxy Setup

### Nginx Configuration

```nginx
# /etc/nginx/sites-available/aphrodite

upstream aphrodite_backend {
    server localhost:2242;
    # For multiple instances:
    # server localhost:2243;
    # server localhost:2244;
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # CORS headers
    add_header 'Access-Control-Allow-Origin' 'https://org-echo-opencog.github.io' always;
    add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS' always;
    add_header 'Access-Control-Allow-Headers' 'Content-Type, Authorization' always;
    add_header 'Access-Control-Max-Age' 86400 always;

    # Handle preflight requests
    if ($request_method = 'OPTIONS') {
        return 204;
    }

    location / {
        proxy_pass http://aphrodite_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Streaming support
        proxy_buffering off;
        proxy_cache off;
        
        # Timeouts
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

### Traefik Configuration

```yaml
# docker-compose.yml with Traefik
version: '3.8'

services:
  traefik:
    image: traefik:v2.10
    command:
      - "--api.insecure=true"
      - "--providers.docker=true"
      - "--entrypoints.web.address=:80"
      - "--entrypoints.websecure.address=:443"
      - "--certificatesresolvers.myresolver.acme.email=you@domain.com"
      - "--certificatesresolvers.myresolver.acme.storage=/letsencrypt/acme.json"
      - "--certificatesresolvers.myresolver.acme.httpchallenge.entrypoint=web"
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - "/var/run/docker.sock:/var/run/docker.sock:ro"
      - "./letsencrypt:/letsencrypt"

  aphrodite:
    image: alpindale/aphrodite-openai:latest
    command: aphrodite run mistralai/Mistral-7B-Instruct-v0.3
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.aphrodite.rule=Host(`api.yourdomain.com`)"
      - "traefik.http.routers.aphrodite.entrypoints=websecure"
      - "traefik.http.routers.aphrodite.tls.certresolver=myresolver"
      - "traefik.http.middlewares.aphrodite-cors.headers.accesscontrolalloworiginlist=https://org-echo-opencog.github.io"
      - "traefik.http.middlewares.aphrodite-cors.headers.accesscontrolallowmethods=GET,POST,OPTIONS"
      - "traefik.http.routers.aphrodite.middlewares=aphrodite-cors"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

---

## 6. SSL/TLS Configuration

### Let's Encrypt with Certbot

```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d api.yourdomain.com

# Auto-renewal is set up automatically
# Test renewal:
sudo certbot renew --dry-run
```

### Self-Signed Certificate (Development)

```bash
# Generate self-signed certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/ssl/private/aphrodite-selfsigned.key \
  -out /etc/ssl/certs/aphrodite-selfsigned.crt

# Use in Aphrodite
aphrodite run model-name \
  --ssl-keyfile /etc/ssl/private/aphrodite-selfsigned.key \
  --ssl-certfile /etc/ssl/certs/aphrodite-selfsigned.crt
```

---

## 7. Scaling Strategies

### Horizontal Scaling

Run multiple Aphrodite instances and load balance:

```bash
# Instance 1
aphrodite run model-name --port 2242

# Instance 2
aphrodite run model-name --port 2243

# Instance 3
aphrodite run model-name --port 2244
```

Configure nginx upstream:
```nginx
upstream aphrodite_backend {
    least_conn;  # Load balancing method
    server localhost:2242;
    server localhost:2243;
    server localhost:2244;
}
```

### Vertical Scaling

Use larger GPU instances:
```bash
# Multi-GPU inference
aphrodite run model-name --tensor-parallel-size 2

# With pipeline parallelism
aphrodite run model-name \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 2
```

---

## 8. Cost Optimization

### Spot Instances (AWS)

```bash
# Use spot instances for 70% cost savings
aws ec2 request-spot-instances \
  --spot-price "0.50" \
  --instance-count 1 \
  --type "persistent" \
  --launch-specification file://specification.json
```

### Serverless with Auto-scaling

Use Modal/RunPod that automatically scale to zero when not in use:
- Pay only for active inference time
- No idle costs
- Automatic scaling based on demand

### Model Quantization

Reduce VRAM requirements:
```bash
# Use 4-bit quantization
aphrodite run model-name --quantization awq

# Use 8-bit quantization
aphrodite run model-name --quantization bitsandbytes
```

---

## 9. Monitoring & Observability

### Prometheus & Grafana

Aphrodite exposes Prometheus metrics at `/metrics`:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'aphrodite'
    static_configs:
      - targets: ['localhost:2242']
```

### Logging

```bash
# Structured logging
aphrodite run model-name --log-level INFO

# Log to file
aphrodite run model-name 2>&1 | tee aphrodite.log
```

### Health Checks

```bash
# Add to monitoring
curl http://localhost:2242/health

# In Docker Compose
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:2242/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

---

## Summary: Recommended Deployments

| Use Case | Frontend | Backend | Estimated Cost |
|----------|----------|---------|----------------|
| **Development** | Local | Local | Free |
| **Demo/Testing** | GitHub Pages | Ngrok/Cloudflare Tunnel | Free-$5/mo |
| **Small Team** | GitHub Pages | Cloud VM (g4dn.xlarge) | $350-400/mo |
| **Production** | CDN | Load-balanced VMs | $1000-2000/mo |
| **Cost-Optimized** | GitHub Pages | Serverless GPU | $50-500/mo |
| **High-Traffic** | CDN | Kubernetes Cluster | $2000-10000/mo |

Choose based on your requirements for availability, performance, and budget.

---

For questions or issues, please refer to the [main integration guide](frontend/APHRODITE_INTEGRATION.md) or open an issue on GitHub.

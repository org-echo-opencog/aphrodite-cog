# Quick Start Guide: SillyTavern Frontend + Aphrodite Backend

This guide will help you quickly set up and run the integrated SillyTavern UI with an Aphrodite backend.

## Prerequisites

- **Node.js** 18 or higher
- **Python** 3.8 or higher
- **GPU** (optional but recommended for optimal performance)

## Option 1: Local Development (Fastest Setup)

### Step 1: Start the Aphrodite Backend

First, install Aphrodite:

```bash
pip install aphrodite-engine --extra-index-url https://downloads.pygmalion.chat/whl
```

Launch the backend with a model:

```bash
# For a small model (good for testing):
aphrodite run Qwen/Qwen3-0.6B

# For a better quality model (requires more VRAM):
# aphrodite run mistralai/Mistral-7B-Instruct-v0.3
```

The backend will start on `http://localhost:2242`.

**Verify it's running:**
```bash
curl http://localhost:2242/v1/models
```

### Step 2: Start the Frontend

In a new terminal:

```bash
cd frontend
npm install
npm start
```

The frontend will open automatically in your browser at `http://localhost:8000`.

### Step 3: Configure the API Connection

1. In SillyTavern, click on the **plug icon** (API Connections)
2. Select **Chat Completion Source**: `OpenAI`
3. Set **API URL**: `http://localhost:2242/v1`
4. Click **Connect**
5. The model should auto-detect

You're ready to chat! 🎉

## Option 2: Remote Backend Setup

If you have deployed Aphrodite on a remote server:

### Frontend (GitHub Pages)

The frontend is automatically deployed to GitHub Pages when you push to the main branch:

```
https://org-echo-opencog.github.io/aphrodite-cog/
```

### Backend Configuration

1. Deploy Aphrodite on your server
2. Start with CORS enabled:

```bash
aphrodite run <model-name> --host 0.0.0.0 --port 443 \
  --cors-allowed-origins https://org-echo-opencog.github.io
```

3. In the GitHub Pages UI, configure:
   - **API URL**: `https://your-server.com/v1`
   - **API Key**: (if you enabled authentication)

## Option 3: Docker Deployment

### Using Docker Compose

Create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  aphrodite-backend:
    image: alpindale/aphrodite-engine:latest
    ports:
      - "2242:2242"
    environment:
      - APHRODITE_CORS_ORIGINS=http://localhost:8000
    command: aphrodite run Qwen/Qwen3-0.6B --host 0.0.0.0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  sillytavern-frontend:
    build: ./frontend
    ports:
      - "8000:8000"
    depends_on:
      - aphrodite-backend
    environment:
      - APHRODITE_API_URL=http://aphrodite-backend:2242/v1
```

Start both services:

```bash
docker-compose up
```

Access the UI at `http://localhost:8000`.

## Troubleshooting

### Backend Not Connecting

**Problem**: "Failed to connect to API"

**Solutions**:
1. Verify Aphrodite is running: `curl http://localhost:2242/v1/models`
2. Check CORS settings - ensure origins are allowed
3. Verify the API URL includes `/v1` at the end

### Model Not Loading

**Problem**: Model fails to load or is too slow

**Solutions**:
1. Use a smaller model for testing (e.g., `Qwen/Qwen3-0.6B`)
2. Check GPU memory: `nvidia-smi`
3. Enable CPU mode if no GPU: `aphrodite run <model> --device cpu`

### CORS Errors

**Problem**: Browser shows CORS policy errors

**Solutions**:
1. Add your domain to allowed origins:
   ```bash
   aphrodite run <model> --cors-allowed-origins https://your-domain.com
   ```
2. For development, allow all origins:
   ```bash
   aphrodite run <model> --cors-allowed-origins "*"
   ```

### Port Already in Use

**Problem**: "Address already in use"

**Solutions**:
1. Change Aphrodite port:
   ```bash
   aphrodite run <model> --port 8080
   ```
2. Change SillyTavern port:
   ```bash
   cd frontend
   node server.js --port 3000
   ```

## Configuration Tips

### Optimizing for Performance

1. **Use GPU acceleration**: Ensure Aphrodite detects your GPU
2. **Adjust batch size**: Add `--max-num-batched-tokens 8192` for throughput
3. **Enable continuous batching**: Default in Aphrodite for multiple users

### Adjusting Sampling Parameters

In SillyTavern UI:
- **Temperature**: 0.7 (creativity vs consistency)
- **Top-p**: 0.9 (nucleus sampling)
- **Top-k**: 40 (limits token choices)
- **Max Tokens**: 512-2048 (response length)

### Multi-User Setup

For serving multiple users:

```bash
# Remove single-user mode for better concurrency
aphrodite run <model> --host 0.0.0.0 --port 2242
```

## Next Steps

- **Explore Characters**: Import or create characters in the UI
- **Configure Presets**: Fine-tune generation settings
- **Enable Extensions**: Explore SillyTavern extensions
- **Add Multiple Models**: Run multiple Aphrodite instances on different ports
- **Set up SSL**: Use HTTPS for production deployments

## Resources

- **SillyTavern Docs**: https://docs.sillytavern.app/
- **Aphrodite Docs**: https://aphrodite.pygmalion.chat/
- **Integration Guide**: [frontend/APHRODITE_INTEGRATION.md](frontend/APHRODITE_INTEGRATION.md)
- **GitHub Issues**: Report problems or request features

## Common Commands Reference

### Aphrodite

```bash
# Start with default settings
aphrodite run <model-name>

# Start with custom settings
aphrodite run <model-name> --host 0.0.0.0 --port 2242 --dtype auto

# List available models
aphrodite list-models

# Check version
aphrodite --version

# Get help
aphrodite run --help
```

### SillyTavern

```bash
# Start server
npm start

# Start with custom port
npm start -- --port 3000

# Start without CSRF protection (for development)
npm run start:no-csrf

# Update dependencies
npm install

# Run in debug mode
npm run debug
```

## Security Checklist for Production

- [ ] Enable HTTPS/SSL on both frontend and backend
- [ ] Configure API key authentication
- [ ] Restrict CORS to specific domains
- [ ] Use environment variables for secrets
- [ ] Enable rate limiting
- [ ] Set up monitoring and logging
- [ ] Use a reverse proxy (nginx/traefik)
- [ ] Regular security updates

---

**Need help?** Check the [full integration guide](frontend/APHRODITE_INTEGRATION.md) or open an issue!

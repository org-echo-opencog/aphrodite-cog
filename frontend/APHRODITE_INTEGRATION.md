# SillyTavern Frontend - Aphrodite Backend Integration

This directory contains the SillyTavern frontend UI integrated as native code for the Aphrodite cognitive inference engine.

## Overview

SillyTavern is a powerful LLM frontend that has been integrated into this repository to provide a user-friendly interface for the Aphrodite Engine backend. The integration enables:

- **Chat interface** for interacting with Aphrodite-powered language models
- **OpenAI-compatible API** connection
- **Character management** and persona creation
- **Advanced sampling controls** for fine-tuning model outputs
- **Multi-modal support** for text and image generation

## Architecture

```
┌─────────────────────┐         ┌──────────────────────┐
│  SillyTavern UI     │         │  Aphrodite Engine    │
│  (Frontend)         │◄───────►│  (Backend)           │
│  - GitHub Pages     │  HTTP   │  - OpenAI API        │
│  - Static HTML/JS   │         │  - Port 2242         │
└─────────────────────┘         └──────────────────────┘
```

## Local Development

### Prerequisites

- Node.js 18 or higher
- Aphrodite Engine running locally or remotely

### Setup

1. **Install dependencies:**
```bash
cd frontend
npm install
```

2. **Configure the backend connection:**

The frontend connects to Aphrodite via its OpenAI-compatible API endpoint. By default, Aphrodite runs on `http://localhost:2242`.

In the SillyTavern UI:
- Go to **API Connections** → **Chat Completion API**
- Select **OpenAI** as the API type
- Set the API URL to: `http://localhost:2242/v1`
- Add your API key (if authentication is enabled on Aphrodite)

3. **Start the development server:**
```bash
npm start
```

The UI will be available at `http://localhost:8000`.

## GitHub Pages Deployment

The SillyTavern UI can be deployed to GitHub Pages as a static frontend. The GitHub Actions workflow automatically builds and deploys the frontend.

### Deployment Configuration

The deployment is handled by `.github/workflows/deploy-frontend.yml`. The workflow:

1. Checks out the repository
2. Installs dependencies
3. Builds the static assets
4. Deploys to GitHub Pages

### Accessing the Deployed UI

After deployment, the UI will be available at:
```
https://<your-username>.github.io/<repository-name>/
```

For this repository:
```
https://org-echo-opencog.github.io/aphrodite-cog/
```

### Connecting to Remote Aphrodite Backend

When using the GitHub Pages deployment, you'll need to configure the frontend to connect to your Aphrodite backend:

1. **Deploy Aphrodite backend** (see Backend Deployment Options below)
2. **Configure CORS** on your Aphrodite instance to allow requests from your GitHub Pages domain
3. **In the SillyTavern UI**, set the API URL to your Aphrodite backend endpoint

## Backend Deployment Options

### Option 1: Cloud VM (Recommended for Production)

Deploy Aphrodite on a cloud VM (AWS EC2, Google Cloud Compute, Azure VM, etc.):

**Advantages:**
- Full control over hardware and GPU resources
- Best performance for large models
- Persistent deployment
- Custom domain support

**Steps:**
1. Provision a GPU-enabled VM instance
2. Install Aphrodite:
   ```bash
   pip install aphrodite-engine --extra-index-url https://downloads.pygmalion.chat/whl
   ```
3. Launch with SSL and CORS enabled:
   ```bash
   aphrodite run <model-name> --host 0.0.0.0 --port 443 \
     --ssl-keyfile /path/to/key.pem \
     --ssl-certfile /path/to/cert.pem \
     --cors-allowed-origins https://org-echo-opencog.github.io
   ```
4. Configure DNS to point to your VM
5. Use the URL in SillyTavern: `https://your-domain.com/v1`

**Cost Estimate:**
- AWS g4dn.xlarge: ~$0.50-1.00/hour (~$360-720/month)
- Google Cloud A100: ~$2.00-4.00/hour (~$1,440-2,880/month)
- Azure NC6s_v3: ~$0.90/hour (~$650/month)

### Option 2: Docker Container

Deploy Aphrodite in a containerized environment:

**Advantages:**
- Easy deployment and scaling
- Consistent environment
- Works on any platform

**Steps:**
1. Build the Docker image:
   ```bash
   cd /path/to/aphrodite-cog
   docker build -t aphrodite-engine .
   ```
2. Run the container:
   ```bash
   docker run -d --gpus all -p 2242:2242 \
     -e APHRODITE_CORS_ORIGINS=https://org-echo-opencog.github.io \
     aphrodite-engine aphrodite run <model-name>
   ```
3. Deploy to container orchestration (Kubernetes, ECS, etc.)

### Option 3: Serverless GPU (Modal, RunPod, etc.)

Use serverless GPU platforms for cost-effective deployment:

**Advantages:**
- Pay only for usage
- Auto-scaling
- No infrastructure management

**Platforms:**
- **Modal.com**: Serverless GPU with Python
- **RunPod**: On-demand GPU instances
- **Replicate**: ML model hosting
- **Hugging Face Inference Endpoints**: Managed inference

**Example (Modal):**
```python
import modal

stub = modal.Stub("aphrodite-engine")

@stub.function(
    gpu="A100",
    image=modal.Image.debian_slim().pip_install("aphrodite-engine")
)
def run_aphrodite():
    import subprocess
    subprocess.run([
        "aphrodite", "run", "model-name",
        "--host", "0.0.0.0",
        "--port", "8000"
    ])
```

**Cost Estimate:**
- Modal: ~$1.00-2.50/hour when running
- RunPod: ~$0.30-1.00/hour
- Replicate: Per-request pricing, ~$0.001-0.01 per request

### Option 4: Localhost Development

For development and testing:

**Advantages:**
- Free
- No deployment complexity
- Quick iteration

**Limitations:**
- Not accessible from GitHub Pages (CORS issues)
- Requires local setup
- No persistence

**Steps:**
1. Install Aphrodite locally:
   ```bash
   pip install aphrodite-engine --extra-index-url https://downloads.pygmalion.chat/whl
   ```
2. Run the engine:
   ```bash
   aphrodite run <model-name>
   ```
3. Access at `http://localhost:2242`

### Option 5: Tunneling Services (Development/Demo)

Use tunneling for temporary public access:

**Services:**
- **ngrok**: `ngrok http 2242`
- **Cloudflare Tunnel**: Free, persistent URLs
- **localhost.run**: `ssh -R 80:localhost:2242 localhost.run`

**Advantages:**
- Quick setup for demos
- No server provisioning
- Free tiers available

**Limitations:**
- Not suitable for production
- May have performance issues
- Temporary URLs (ngrok free tier)

## Recommended Deployment Scenarios

### Scenario 1: Personal Use / Development
- **Frontend**: GitHub Pages
- **Backend**: Localhost with tunneling (ngrok/Cloudflare)
- **Cost**: Free - $5/month
- **Effort**: Low

### Scenario 2: Small Team / Shared Access
- **Frontend**: GitHub Pages
- **Backend**: Small cloud VM (g4dn.xlarge or equivalent)
- **Cost**: $300-500/month
- **Effort**: Medium

### Scenario 3: Production / High Traffic
- **Frontend**: GitHub Pages or CDN
- **Backend**: Auto-scaling container cluster or serverless
- **Cost**: Variable based on usage ($500-5000/month)
- **Effort**: High

### Scenario 4: Cost-Optimized
- **Frontend**: GitHub Pages
- **Backend**: Serverless GPU (Modal/RunPod) with cold starts
- **Cost**: Pay-per-use (~$50-300/month depending on usage)
- **Effort**: Medium

## Configuration Files

- **aphrodite-config.yaml**: SillyTavern configuration optimized for Aphrodite
- **default/config.yaml**: Default SillyTavern configuration

## Security Considerations

1. **HTTPS Required**: Always use HTTPS for production deployments
2. **CORS Configuration**: Properly configure CORS on Aphrodite to only allow your frontend domain
3. **API Keys**: Use API key authentication on Aphrodite for production
4. **Rate Limiting**: Implement rate limiting on the backend
5. **Content Security Policy**: Configure CSP headers appropriately

## Troubleshooting

### Common Issues

1. **CORS Errors**:
   - Ensure Aphrodite is launched with `--cors-allowed-origins` flag
   - Check that the frontend domain matches the allowed origins

2. **Connection Refused**:
   - Verify Aphrodite is running: `curl http://localhost:2242/v1/models`
   - Check firewall rules allow the port
   - Ensure the API URL in SillyTavern is correct

3. **404 Not Found**:
   - Verify the API endpoint path includes `/v1`
   - Check Aphrodite logs for errors

4. **Authentication Errors**:
   - Ensure API key is correctly set in SillyTavern
   - Verify API key is valid on Aphrodite backend

## Development

### Building from Source

```bash
cd frontend
npm install
npm run build
```

### Running Tests

```bash
npm run lint
npm test
```

## License

SillyTavern is licensed under AGPL-3.0. See the LICENSE file for details.

This integration maintains compatibility with the Aphrodite Engine license.

## Support

- **SillyTavern Documentation**: https://docs.sillytavern.app/
- **Aphrodite Documentation**: https://aphrodite.pygmalion.chat/
- **GitHub Issues**: Report integration issues in this repository

## Contributing

Contributions to improve the integration are welcome! Please ensure:
- Changes maintain compatibility with both SillyTavern and Aphrodite
- Documentation is updated
- Testing is performed on both local and deployed environments

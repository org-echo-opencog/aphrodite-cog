# SillyTavern Frontend for Aphrodite Engine

This directory contains the SillyTavern UI integrated as the native frontend for the Aphrodite cognitive inference engine.

## What is This?

SillyTavern is a powerful LLM frontend interface that has been integrated into this repository to provide a complete, user-friendly solution for interacting with Aphrodite-powered language models.

## Quick Links

- 🚀 **[Quick Start Guide](../QUICKSTART.md)** - Get up and running in 10 minutes
- 📖 **[Integration Guide](./APHRODITE_INTEGRATION.md)** - Detailed integration documentation
- 🚢 **[Deployment Options](../DEPLOYMENT.md)** - Comprehensive deployment scenarios
- 💡 **[Recommendations](../DEPLOYMENT_RECOMMENDATIONS.md)** - Optimal deployment choices
- 📊 **[Integration Summary](../INTEGRATION_SUMMARY.md)** - Complete overview

## Quick Start

### Local Development

```bash
# 1. Start Aphrodite backend
pip install aphrodite-engine --extra-index-url https://downloads.pygmalion.chat/whl
aphrodite run Qwen/Qwen3-0.6B

# 2. Start frontend (in this directory)
npm install
npm start

# 3. Open http://localhost:8000
# 4. Configure API: http://localhost:2242/v1
```

### Verify Backend Connection

```bash
./configure-aphrodite.py --check --url http://localhost:2242/v1
```

## Configuration Files

- **`aphrodite-config.yaml`** - Server configuration optimized for Aphrodite
- **`aphrodite-config.js`** - Browser-side configuration helper
- **`configure-aphrodite.py`** - CLI tool for backend verification
- **`APHRODITE_INTEGRATION.md`** - Complete integration guide

## GitHub Pages Deployment

The frontend is automatically deployed to GitHub Pages when changes are pushed to the main branch:

**Live URL**: https://org-echo-opencog.github.io/aphrodite-cog/

### Deployment Workflow

The GitHub Actions workflow (`.github/workflows/deploy-frontend.yml`) automatically:
1. Builds the frontend on push to main
2. Prepares static files
3. Deploys to GitHub Pages

## Connecting to Backend

### Local Backend
```javascript
// In SillyTavern UI:
API Type: OpenAI
API URL: http://localhost:2242/v1
```

### Remote Backend
```javascript
// In SillyTavern UI:
API Type: OpenAI
API URL: https://your-backend-domain.com/v1
API Key: your-api-key (if required)
```

## Features

- ✅ **Character Management** - Create and manage AI characters
- ✅ **Chat Interface** - Rich conversation UI
- ✅ **Advanced Sampling** - Fine-tune generation parameters
- ✅ **OpenAI Compatible** - Works with Aphrodite's OpenAI API
- ✅ **Extensions Support** - Rich plugin ecosystem
- ✅ **Multi-modal** - Text and image support

## Architecture

```
┌─────────────────────┐         ┌──────────────────────┐
│  SillyTavern UI     │         │  Aphrodite Engine    │
│  (This Directory)   │◄───────►│  (Backend)           │
│                     │  HTTP   │                      │
│  - HTML/CSS/JS      │         │  - OpenAI API        │
│  - Static Files     │         │  - Port 2242         │
│  - Configuration    │         │  - Model Inference   │
└─────────────────────┘         └──────────────────────┘
```

## Directory Structure

```
frontend/
├── public/              # Static UI files (HTML, CSS, JS)
├── src/                 # Server source (for local mode)
├── default/             # Default content (characters, presets)
├── aphrodite-config.yaml       # Server configuration
├── aphrodite-config.js         # Browser configuration
├── configure-aphrodite.py      # CLI helper tool
├── APHRODITE_INTEGRATION.md    # Integration guide
├── package.json         # Node.js dependencies
└── server.js           # Local server (not needed for GitHub Pages)
```

## Configuration

### For Static Deployment (GitHub Pages)

Use `aphrodite-config.yaml` which includes:
- CORS enabled for all origins
- CSRF protection disabled (backend handles security)
- Static deployment optimizations

### For Local Server

Use the default SillyTavern configuration or customize `config.yaml`.

## Security Notes

⚠️ **Important**: This configuration is optimized for static GitHub Pages deployment where the backend (Aphrodite) handles all security, authentication, and authorization.

For production deployments:
- Enable HTTPS on the backend
- Configure proper CORS (not wildcard)
- Use API key authentication
- Implement rate limiting
- Follow the security checklist in `../DEPLOYMENT_RECOMMENDATIONS.md`

## Backend Deployment Options

Choose the deployment that fits your needs:

| Option | Cost/Month | Best For |
|--------|-----------|----------|
| **Local** | Free | Development, Testing |
| **Serverless GPU** | $50-300 | Variable traffic, Budget-conscious |
| **Cloud VM** | $380 | Team use, Consistent traffic |
| **Kubernetes** | $2,200+ | High traffic, Enterprise |

See [DEPLOYMENT_RECOMMENDATIONS.md](../DEPLOYMENT_RECOMMENDATIONS.md) for detailed guidance.

## Troubleshooting

### Connection Issues

1. **Verify backend is running:**
   ```bash
   curl http://localhost:2242/v1/models
   ```

2. **Check CORS settings:**
   Ensure Aphrodite was started with:
   ```bash
   aphrodite run <model> --cors-allowed-origins "http://localhost:8000"
   ```

3. **Use the helper script:**
   ```bash
   ./configure-aphrodite.py --check
   ```

### Common Errors

- **CORS Error**: Add your domain to backend's allowed origins
- **Connection Refused**: Check if backend is running and port is correct
- **404 Not Found**: Ensure API URL ends with `/v1`
- **Authentication Error**: Check if API key is required and configured

See [QUICKSTART.md](../QUICKSTART.md) for more troubleshooting tips.

## Updating SillyTavern

To update to a newer version of SillyTavern:

1. Backup custom configurations:
   ```bash
   cp aphrodite-config.yaml /tmp/
   cp APHRODITE_INTEGRATION.md /tmp/
   ```

2. Clone latest SillyTavern:
   ```bash
   git clone https://github.com/SillyTavern/SillyTavern.git /tmp/st-latest
   rm -rf /tmp/st-latest/.git
   ```

3. Replace and restore:
   ```bash
   rm -rf ../frontend
   mv /tmp/st-latest ../frontend
   mv /tmp/aphrodite-config.yaml ../frontend/
   mv /tmp/APHRODITE_INTEGRATION.md ../frontend/
   ```

## Development

### Running Locally

```bash
npm install
npm start
```

### Building for Production

```bash
npm run build
# Static files ready for deployment
```

### Linting

```bash
npm run lint
```

## Resources

### Documentation
- [SillyTavern Official Docs](https://docs.sillytavern.app/)
- [Aphrodite Engine Docs](https://aphrodite.pygmalion.chat/)
- [Integration Guide](./APHRODITE_INTEGRATION.md)

### Community
- [SillyTavern Discord](https://discord.gg/sillytavern)
- [SillyTavern Reddit](https://reddit.com/r/SillyTavernAI)
- [GitHub Issues](https://github.com/org-echo-opencog/aphrodite-cog/issues)

## License

SillyTavern is licensed under AGPL-3.0. See [LICENSE](./LICENSE) for details.

This integration maintains compatibility with both SillyTavern and Aphrodite Engine licenses.

## Contributing

Contributions to improve the integration are welcome! Please:
1. Maintain compatibility with both SillyTavern and Aphrodite
2. Update documentation as needed
3. Test on both local and deployed environments
4. Follow the existing code style

## Support

- 📖 **Documentation**: Start with [QUICKSTART.md](../QUICKSTART.md)
- 🐛 **Issues**: Report in [GitHub Issues](https://github.com/org-echo-opencog/aphrodite-cog/issues)
- 💬 **Questions**: Check the integration docs first
- 🔧 **Help**: Use the `configure-aphrodite.py` tool for diagnostics

---

**Ready to start?** Follow the [Quick Start Guide](../QUICKSTART.md)! 🚀

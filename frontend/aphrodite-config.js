/**
 * Aphrodite Backend Configuration Helper
 * 
 * This script helps configure the SillyTavern frontend to connect to an Aphrodite backend.
 * It provides default settings optimized for Aphrodite's OpenAI-compatible API.
 */

// Default Aphrodite configuration
const APHRODITE_CONFIG = {
    // API Configuration
    api: {
        type: 'openai',
        endpoint: 'http://localhost:2242/v1',  // Default Aphrodite endpoint
        model: 'auto',  // Auto-detect from backend
    },
    
    // Default sampling parameters optimized for Aphrodite
    sampling: {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        max_tokens: 512,
        repetition_penalty: 1.15,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
    },
    
    // Connection settings
    connection: {
        cors: true,
        streaming: true,
        timeout: 120000,  // 2 minutes
    },
    
    // Feature flags for Aphrodite compatibility
    features: {
        supports_functions: true,
        supports_vision: true,  // If using multimodal models
        supports_logprobs: true,
        supports_stop_sequences: true,
    }
};

// Helper function to generate OpenAI API connection string
function getAphroditeConnectionString(host = 'localhost', port = 2242, ssl = false) {
    const protocol = ssl ? 'https' : 'http';
    
    // Port validation: warn if SSL is enabled but port is not standard HTTPS
    if (ssl && port !== 443) {
        console.warn(
            `Warning: SSL is enabled but port is ${port}. ` +
            'Consider using port 443 for HTTPS or ensure your server is configured correctly.'
        );
    }
    
    return `${protocol}://${host}:${port}/v1`;
}

// Export configuration
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        APHRODITE_CONFIG,
        getAphroditeConnectionString,
    };
}

// For browser environments
if (typeof window !== 'undefined') {
    window.APHRODITE_CONFIG = APHRODITE_CONFIG;
    window.getAphroditeConnectionString = getAphroditeConnectionString;
}

/**
 * Usage Examples:
 * 
 * 1. Local development:
 *    const apiUrl = getAphroditeConnectionString('localhost', 2242, false);
 *    // Result: 'http://localhost:2242/v1'
 * 
 * 2. Remote server with SSL:
 *    const apiUrl = getAphroditeConnectionString('api.example.com', 443, true);
 *    // Result: 'https://api.example.com:443/v1'
 * 
 * 3. Cloud deployment:
 *    const apiUrl = getAphroditeConnectionString('my-aphrodite.cloud.com', 443, true);
 *    // Result: 'https://my-aphrodite.cloud.com:443/v1'
 */

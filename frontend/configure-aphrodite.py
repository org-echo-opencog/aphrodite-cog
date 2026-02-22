#!/usr/bin/env python3
"""
Aphrodite Backend Configuration Helper

This script helps configure and verify the connection between
SillyTavern frontend and Aphrodite backend.
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
from typing import Dict, Any


def check_backend(api_url: str, timeout: int = 5) -> Dict[str, Any]:
    """
    Check if Aphrodite backend is accessible and retrieve models.
    
    Args:
        api_url: Base URL of the Aphrodite API
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary with status, message, and models (always present)
    """
    models_endpoint = f"{api_url}/models"
    
    try:
        req = urllib.request.Request(
            models_endpoint,
            headers={'Content-Type': 'application/json'}
        )
        
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode())
            
            return {
                'status': 'connected',
                'models': data.get('data', []),
                'message': 'Successfully connected to Aphrodite backend'
            }
            
    except urllib.error.HTTPError as e:
        return {
            'status': 'error',
            'models': [],
            'message': f'HTTP Error {e.code}: {e.reason}'
        }
    except urllib.error.URLError as e:
        return {
            'status': 'error',
            'models': [],
            'message': f'Connection failed: {e.reason}'
        }
    except Exception as e:
        return {
            'status': 'error',
            'models': [],
            'message': f'Unexpected error: {str(e)}'
        }


def generate_config(api_url: str, api_key: str = None) -> Dict[str, Any]:
    """
    Generate SillyTavern configuration for Aphrodite.
    
    Args:
        api_url: Base URL of the Aphrodite API
        api_key: Optional API key for authentication
        
    Returns:
        Configuration dictionary
    """
    config = {
        'api_type': 'openai',
        'api_url': api_url,
        'api_key': api_key or '',
        'model': 'auto',
        'streaming': True,
        'temperature': 0.7,
        'top_p': 0.9,
        'top_k': 40,
        'max_tokens': 512,
        'frequency_penalty': 0.0,
        'presence_penalty': 0.0,
        'repetition_penalty': 1.15,
    }
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Configure and verify Aphrodite backend connection'
    )
    parser.add_argument(
        '--url',
        default='http://localhost:2242/v1',
        help='Aphrodite API URL (default: http://localhost:2242/v1)'
    )
    parser.add_argument(
        '--api-key',
        help='API key for authentication (optional)'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check backend connectivity'
    )
    parser.add_argument(
        '--generate-config',
        action='store_true',
        help='Generate SillyTavern configuration'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=5,
        help='Connection timeout in seconds (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Default action: check connectivity
    if not args.generate_config:
        args.check = True
    
    # Check backend
    if args.check:
        print(f"🔍 Checking Aphrodite backend at {args.url}...")
        result = check_backend(args.url, args.timeout)
        
        if result['status'] == 'connected':
            print(f"✅ {result['message']}")
            
            if result['models']:
                print(f"\n📦 Available models:")
                for model in result['models']:
                    model_id = model.get('id', 'unknown')
                    print(f"  - {model_id}")
            else:
                print("\n⚠️  No models found")
                
            if args.generate_config:
                print("\n📝 Generating configuration...")
                config = generate_config(args.url, args.api_key)
                print(json.dumps(config, indent=2))
                
        else:
            print(f"❌ {result['message']}")
            print("\n💡 Troubleshooting tips:")
            print("  1. Verify Aphrodite is running: curl http://localhost:2242/v1/models")
            print("  2. Check if the URL is correct")
            print("  3. Ensure firewall allows the connection")
            print("  4. Try increasing --timeout value")
            sys.exit(1)
    
    # Generate config only
    elif args.generate_config:
        print("📝 Generating SillyTavern configuration...")
        config = generate_config(args.url, args.api_key)
        print(json.dumps(config, indent=2))


if __name__ == '__main__':
    main()

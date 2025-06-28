#!/usr/bin/env python3
"""
Standalone script to start the Copilot Strategy API server.

This script starts the FastAPI server that provides REST endpoints
for GitHub Copilot to access historical strategy performance data.
"""

import sys
import os
import uvicorn

# Add the project root to the path
project_root = "/home/runner/work/Agentic-Trader/Agentic-Trader"
sys.path.insert(0, project_root)

from copilot_integration.api_endpoints import app

if __name__ == "__main__":
    print("🚀 Starting Copilot Strategy Context API...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    print("❤️ Health check at: http://localhost:8000/api/health")
    print("\nPress Ctrl+C to stop the server")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
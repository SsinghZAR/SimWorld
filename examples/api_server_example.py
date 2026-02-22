"""Example script to start the SimWorld API server."""
import sys
from pathlib import Path

# Add the parent directory to Python path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import uvicorn
from simworld.api.server import app

if __name__ == "__main__":
    print("Starting SimWorld API Server...")
    print("API Documentation: http://localhost:8000/docs")
    print("API Health Check: http://localhost:8000/health")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

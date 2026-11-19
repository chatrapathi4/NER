"""
main.py — Application Entry Point

Launches the FastAPI server with uvicorn.

Usage:
    python main.py
    python main.py --port 8080
    python main.py --host 127.0.0.1 --port 5000
"""

import sys
import logging
import argparse
from pathlib import Path

# Ensure project root is in Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Medical NER API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    import uvicorn

    print("\n" + "=" * 60)
    print("  Medical NER — API Server")
    print("=" * 60)
    print(f"\n  Server:    http://{args.host}:{args.port}")
    print(f"  Frontend:  http://localhost:{args.port}")
    print(f"  API Docs:  http://localhost:{args.port}/docs")
    print(f"  Health:    http://localhost:{args.port}/health")
    print("\n" + "=" * 60 + "\n")

    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()

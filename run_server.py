#!/usr/bin/env python3
"""
IDP Kit — Web Server Entry Point

Start the IDP Kit web server with:
    python run_server.py
    python run_server.py --port 8000 --reload

Or use the CLI:
    python run_idpkit.py serve --port 8000
"""

import os
import sys


def main():
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed. Run: pip install uvicorn[standard]")
        sys.exit(1)

    host = os.getenv("IDP_HOST", "0.0.0.0")
    port = int(os.getenv("IDP_PORT", "8000"))
    reload = "--reload" in sys.argv

    print(f"Starting IDP Kit server at http://{host}:{port}")
    print("API docs at http://{host}:{port}/docs")

    uvicorn.run(
        "idpkit.api.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


if __name__ == "__main__":
    main()

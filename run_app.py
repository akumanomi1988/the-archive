"""Launcher for The Archive: starts uvicorn on 0.0.0.0, prints LAN IP and opens browser.

This is intended for use by end-users who want a simple "double-click" launcher.
It writes a small console output showing the URL to connect from other devices on the LAN.
"""
from __future__ import annotations

import socket
import webbrowser
from pathlib import Path
import sys

from uvicorn import Config, Server

ROOT = Path(__file__).parent


def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def main(host: str = "0.0.0.0", port: int = 8000, open_browser: bool = True):
    lan = get_local_ip()
    url_local = f"http://127.0.0.1:{port}"
    url_lan = f"http://{lan}:{port}"
    print("The Archive is starting")
    print(f"Local: {url_local}")
    print(f"LAN:   {url_lan}")
    if open_browser:
        try:
            webbrowser.open(url_local)
        except Exception:
            pass

    # Start uvicorn programmatically
    config = Config("backend:app", host=host, port=port, log_level="info")
    server = Server(config=config)
    server.run()


if __name__ == "__main__":
    # Allow optional CLI args: port and --no-browser
    port = 8000
    open_browser = True
    args = sys.argv[1:]
    for a in args:
        if a.startswith("--port="):
            try:
                port = int(a.split("=", 1)[1])
            except Exception:
                pass
        if a == "--no-browser":
            open_browser = False
    main(port=port, open_browser=open_browser)

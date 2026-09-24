"""Start Chainlit headlessly and verify its HTTP page and active upload settings."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import urlopen


def main() -> None:
    """Test the documented launch directory without opening or modifying the index."""
    app_dir = Path(__file__).resolve().parents[1] / "hw2"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    # Some IDEs export DEBUG=release; Chainlit's CLI requires a boolean value.
    env = {**os.environ, "CHAINLIT_APP_ROOT": str(app_dir), "DEBUG": "false"}
    with tempfile.TemporaryFile(mode="w+b") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "chainlit", "run", "app.py", "--headless",
             "--host", "127.0.0.1", "--port", str(port)],
            cwd=app_dir, env=env, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            deadline = time.monotonic() + 30
            while True:
                try:
                    with urlopen(f"http://127.0.0.1:{port}/", timeout=1) as response:
                        assert response.status == 200
                    break
                except (URLError, TimeoutError):
                    if process.poll() is not None or time.monotonic() >= deadline:
                        log.seek(0)
                        raise RuntimeError(log.read().decode(errors="replace"))
                    time.sleep(0.25)
            with urlopen(f"http://127.0.0.1:{port}/project/settings", timeout=5) as response:
                settings = json.load(response)
            uploads = settings["features"]["spontaneous_file_upload"]
            assert uploads["max_size_mb"] == 10, uploads
            assert uploads["max_files"] == 5, uploads
            assert uploads["accept"] == {"text/plain": [".txt"], "application/json": [".json"]}, uploads
            print(json.dumps({"homepage_status": 200, "upload_settings": uploads}, indent=2))
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()

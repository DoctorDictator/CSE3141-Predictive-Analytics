"""Start the TrafficRush API and website together: python run.py."""

from pathlib import Path
import json
import shutil
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
API = "http://127.0.0.1:8000/api/health"
WEBSITE = "http://127.0.0.1:5173/CSE3141-Predictive-Analytics/"


def ready(url):
    """Check our service, rather than assuming any process on the port is ours."""
    try:
        with urlopen(url, timeout=1) as response:
            body = response.read().decode("utf-8")
        if url == API:
            return json.loads(body).get("model_loaded") is True
        return "TrafficRush" in body
    except (OSError, ValueError):
        return False


def main():
    node = shutil.which("node")
    vite = ROOT / "frontend/node_modules/vite/bin/vite.js"
    if not node or not vite.exists():
        raise SystemExit("Install Node, then run npm ci inside frontend/.")
    if not (ROOT / "backend/artifacts/traffic_model.joblib").exists():
        raise SystemExit("First run: python make_notebook.py --execute")

    services = [
        (
            API,
            [
                sys.executable,
                "-m",
                "uvicorn",
                "backend.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ],
            ROOT,
        ),
        (
            WEBSITE,
            [node, str(vite), "--host", "127.0.0.1", "--port", "5173", "--strictPort"],
            ROOT / "frontend",
        ),
    ]
    started = []
    try:
        for url, command, folder in services:
            if ready(url):
                print("Already running:", url, flush=True)
                continue
            process = subprocess.Popen(command, cwd=folder)
            started.append(process)
            for _ in range(100):
                if ready(url):
                    break
                if process.poll() is not None:
                    raise RuntimeError(f"Service failed to start: {url}")
                time.sleep(0.2)
            else:
                raise RuntimeError(f"Service was not ready: {url}")
        print("\nTrafficRush is running: " + WEBSITE, flush=True)
        print("Press Ctrl+C to stop services started by this command.", flush=True)
        while True:
            if any(process.poll() is not None for process in started):
                raise RuntimeError(
                    "A TrafficRush service stopped. Check its output above."
                )
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping TrafficRush.")
    finally:
        # Leave pre-existing services alone; stop only children this command owns.
        for process in started:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)


if __name__ == "__main__":
    main()

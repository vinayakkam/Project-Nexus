"""
Run this file to start OLIT Nexus locally: python run.py
It starts the server and opens your browser to it automatically.
Press Ctrl+C in this window to stop the app.
"""
import os
import threading
import time
import webbrowser

PORT = int(os.environ.get("PORT", 5000))


def open_browser():
    time.sleep(1.5)  # give Flask a moment to start listening
    webbrowser.open(f"http://127.0.0.1:{PORT}")


if __name__ == "__main__":
    from app import app

    threading.Thread(target=open_browser, daemon=True).start()

    print(f"\nOLIT Nexus starting at http://127.0.0.1:{PORT}")
    print("First-ever chat message will take a while (~1GB model download).")
    print("Press Ctrl+C to stop.\n")

    app.run(host="127.0.0.1", port=PORT, debug=False)

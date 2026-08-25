# OLIT Nexus — Local

Runs entirely on your own machine. No Cloud Run, no Firestore, no billing.

## Setup (one-time)

1. Install Python 3.11 (other 3.x versions likely work too, untested).
2. In this folder, install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Place your `logo1.png` in the `frontend/` folder (referenced by both
   HTML files — the app runs fine without it, just shows a broken image icon).

## Running it

```
python run.py
```

This starts the server and opens your browser to `http://127.0.0.1:5000`
automatically. Press Ctrl+C in the terminal to stop it.

**First chat message will be slow** (30–90 seconds) — it downloads the
~1GB model from Hugging Face the first time only. After that it's cached
in `data/model_cache/` and loads instantly on every future run.

## What's stored where (all in `data/`, auto-created, gitignored)

- `olit_nexus.db` — SQLite database: accounts and chat history
- `.jwt_secret` — auto-generated login secret, stays stable across restarts
- `model_cache/` — the downloaded model, ~1GB

To fully reset (wipe all accounts and chat history), just delete the
`data/` folder — everything regenerates automatically on next run.

## Requirements

- Internet access is needed for: the first-run model download, web search
  (if you use it), and email validation on signup (checks the domain has
  real mail servers configured). Chat/coding and document summarization
  work fully offline once the model is downloaded once.
- Performance depends entirely on your CPU — no fixed vCPU ceiling like
  the cloud version had, but also no guarantee it's faster. A modern
  multi-core CPU should outperform the 2-vCPU cloud setup; an older or
  weaker machine may not.

## Multiple people using this

The account system works the same as the cloud version — anyone with
access to this machine (or your local network, if you change
`host="127.0.0.1"` to `host="0.0.0.0"` in `run.py`) can sign up and gets
their own separate chat history. Be aware `0.0.0.0` exposes it to your
whole network, not just your own machine.

"""Tiny Flask fixture that simulates an edge node serving job logs.

Log names are restricted to a strict allowlist pattern and resolved paths must
stay inside the log directory, so traversal attempts (``../../config``) always
return 404.
"""
from __future__ import annotations

import re
from pathlib import Path

from flask import Flask, abort, jsonify, request

app = Flask(__name__)

_LOG_DIR = (Path(__file__).resolve().parent / "logs").resolve()
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/logs")
def read_log():
    name = request.args.get("name", "")
    if not _SAFE_NAME.fullmatch(name) or ".." in name:
        abort(404)
    candidate = (_LOG_DIR / name).resolve()
    if candidate.parent != _LOG_DIR or not candidate.is_file():
        abort(404)
    return jsonify({"name": name, "content": candidate.read_text(encoding="utf-8")})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001)

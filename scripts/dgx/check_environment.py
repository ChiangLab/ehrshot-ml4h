#!/usr/bin/env python3
"""Check compute paths and an optional local OpenAI-compatible LLM endpoint."""

from __future__ import annotations

import argparse
import json
import os
import platform
import urllib.request
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/private/environment/dgx_environment.json"))
    args = parser.parse_args()
    endpoint = os.environ.get("LOCAL_LLM_BASE_URL")
    endpoint_models: list[str] | str = "not_checked"
    if endpoint:
        try:
            with urllib.request.urlopen(f"{endpoint.rstrip('/')}/models", timeout=10) as response:
                payload = json.load(response)
            endpoint_models = [str(item.get("id")) for item in payload.get("data", [])]
        except Exception as exc:
            endpoint_models = f"unavailable: {type(exc).__name__}"
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "project_root_exists": args.project_root.exists(),
        "project_root_writable": os.access(args.project_root, os.W_OK),
        "model_root_exists": args.model_root.exists(),
        "model_root_writable": os.access(args.model_root, os.W_OK) if args.model_root.exists() else False,
        "local_llm_endpoint_configured": bool(endpoint),
        "local_llm_model_ids": endpoint_models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["project_root_exists"] and result["project_root_writable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

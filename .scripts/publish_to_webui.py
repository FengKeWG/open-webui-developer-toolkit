#!/usr/bin/env python3
"""Upload or update a plugin file on an Open WebUI instance."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Final
from urllib.error import HTTPError
from urllib.request import Request, urlopen

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

CREATE: Final = "/api/v1/functions/create"
UPDATE: Final = "/api/v1/functions/id/{id}/update"


def _post(base_url: str, api_key: str, path: str, payload: dict) -> int:
    data = json.dumps(payload).encode()
    req = Request(
        url=f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Content-Length": str(len(data)),
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=30) as resp:
            return resp.getcode()
    except HTTPError as exc:
        return exc.code


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish one plugin to Open WebUI")
    parser.add_argument("file_path", nargs="+", help="Path to the .py plugin file")
    parser.add_argument("--type", choices=("pipe", "filter", "tool"))
    parser.add_argument("--url", default=os.getenv("WEBUI_URL", "http://localhost:8080"))
    parser.add_argument("--key", default=os.getenv("WEBUI_KEY", ""))
    return parser.parse_args()


def _detect_type(path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    parts = {part.lower() for part in path.parts}
    if "pipes" in parts:
        return "pipe"
    if "filters" in parts:
        return "filter"
    if "tools" in parts:
        return "tool"
    return "pipe"


def _extract_metadata(code: str) -> tuple[str, str, str]:
    plugin_id = next(
        (ln.split(":", 1)[1].strip() for ln in code.splitlines() if ln.lower().startswith("id:")), None
    )
    if not plugin_id:
        raise ValueError("'id:' line not found at top of file -- aborting")

    plugin_title = next(
        (ln.split(":", 1)[1].strip() for ln in code.splitlines() if ln.lower().startswith("title:")),
        plugin_id,
    )

    plugin_description = next(
        (ln.split(":", 1)[1].strip() for ln in code.splitlines() if ln.lower().startswith("description:")),
        "",
    )
    return plugin_id, plugin_title, plugin_description


def _build_payload(
    plugin_id: str, plugin_type: str, code: str, description: str, name: str
) -> dict:
    return {
        "id": plugin_id,
        "name": name,
        "type": plugin_type,
        "content": code,
        "meta": {"description": description, "manifest": {}},
        "is_active": True,
    }


def main() -> None:
    args = _parse_args()

    if not args.key:
        sys.exit("❌  WEBUI_KEY not set (flag --key or env var)")

    raw_path = " ".join(args.file_path)
    path = Path(raw_path)
    if not path.is_file():
        sys.exit(f"❌  File not found: {path}")
    logging.info("Reading plugin from %s", path)

    code = path.read_text(encoding="utf-8")

    try:
        plugin_id, plugin_name, description = _extract_metadata(code)
        logging.info("Plugin id: %s", plugin_id)
    except ValueError as exc:
        sys.exit(f"❌  {exc}")

    if not description:
        logging.warning("'description:' line not found - using empty description")

    plugin_type = _detect_type(path, args.type)
    payload = _build_payload(plugin_id, plugin_type, code, description, plugin_name)

    logging.info("Publishing '%s' (%s) to %s", plugin_id, plugin_type, args.url)
    status = _post(args.url, args.key, CREATE, payload)
    if status in (200, 201):
        logging.info("Created '%s' on %s [HTTP %s]", plugin_id, args.url, status)
        return

    if status == 400:
        status = _post(args.url, args.key, UPDATE.format(id=plugin_id), payload)
        if status in (200, 201):
            logging.info("Updated '%s' on %s [HTTP %s]", plugin_id, args.url, status)
            return

    sys.exit(f"❌  WebUI returned HTTP {status}")


if __name__ == "__main__":
    main()

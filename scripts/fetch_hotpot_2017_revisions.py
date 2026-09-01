#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

API = "https://en.wikipedia.org/w/api.php"
ANCHOR = "2017-10-01T23:59:59Z"
USER_AGENT = "VeriJoin-research/0.2 (academic reproducibility experiment)"


def fetch(title: str, *, direction: str, limit: int) -> dict[str, Any]:
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
        "prop": "revisions",
        "titles": title,
        "rvslots": "main",
        "rvprop": "ids|timestamp|sha1|size|content",
        "rvlimit": str(limit),
        "rvstart": ANCHOR,
        "rvdir": direction,
    }
    request = urllib.request.Request(
        API + "?" + urllib.parse.urlencode(params),
        headers={"User-Agent": USER_AGENT},
    )
    last_error: Exception | None = None
    for delay in (0.0, 1.0, 3.0, 10.0):
        if delay:
            time.sleep(delay)
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = json.loads(response.read())
            if "error" in payload:
                raise RuntimeError(str(payload["error"]))
            return payload
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as error:
            last_error = error
    raise RuntimeError(f"request failed after retries: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--newer-revisions", type=int, default=10)
    args = parser.parse_args()

    source = json.loads(args.profiles.read_text(encoding="utf-8"))
    titles = [
        item["title"]
        for item in source["requested_profiles"]
        if item["dataset"] == "hotpotqa"
    ]
    if args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        payload = {
            "api_endpoint": API,
            "anchor": ANCHOR,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "requested_titles": titles,
            "newer_revisions": args.newer_revisions,
            "responses": [],
            "complete": False,
        }
    complete = {
        item["request_title"]
        for item in payload["responses"]
        if "older" in item and "newer" in item
    }

    def persist() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    for index, title in enumerate(titles, start=1):
        if title in complete:
            continue
        try:
            older = fetch(title, direction="older", limit=1)
            time.sleep(0.5)
            newer = fetch(title, direction="newer", limit=args.newer_revisions)
            payload["responses"].append(
                {"request_title": title, "older": older, "newer": newer}
            )
        except RuntimeError as error:
            payload["responses"].append(
                {"request_title": title, "fetch_error": str(error)}
            )
        persist()
        print(f"{index}/{len(titles)} {title}", flush=True)
        time.sleep(0.5)
    payload["complete"] = True
    persist()


if __name__ == "__main__":
    main()

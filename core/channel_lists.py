# core/channel_lists.py
"""
Small persistence helpers for named channel-inclusion/exclusion lists.

A set of channel names - typically the ones SCI/SQA rejected during an
offline run - can be exported here and re-loaded by the offline or real-time
pipeline's 'channel_filter' step (see core.signal_pipeline), so the same
channels are excluded consistently instead of being re-discovered per context.

Deliberately free of pandas/UI dependencies: plain JSON/text file helpers.
"""
import json
import os
from datetime import datetime


def save_channel_list(path, channels, notes=""):
    """
    Writes a sorted, de-duplicated channel name list to a JSON file.

    Returns the sorted channel list actually written, so callers can log
    or display the final count without re-reading the file.
    """
    payload = {
        "channels": sorted(set(channels)),
        "metadata": {
            "created": datetime.now().isoformat(timespec="seconds"),
            "notes": notes,
        }
    }
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload["channels"]


def load_channel_list(path):
    """
    Reads a channel list file back into a plain list of channel names.

    Accepts three shapes so users aren't locked into exactly what
    save_channel_list() writes:
      - the structured {'channels': [...], 'metadata': {...}} this module writes
      - a bare JSON list: ["S(1)_D(2)", "S(3)_D(4)"]
      - a flat text/CSV file: names separated by commas and/or newlines
    """
    with open(path, "r") as f:
        raw = f.read()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "channels" in parsed:
            return [str(c) for c in parsed["channels"]]
        if isinstance(parsed, list):
            return [str(c) for c in parsed]
    except json.JSONDecodeError:
        pass

    # Fallback: plain text, comma and/or newline separated
    tokens = [tok.strip() for line in raw.splitlines() for tok in line.split(",")]
    return [t for t in tokens if t]

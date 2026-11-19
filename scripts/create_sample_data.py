"""Validate the checked-in synthetic incidents used by the local demo."""
import json
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "data" / "sample_incidents.json"
items = json.loads(path.read_text(encoding="utf-8"))
assert items and all({"id", "summary", "resolution"} <= item.keys() for item in items)
print(f"Validated {len(items)} synthetic incidents")
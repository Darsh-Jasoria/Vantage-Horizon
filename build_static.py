"""Bake the latest forecast into a single self-contained HTML snapshot.

    python build_static.py   ->  outputs/vantage_horizon_snapshot.html

Handy for sharing or for judges without Python: open the file in any browser.
"""
import json
from vantage.config import OUT_DIR, ROOT

page = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
data = json.loads((OUT_DIR / "dashboard_data.json").read_text())
data["meta"]["mode"] = "snapshot"
blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
html = page.replace("/*__VANTAGE_DATA__*/null", blob)
out = OUT_DIR / "vantage_horizon_snapshot.html"
out.write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
               '<meta name="viewport" content="width=device-width, initial-scale=1">' + html, encoding="utf-8")
# body-only version for hosts that add their own document skeleton
(OUT_DIR / "vantage_horizon_artifact.html").write_text(html, encoding="utf-8")
print("wrote", out)

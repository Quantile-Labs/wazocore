"""v12 stages A and B (docs/METHOD.md, section 4.6), run on the pod so no person chooses.

  python -I scripts/v12_pick.py rank RESULTS     # rank.json; prints the ranking
  python -I scripts/v12_pick.py choose RESULTS   # pick.json from sel-*.json samples

Stage B uses the project's 1.0 chrF++ margin (v9 and v11 used 0.5); the
choice under 0.5 is also written, as the plan's sensitivity analysis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "wazocore"))
sys.path.insert(0, "/workspace/wazocore")
from v9_pick import r_half, translation  # noqa: E402
from wazocore.vx_core import qualifies  # noqa: E402

MODELS = ("base", "short", "anchor")  # the plan's order, which breaks ties
MARGIN, OLD_MARGIN = 1.0, 0.5


def ranked(r: dict[str, float]) -> list[str]:
    return sorted(r, key=lambda m: (-r[m], MODELS.index(m)))


def main(argv: list[str]) -> int:
    mode, results = argv[0], Path(argv[1])
    if mode == "rank":
        r = {m: r_half(results, m) for m in MODELS}
        order = ranked(r)
        (results / "rank.json").write_text(json.dumps({"selection_r": r, "top": order}, indent=1) + "\n")
        print(" ".join(order))
        return 0
    if mode == "choose":
        order = json.loads((results / "rank.json").read_text())["top"]
        release = translation(results / "sel-release.json")
        checked = {m: translation(results / f"sel-{m}.json") for m in order}
        ok = {m: qualifies(checked[m], release, margin=MARGIN) for m in order}
        ok_old = {m: qualifies(checked[m], release, margin=OLD_MARGIN) for m in order}
        cand = next((m for m in order if ok[m]), None)
        (results / "pick.json").write_text(json.dumps({
            "margin": MARGIN, "release": release, "checked": checked, "qualifies": ok, "candidate": cand,
            "sensitivity": {"margin": OLD_MARGIN, "qualifies": ok_old, "candidate": next((m for m in order if ok_old[m]), None)}}, indent=1) + "\n")
        print(cand)
        return 0
    raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""v11 stages A and B (docs/METHOD.md, section 4.6), run on the pod so no person chooses.

  python -I scripts/v11_pick.py rank RESULTS     # rank.json; prints the top three
  python -I scripts/v11_pick.py choose RESULTS   # pick.json from sel-*.json samples

`rank` reads only the selection half (even doc_id) of R in each
iroko-<model>/ folder; ties go to the model listed first in MODELS.
`choose` applies stage B to English -> X samples on NTREX lines 0 to 499.
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

MODELS = ("sdft-s1", "sdft-mid", "l1", "lowlr", "top")  # the plan's order, which breaks ties


def top_three(r: dict[str, float]) -> list[str]:
    return sorted(r, key=lambda m: (-r[m], MODELS.index(m)))[:3]


def main(argv: list[str]) -> int:
    mode, results = argv[0], Path(argv[1])
    if mode == "rank":
        r = {m: r_half(results, m) for m in MODELS}
        top = top_three(r)
        (results / "rank.json").write_text(json.dumps({"selection_r": r, "top": top}, indent=1) + "\n")
        print(" ".join(top))
        return 0
    if mode == "choose":
        rank = json.loads((results / "rank.json").read_text())
        release = translation(results / "sel-release.json")
        checked = {m: translation(results / f"sel-{m}.json") for m in rank["top"]}
        ok = {m: qualifies(checked[m], release) for m in rank["top"]}
        cand = next((m for m in rank["top"] if ok[m]), None)  # top is already in selection-R order
        (results / "pick.json").write_text(json.dumps({"release": release, "checked": checked, "qualifies": ok, "candidate": cand}, indent=1) + "\n")
        print(cand)
        return 0
    raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

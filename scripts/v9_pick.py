"""v9 stages A and B (docs/METHOD.md, section 4.6), run on the pod so no person chooses.

  python -I scripts/v9_pick.py rank RESULTS     # writes rank.json, prints the top three
  python -I scripts/v9_pick.py choose RESULTS   # writes pick.json from sel-*.json samples

`rank` reads only the selection half (even doc_id) of the R items in each
iroko-<point>/ folder. `choose` applies the stage B rule to English -> X
samples on NTREX lines 0 to 499 (sel-<point>.json, sel-release.json).
"""

from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "wazocore"))
sys.path.insert(0, "/workspace/wazocore")
from wazocore.vx_core import half, qualifies, top_points  # noqa: E402

R_TASKS = [f"{t}_{l}" for t in ("afrixnli", "afrimgsm") for l in ("yor", "hau", "ibo")]
LANGS = ("yo", "ha", "ig")


def grid() -> dict[str, tuple[float, float]]:
    """Release grid points, by folder name, with their (b, t). (1, 1) is the release."""
    tag = {0.0: "0", 0.5: "05", 1.0: "1"}
    return {("release" if (b, t) == (1.0, 1.0) else f"rel-b{tag[b]}-t{tag[t]}"): (b, t) for b in tag for t in tag}


def r_half(results: Path, point: str, which: str = "selection") -> float:
    hits = []
    for task in R_TASKS:
        files = list(results.glob(f"iroko-{point}/*/samples_{task}_prompt_1_*.jsonl"))
        if len(files) != 1:
            raise SystemExit(f"{point}: expected one samples file for {task}, found {len(files)}")
        key = "exact_match" if "afrimgsm" in task else "acc"
        rows = [x for x in map(json.loads, files[0].read_text().splitlines()) if x["filter"] in ("none", "flexible-extract")]
        if len({x["doc_id"] for x in rows}) != len(rows):
            raise SystemExit(f"{point} {task}: an item is scored twice")
        hits += [float(x[key]) for x in rows if half(x["doc_id"]) == which]
    return 100 * sum(hits) / len(hits)


def translation(path: Path) -> dict[str, dict]:
    import sacrebleu

    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    nfc = lambda s: unicodedata.normalize("NFC", s)
    out = {}
    for lang in LANGS:
        rs = [r for r in rows if r["lang"] == lang]
        out[lang] = {"chrf++": sacrebleu.corpus_chrf([nfc(r["output"]) for r in rs], [[nfc(r["reference"]) for r in rs]], word_order=2).score,
                     "chars_per_token": sum(len(r["output"]) for r in rs) / max(1, sum(r["tokens"] for r in rs)), "n": len(rs)}
    return out


def main(argv: list[str]) -> int:
    mode, results = argv[0], Path(argv[1])
    points = grid()
    if mode == "rank":
        r = {p: r_half(results, p) for p in points}
        top = top_points(r, {p: b + t for p, (b, t) in points.items()})
        (results / "rank.json").write_text(json.dumps({"selection_r": r, "top": top}, indent=1) + "\n")
        print(" ".join(top))
        return 0
    rank = json.loads((results / "rank.json").read_text())
    release = translation(results / "sel-release.json")
    checked = {p: translation(results / f"sel-{p}.json") for p in rank["top"]}
    ok = {p: qualifies(s, release) for p, s in checked.items()}
    candidate = next((p for p in rank["top"] if ok[p]), None)
    (results / "pick.json").write_text(json.dumps({"release": release, "checked": checked, "qualifies": ok, "candidate": candidate,
                                                   "b_t": points.get(candidate)}, indent=1) + "\n")
    print(candidate or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

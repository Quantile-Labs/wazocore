"""v13 comparisons from docs/METHOD.md, section 4.6, on the pulled results. `base` is
v12's (same recipe, code and seed); the release and the original are v9's
(generative and domain-conditional AfriXNLI v12's); held-out baselines as
in v12. Paired over items or sentences, 5,000 bootstrap rounds.

    uv run --project wazocore --with sacrebleu python scripts/v13_analysis.py results/vocab
    python -I v13_analysis.py /workspace/work                       # on the pod: decision.json
    python -I v13_analysis.py check-dc RESULTS_V13 MAIN_DIR MODEL   # on the pod: the bug check
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "wazocore"))
sys.path.insert(0, "/workspace/wazocore")
from v7_analysis import LANGS, compare  # noqa: E402
from v8_analysis import ACC, R_TASKS  # noqa: E402
from v9_analysis import boot, items, pooled, show, translation  # noqa: E402
from v10_analysis import chars_per_token, formats  # noqa: E402
from v12_analysis import MGSM, XNLI, baseline, check_dc, contradiction, dc_accuracy, halves, r_gen  # noqa: E402

MARGIN, SPREAD, RATIO, MATHS = 1.0, 2.0, 0.95, 3.0
ARMS = ("no-chat", "no-math")


def folders(base: Path) -> dict[int, Path]:
    return {k: base / f"v{k}" if (base / f"v{k}").exists() else base / f"results-v{k}" for k in range(7, 14)}


def main(base: Path) -> int:
    dirs = folders(base)
    v7, v8, v9, v12, v13 = (dirs[k] for k in (7, 8, 9, 12, 13))
    out: dict = {}
    pick = json.loads((v13 / "pick.json").read_text()) if (v13 / "pick.json").exists() else None
    cand = pick and pick["candidate"]
    where = [("n-atlas", v9), ("release", v9), ("release-v13", v13), ("base", v12), ("long-s1", v8), *[(m, v13) for m in ARMS]]
    if cand:
        where += [(f"{cand}-s3", v13), (f"{cand}-s4", v13)]
    scores = {m: s for m, d in where if (s := items(d, m)) and all(t in s for t in R_TASKS)}
    gen_dir = {"n-atlas": v12, "release": v12, "base": v12, **{m: v13 for m in ARMS}}
    gen = {m: g for m, d in gen_dir.items() if (g := items(d, f"gen-{m}")) and all(t in g for t in XNLI)}
    out["R"] = {m: {h or "all": ACC(pooled(s, h)) for h in ("selection", "confirmation", None)} for m, s in scores.items()}
    out["R-gen"] = {m: ACC(r_gen(gen[m], scores[m])) for m in gen if m in scores}
    for m, r in out["R"].items():
        print(f"R {m:>12}: selection {r['selection']:.1f}  confirmation {r['confirmation']:.1f}  all {r['all']:.1f}"
              + (f"  R-gen {out['R-gen'][m]:.1f}" if m in out["R-gen"] else ""))
    if "release-v13" in scores:
        same = all(np.array_equal(scores["release-v13"][t][1], scores["release"][t][1]) for t in R_TASKS)
        out["reproduction | release v13 equals v9"] = bool(same)
        print(f"reproduction | release v13 equals v9 item for item: {same}")

    decision: dict = {"stage2": False, "candidate": cand}

    # Question 1: which half suppresses contradiction
    con = {m: c for m, d in gen_dir.items() if (c := contradiction(d, m))}
    for m, (hits, shares) in con.items():
        out.setdefault("contradiction", {})[m] = {"recall": ACC(hits), "written": shares}
        print(f"contradiction | {m}: recall {ACC(hits):.1f} on {len(hits)} items; writes {shares}")
    for m, half in (("no-chat", "chat"), ("no-math", "maths")):
        if m in con and "base" in con:
            h = {x: con[x][0] for x in (m, "base")}
            c = out[f"contradiction | {m} vs base"] = compare(h, [m], ["base"], ACC, len(h[m]))
            show(f"contradiction | {m} vs base | recall", c)
            decision[f"q1_{half}_suppresses"] = bool(c["ci95"][0] > 0)
    if all(m in con for m in (*ARMS, "base")):
        h = {m: con[m][0] for m in (*ARMS, "base")}
        n = len(h["base"])
        out["contradiction | interaction"] = c = boot(lambda i: ACC(h["no-chat"][i]) + ACC(h["no-math"][i]) - 2 * ACC(h["base"][i]), n)
        show("contradiction | no-chat + no-math - 2 base", c)

    # Question 2: recovery, as in v12
    out["pick"] = pick
    if pick:
        print(f"stage A: {json.loads((v13 / 'rank.json').read_text())['top']}; stage B (margin {pick['margin']}): {cand}; "
              f"under 0.5: {pick['sensitivity']['candidate']}")
    if cand and cand in scores:
        half = {m: pooled(scores[m], "confirmation") for m in (cand, "release")}
        c = out["primary | candidate vs release | R confirmation"] = compare(half, [cand], ["release"], ACC, len(half[cand]))
        show(f"q2 | {cand} vs release | R, confirmation half", c)
        q2 = {"recovery_R": bool(c["ci95"][0] > 0 and c["delta"] > SPREAD)}
        if cand in gen and "release" in gen:
            hg = {m: r_gen(gen[m], scores[m], "confirmation") for m in (cand, "release")}
            g = out["primary | candidate vs release | R-gen confirmation"] = compare(hg, [cand], ["release"], ACC, len(hg[cand]))
            show(f"q2 | {cand} vs release | R-gen, confirmation half", g)
            q2["recovery_R_gen"] = bool(g["ci95"][0] > 0)
        q2["recovery"] = bool(q2["recovery_R"] and q2.get("recovery_R_gen", False))
        if all((v13 / f"heldout-{w}-{cand}.json").exists() for w in ("en-xx", "xx-en")):
            lows = []
            for way in ("en-xx", "xx-en"):
                files = {cand: v13 / f"heldout-{way}-{cand}.json", **{m: baseline(dirs, way, m) for m in ("release", "n-atlas")}}
                translation(out, files, [(cand, "release"), (cand, "n-atlas")], way, "held-out")
                lows += [out[f"held-out | {cand} vs release | {way} | {l}"]["chrf"]["lower95"] for l in LANGS]
            q2["translation_holds"] = bool(all(x > -MARGIN for x in lows))
            mine, rel = chars_per_token(v13 / f"heldout-en-xx-{cand}.json"), chars_per_token(v7 / "heldout-en-xx-release.json")
            q2["token_saving_holds"] = bool(all(mine[l] >= RATIO * rel[l] for l in LANGS))
        if "base" in scores:
            mg = {m: halves(scores[m], MGSM, None) for m in (cand, "base")}
            c = out["maths | candidate vs base"] = compare(mg, [cand], ["base"], ACC, len(mg[cand]))
            show(f"maths | {cand} vs base | AfriMGSM yo, ha, ig", c)
            q2["maths_holds"] = bool(c["lower95"] > -MATHS)
        decision["q2"] = q2
        decision["stage2"] = bool(all(q2.get(k) for k in ("recovery", "translation_holds", "token_saving_holds", "maths_holds")))
        print(f"decision | question 2: {q2}")

    # Secondary
    for m in [x for x in ARMS if x in scores]:
        for b in [b for b in ("base", "release", "n-atlas") if b in scores]:
            for t in sorted(scores[m]):
                if t in scores[b]:
                    s = {m: scores[m][t][1], b: scores[b][t][1]}
                    out[f"secondary | {m} vs {b} | {t}"] = compare(s, [m], [b], ACC, len(s[m]))
            for h in ("selection", "confirmation", None):
                p = {x: pooled(scores[x], h) for x in (m, b)}
                out[f"secondary | {m} vs {b} | R {h or 'all'}"] = compare(p, [m], [b], ACC, len(p[m]))
            show(f"secondary | {m} vs {b} | R all", out[f"secondary | {m} vs {b} | R all"])
            if m in gen and b in gen:
                p = {x: r_gen(gen[x], scores[x]) for x in (m, b)}
                out[f"secondary | {m} vs {b} | R-gen all"] = c = compare(p, [m], [b], ACC, len(p[m]))
                show(f"secondary | {m} vs {b} | R-gen all", c)
    dc = {m: a for m, d in (("base", v12), *[(m, v13) for m in ARMS]) if (a := dc_accuracy(d, d, m)) is not None}
    out["domain-conditional"] = {m: ACC(a) for m, a in dc.items()}
    out["formats"] = {m: f for m, d in [("base", v12), *[(m, v13) for m in ARMS]] if (f := formats(d, m))}

    seeds = [m for m in (cand, f"{cand}-s3", f"{cand}-s4") if cand and m in scores]
    if len(seeds) == 3 and "long-s1" in scores:
        p = {m: pooled(scores[m]) for m in (*seeds, "release", "long-s1")}
        out["stage 2 | candidate (3 seeds) vs release (2 seeds) | R"] = c = compare(p, seeds, ["release", "long-s1"], ACC, len(p["release"]))
        show("stage 2 | candidate (3 seeds) vs release (2 seeds) | R", c)
        # Secondary, added after the result (a logged deviation): each seed's R-gen and recall
        for m in seeds[1:]:
            if (g := items(v13, f"gen-{m}")) and all(t in g for t in XNLI):
                gen[m] = g
            if c := contradiction(v13, m):
                con[m] = c
        if all(m in gen for m in seeds):
            for m in seeds:
                out.setdefault("stage 2 | R-gen", {})[m] = ACC(r_gen(gen[m], scores[m]))
                out.setdefault("stage 2 | recall", {})[m] = ACC(con[m][0])
                print(f"stage 2 | {m}: R-gen {ACC(r_gen(gen[m], scores[m])):.1f}  recall {ACC(con[m][0]):.1f}")
            p = {m: r_gen(gen[m], scores[m]) for m in (*seeds, "release")}
            out["stage 2 | candidate (3 seeds) vs release | R-gen"] = c = compare(p, seeds, ["release"], ACC, len(p["release"]))
            show("stage 2 | candidate (3 seeds) vs release (seed 0) | R-gen", c)
    if (v13 / "split-report.json").exists():
        out["split-report"] = json.loads((v13 / "split-report.json").read_text())
    out["decision"] = decision
    print(f"decision: {decision}")
    if v13.exists():
        (v13 / "analysis.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item()) + "\n")
        (v13 / "decision.json").write_text(json.dumps(decision, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "check-dc":
        sys.exit(check_dc(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]))
    sys.exit(main(Path(sys.argv[1])))

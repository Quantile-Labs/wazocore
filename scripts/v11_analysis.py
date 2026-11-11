"""v11 comparisons from docs/METHOD.md, section 4.6, on the pulled results. Sections run
only when their files have arrived. IrokoBench for the release and the
original is v9's (AfriMMLU v10's), both at batch 16 with the same lm-eval;
English -> X samples of the release and the original are v7's, X -> English
v9's level-0 samples (NFC sources). Paired over items or sentences, 5,000
bootstrap rounds.

    uv run --project wazocore --with sacrebleu python scripts/v11_analysis.py results/vocab
    python -I v11_analysis.py /workspace/work          # on the pod: writes decision.json
"""

import json
import sys
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "wazocore"))
sys.path.insert(0, "/workspace/wazocore")
from v7_analysis import LANGS, compare  # noqa: E402
from v8_analysis import ACC, R_TASKS  # noqa: E402
from v9_analysis import boot, items, pooled, show, translation  # noqa: E402
from v10_analysis import chars_per_token, formats, samples  # noqa: E402

MARGIN, SPREAD, RATIO, MATHS, ARTEFACT = 1.0, 2.0, 0.95, 3.0, 2.0
MODELS = ("sdft-s1", "sdft-mid", "l1", "lowlr", "top")
CODE = {"yo": "yor", "ha": "hau", "ig": "ibo"}
XNLI = [f"afrixnli_{c}" for c in CODE.values()]
MGSM = [f"afrimgsm_{c}" for c in CODE.values()]


def folders(base: Path) -> dict[int, Path]:
    return {k: base / f"v{k}" if (base / f"v{k}").exists() else base / f"results-v{k}" for k in (7, 8, 9, 10, 11)}


def baseline(dirs: dict, way: str, model: str) -> Path:
    return dirs[9] / f"unmarked-l0-xx-en-{model}.json" if way == "xx-en" else dirs[7] / f"heldout-{way}-{model}.json"


def cat(scores: dict, tasks: list[str], which: str | None = None) -> np.ndarray:
    keep = {"selection": 0, "confirmation": 1}
    return np.concatenate([s if which is None else s[ids % 2 == keep[which]] for ids, s in (scores[t] for t in tasks)])


def changed_by_nfc(xs: list[dict]) -> np.ndarray:
    text = lambda d: " ".join(v for v in d.values() if isinstance(v, str))
    return np.array([unicodedata.normalize("NFC", text(x["doc"])) != text(x["doc"]) for x in xs])


def main(base: Path) -> int:
    dirs = folders(base)
    v7, v8, v9, v10, v11 = (dirs[k] for k in (7, 8, 9, 10, 11))
    out: dict = {}
    where = [("n-atlas", v9), ("release", v9), ("release-v11", v11), ("sdft", v10), ("long-s1", v8), *[(m, v11) for m in MODELS]]
    pick = json.loads((v11 / "pick.json").read_text()) if (v11 / "pick.json").exists() else None
    cand = pick and pick["candidate"]
    if cand:
        where += [(f"{cand}-s2", v11), (f"{cand}-s3", v11)]
    scores = {m: s for m, d in where if (s := items(d, m)) and all(t in s for t in R_TASKS)}
    out["R"] = {m: {"selection": ACC(pooled(s, "selection")), "confirmation": ACC(pooled(s, "confirmation")), "all": ACC(pooled(s))}
                for m, s in scores.items()}
    for m, r in out["R"].items():
        print(f"R {m:>12}: selection {r['selection']:.1f}  confirmation {r['confirmation']:.1f}  all {r['all']:.1f}")
    if "release-v11" in scores:
        same = all(np.array_equal(scores["release-v11"][t][1], scores["release"][t][1]) for t in R_TASKS)
        out["reproduction | release v11 equals v9"] = bool(same)
        print(f"reproduction | release v11 equals v9 item for item: {same}")

    # Question 1
    decision: dict = {"stage2": False, "candidate": cand}
    out["pick"] = pick
    if pick:
        print(f"stage A: {json.loads((v11 / 'rank.json').read_text())['top']}; stage B: candidate {cand}")
    if cand and cand in scores:
        half = {m: pooled(scores[m], "confirmation") for m in (cand, "release", "n-atlas")}
        n = len(half[cand])
        c = out["primary | candidate vs release, confirmation half"] = compare(half, [cand], ["release"], ACC, n)
        show(f"primary | {cand} vs release (confirmation half)", c)
        share = lambda i: (ACC(half[cand][i]) - ACC(half["release"][i])) / (ACC(half["n-atlas"][i]) - ACC(half["release"][i]))
        s = out["primary | recovered share"] = boot(share, n)
        print(f"primary | recovered share: {s['delta']:+.2f} [{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}]")
        q1 = {"recovery": bool(c["ci95"][0] > 0 and c["delta"] > SPREAD)}
        if all((v11 / f"heldout-{w}-{cand}.json").exists() for w in ("en-xx", "xx-en")):
            lows = []
            for way in ("en-xx", "xx-en"):
                files = {cand: v11 / f"heldout-{way}-{cand}.json", **{m: baseline(dirs, way, m) for m in ("release", "n-atlas")}}
                translation(out, files, [(cand, "release"), (cand, "n-atlas")], way, "held-out")
                lows += [out[f"held-out | {cand} vs release | {way} | {l}"]["chrf"]["lower95"] for l in LANGS]
            q1["translation_holds"] = bool(all(x > -MARGIN for x in lows))
            mine, rel = chars_per_token(v11 / f"heldout-en-xx-{cand}.json"), chars_per_token(v7 / "heldout-en-xx-release.json")
            out["candidate | chars per token vs release"] = {l: mine[l] / rel[l] for l in LANGS}
            q1["token_saving_holds"] = bool(all(mine[l] >= RATIO * rel[l] for l in LANGS))
        if "sdft-s1" in scores:
            mg = {m: cat(scores[m], MGSM) for m in (cand, "sdft-s1")}
            c = out["maths | candidate vs sdft-s1 | AfriMGSM yo ha ig"] = compare(mg, [cand], ["sdft-s1"], ACC, len(mg[cand]))
            show(f"maths | {cand} vs sdft-s1 | AfriMGSM yo, ha, ig", c)
            q1["maths_holds"] = bool(c["lower95"] > -MATHS)
        decision["q1"] = q1
        decision["stage2"] = bool(all(q1.get(k) for k in ("recovery", "translation_holds", "token_saving_holds", "maths_holds")))
        print(f"decision | question 1: {q1}")

    # Secondary: every model on both halves and per task; the v10 recipe's seed spread
    for m in [x for x in MODELS if x in scores]:
        for b in [b for b in ("release", "n-atlas", "sdft-s1") if b in scores and b != m]:
            for t in sorted(scores[m]):
                if t in scores[b]:
                    s = {m: scores[m][t][1], b: scores[b][t][1]}
                    out[f"secondary | {m} vs {b} | {t}"] = compare(s, [m], [b], ACC, len(s[m]))
            for h in ("selection", "confirmation", None):
                p = {x: pooled(scores[x], h) for x in (m, b)}
                out[f"secondary | {m} vs {b} | R {h or 'all'}"] = c = compare(p, [m], [b], ACC, len(p[m]))
            show(f"secondary | {m} vs {b} | R all", out[f"secondary | {m} vs {b} | R all"])
    if "sdft-s1" in scores and "sdft" in scores:
        p = {m: pooled(scores[m]) for m in ("sdft-s1", "sdft")}
        out["seed spread | sdft-s1 vs sdft | R"] = c = compare(p, ["sdft-s1"], ["sdft"], ACC, len(p["sdft"]))
        show("seed spread | sdft-s1 vs sdft | R", c)
    out["formats"] = {m: f for m, d in [("n-atlas", v9), ("release", v9), ("sdft", v10), *[(m, v11) for m in MODELS]] if (f := formats(d, m))}

    # Question 2: generative AfriXNLI
    gen = {m: g for m in ("n-atlas", "release", *MODELS) if (g := items(v11, f"gen-{m}")) and all(t in g for t in XNLI)}
    lik = {m: cat(scores[m], XNLI) for m in gen if m in scores}
    for m, g in gen.items():
        nolab = int(sum(x["no_label"] for t in XNLI for x in samples(v11, f"gen-{m}", t) or []))
        out.setdefault("generative", {})[m] = {"generative": ACC(cat(g, XNLI)), "likelihood": ACC(lik[m]) if m in lik else None, "no_label": nolab}
        print(f"generative | {m}: AfriXNLI {ACC(cat(g, XNLI)):.1f} (likelihood {ACC(lik[m]) if m in lik else float('nan'):.1f}); {nolab} of 1,800 replies name no label")
    if "n-atlas" in gen:
        g = {m: cat(x, XNLI) for m, x in gen.items()}
        n = len(g["n-atlas"])
        for m in [x for x in ("release", "sdft-s1") if x in g and x in lik]:
            cg = out[f"generative | {m} vs n-atlas"] = compare(g, [m], ["n-atlas"], ACC, n)
            cl = out[f"likelihood | {m} vs n-atlas"] = compare(lik, [m], ["n-atlas"], ACC, n)
            show(f"generative | {m} vs n-atlas | AfriXNLI yo ha ig", cg)
            show(f"likelihood | {m} vs n-atlas | AfriXNLI yo ha ig", cl)
            art = bool(cg["lower95"] > -ARTEFACT and cl["ci95"][1] < 0)
            decision[f"q2_scoring_artefact_{m}"] = art
            print(f"decision | question 2 | scoring artefact for {m}: {art}")
        for m in [x for x in g if x in lik]:
            out[f"generative - likelihood | {m}"] = c = boot(lambda i, m=m: ACC(g[m][i]) - ACC(lik[m][i]), n)
            show(f"generative - likelihood | {m}", c)

    # Question 3: NFC
    for m in ("n-atlas", "release", *([cand] if cand else [])):
        nfc = items(v11, f"nfc-{m}")
        if not nfc:
            continue
        given = {**(items(v9, m) if m in ("n-atlas", "release") else items(v11, m)), **(items(v10, f"{m}-mmlu") or {} if m in ("n-atlas", "release") else {})}
        for t in sorted(nfc):
            if t not in given:
                continue
            a, b = nfc[t][1], given[t][1]
            d = boot(lambda i: ACC(a[i]) - ACC(b[i]), len(a))
            out[f"nfc | {m} | {t}"] = d
            xs = samples(v11, f"nfc-{m}", t) or []
            src = samples(v9 if m in ("n-atlas", "release") else v11, m, t) or samples(v10, f"{m}-mmlu", t) or []
            ch = changed_by_nfc(src) if len(src) == len(a) else np.zeros(len(a), bool)
            if ch.any():
                idx = np.flatnonzero(ch)
                out[f"nfc | {m} | {t} | changed items"] = dc = boot(lambda i: ACC(a[idx][i]) - ACC(b[idx][i]), len(idx))
                print(f"nfc | {m} | {t}: all {d['delta']:+.1f} [{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}]; {len(idx)} changed items {dc['delta']:+.1f} [{dc['ci95'][0]:+.1f}, {dc['ci95'][1]:+.1f}]")
            else:
                show(f"nfc | {m} | {t} (no item changed)", d)
    a, b = v11 / "heldout-xx-en-release-nfctok.json", v11 / "heldout-xx-en-release-nfc.json"
    if a.exists() and b.exists():
        ra, rb = json.loads(a.read_text())["rows"], json.loads(b.read_text())["rows"]
        same = len(ra) == len(rb) and all(x["output"] == y["output"] for x, y in zip(ra, rb))
        out["tokenizer check | NFC normalizer reproduces NFC input"] = bool(same)
        decision["q3_tokenizer_check"] = bool(same)
        print(f"tokenizer check: every reply matches: {same} ({sum(x['output'] != y['output'] for x, y in zip(ra, rb))} differ)")

    # Stage 2
    seeds = [m for m in (cand, f"{cand}-s2", f"{cand}-s3") if cand and m in scores]
    if len(seeds) == 3 and "long-s1" in scores:
        p = {m: pooled(scores[m]) for m in (*seeds, "release", "long-s1")}
        c = out["stage 2 | candidate (3 seeds) vs release (2 seeds) | R"] = compare(p, seeds, ["release", "long-s1"], ACC, len(p["release"]))
        show("stage 2 | candidate (3 seeds) vs release (2 seeds) | R", c)

    out["decision"] = decision
    print(f"decision: {decision}")
    if v11.exists():
        (v11 / "analysis.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item()) + "\n")
        (v11 / "decision.json").write_text(json.dumps(decision, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))

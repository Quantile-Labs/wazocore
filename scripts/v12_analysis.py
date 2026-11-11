"""v12 comparisons from docs/METHOD.md, section 4.6, on the pulled results. Sections run
only when their files have arrived. IrokoBench for the release and the
original is v9's (generative AfriXNLI v12's own, else v11's); English -> X
samples of the release and the original are v7's, X -> English v9's
level-0 samples (NFC sources). Paired over items or sentences, 5,000
bootstrap rounds.

    uv run --project wazocore --with sacrebleu python scripts/v12_analysis.py results/vocab
    python -I v12_analysis.py /workspace/work                      # on the pod: decision.json
    python -I v12_analysis.py check-dc RESULTS_V12 MAIN_DIR MODEL  # on the pod: the bug check
"""

import json
import sys
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "wazocore"))
sys.path.insert(0, "/workspace/wazocore")
from v7_analysis import LANGS, chrf_stats, compare, rows  # noqa: E402
from v8_analysis import ACC, CHRF, R_TASKS  # noqa: E402
from v9_analysis import boot, items, pooled, show, translation  # noqa: E402
from v10_analysis import chars_per_token, formats, logprobs, samples  # noqa: E402
from wazocore.vx_core import calibrated_choice, written_label  # noqa: E402

MARGIN, SPREAD, RATIO, MATHS, TOLERANCE = 1.0, 2.0, 0.95, 3.0, 0.25
MODELS = ("base", "short", "anchor")
CODE = {"yo": "yor", "ha": "hau", "ig": "ibo"}
XNLI = [f"afrixnli_{c}" for c in CODE.values()]
MGSM = [f"afrimgsm_{c}" for c in CODE.values()]
LABELS = ["entailment", "neutral", "contradiction"]


def folders(base: Path) -> dict[int, Path]:
    return {k: base / f"v{k}" if (base / f"v{k}").exists() else base / f"results-v{k}" for k in (7, 8, 9, 10, 11, 12)}


def baseline(dirs: dict, way: str, model: str) -> Path:
    return dirs[9] / f"unmarked-l0-xx-en-{model}.json" if way == "xx-en" else dirs[7] / f"heldout-{way}-{model}.json"


def halves(scores: dict, tasks: list[str], which: str | None) -> np.ndarray:
    keep = {"selection": 0, "confirmation": 1}
    return np.concatenate([s if which is None else s[ids % 2 == keep[which]] for ids, s in (scores[t] for t in tasks)])


def r_gen(gen: dict, main: dict, which: str | None = None) -> np.ndarray:
    """R with generative AfriXNLI in place of the likelihood score."""
    return np.concatenate([halves(gen, XNLI, which), halves(main, MGSM, which)])


def check_dc(v12: Path, main_dir: Path, model: str) -> int:
    worst, same = 0.0, True
    for code in ("yor", "hau", "ibo", "eng"):
        xs, ref = samples(v12, f"dc-{model}", f"afrixnli_{code}"), samples(main_dir, model, f"afrixnli_{code}")
        if not xs or not ref or [x["doc_id"] for x in xs] != [0, 1] or xs[1]["doc"]["premise"] != "":
            print(f"check-dc {model} {code}: files missing or unexpected documents", file=sys.stderr)
            return 1
        same &= xs[0]["arguments"] == ref[0]["arguments"] and xs[0]["doc"] == ref[0]["doc"]
        worst = max(worst, max(abs(a - b) for a, b in zip(logprobs(xs[0]), logprobs(ref[0]))))
    print(f"check-dc {model}: requests identical {same}; largest difference {worst:.4f} (bound {TOLERANCE})", file=sys.stderr)
    return 0 if same and worst <= TOLERANCE else 1


def dc_accuracy(main_dir: Path, v12: Path, model: str) -> np.ndarray | None:
    out = []
    for code in CODE.values():
        xs, dc = samples(main_dir, model, f"afrixnli_{code}"), samples(v12, f"dc-{model}", f"afrixnli_{code}")
        if not xs or not dc:
            return None
        prior = logprobs(dc[1])
        gold = [LABELS.index(x["target"]) if isinstance(x["target"], str) else int(x["target"]) for x in xs]
        out += [float(calibrated_choice(logprobs(x), prior) == g) for x, g in zip(xs, gold)]
    return np.array(out)


def contradiction(gen_dir: Path, model: str) -> tuple[np.ndarray, dict] | None:
    """Generative recall on gold-contradiction items, and written-label shares."""
    hits, written = [], {l: 0 for l in [*LABELS, None]}
    for code in CODE.values():
        xs = samples(gen_dir, f"gen-{model}", f"afrixnli_{code}")
        if not xs:
            return None
        for x in xs:
            w = written_label(x["resps"][0][0])
            written[w] += 1
            if x["doc"]["label"] == 2:
                hits.append(float(w == "contradiction"))
    n = sum(written.values())
    return np.array(hits), {str(k): round(100 * v / n, 1) for k, v in written.items()}


def main(base: Path) -> int:
    dirs = folders(base)
    v7, v8, v9, v11, v12 = (dirs[k] for k in (7, 8, 9, 11, 12))
    out: dict = {}
    pick = json.loads((v12 / "pick.json").read_text()) if (v12 / "pick.json").exists() else None
    cand = pick and pick["candidate"]
    where = [("n-atlas", v9), ("release", v9), ("release-v12", v12), ("lowlr", v11), ("long-s1", v8), *[(m, v12) for m in MODELS]]
    if cand:
        where += [(f"{cand}-s3", v12), (f"{cand}-s4", v12)]
    scores = {m: s for m, d in where if (s := items(d, m)) and all(t in s for t in R_TASKS)}
    gen_dir = {m: (v12 if (v12 / f"iroko-gen-{m}").exists() else v11) for m in ("n-atlas", "release")} | {m: v12 for m in MODELS}
    gen = {m: g for m, d in gen_dir.items() if (g := items(d, f"gen-{m}")) and all(t in g for t in XNLI)}
    out["R"] = {m: {h or "all": ACC(pooled(s, h)) for h in ("selection", "confirmation", None)} for m, s in scores.items()}
    out["R-gen"] = {m: ACC(r_gen(gen[m], scores[m])) for m in gen if m in scores}
    for m, r in out["R"].items():
        print(f"R {m:>12}: selection {r['selection']:.1f}  confirmation {r['confirmation']:.1f}  all {r['all']:.1f}"
              + (f"  R-gen {out['R-gen'][m]:.1f}" if m in out["R-gen"] else ""))
    if "release-v12" in scores:
        same = all(np.array_equal(scores["release-v12"][t][1], scores["release"][t][1]) for t in R_TASKS)
        out["reproduction | release v12 equals v9"] = bool(same)
        print(f"reproduction | release v12 equals v9 item for item: {same}")

    # Question 1
    decision: dict = {"stage2": False, "candidate": cand}
    out["pick"] = pick
    if pick:
        print(f"stage A: {json.loads((v12 / 'rank.json').read_text())['top']}; stage B (margin {pick['margin']}): {cand}; "
              f"under 0.5: {pick['sensitivity']['candidate']}")
    if cand and cand in scores:
        half = {m: pooled(scores[m], "confirmation") for m in (cand, "release")}
        c = out["primary | candidate vs release | R confirmation"] = compare(half, [cand], ["release"], ACC, len(half[cand]))
        show(f"primary | {cand} vs release | R, confirmation half", c)
        q1 = {"recovery_R": bool(c["ci95"][0] > 0 and c["delta"] > SPREAD)}
        if cand in gen and "release" in gen:
            hg = {m: r_gen(gen[m], scores[m], "confirmation") for m in (cand, "release")}
            g = out["primary | candidate vs release | R-gen confirmation"] = compare(hg, [cand], ["release"], ACC, len(hg[cand]))
            show(f"primary | {cand} vs release | R-gen, confirmation half", g)
            q1["recovery_R_gen"] = bool(g["ci95"][0] > 0)
        q1["recovery"] = bool(q1["recovery_R"] and q1.get("recovery_R_gen", False))
        if all((v12 / f"heldout-{w}-{cand}.json").exists() for w in ("en-xx", "xx-en")):
            lows = []
            for way in ("en-xx", "xx-en"):
                files = {cand: v12 / f"heldout-{way}-{cand}.json", **{m: baseline(dirs, way, m) for m in ("release", "n-atlas")}}
                translation(out, files, [(cand, "release"), (cand, "n-atlas")], way, "held-out")
                lows += [out[f"held-out | {cand} vs release | {way} | {l}"]["chrf"]["lower95"] for l in LANGS]
            q1["translation_holds"] = bool(all(x > -MARGIN for x in lows))
            mine, rel = chars_per_token(v12 / f"heldout-en-xx-{cand}.json"), chars_per_token(v7 / "heldout-en-xx-release.json")
            q1["token_saving_holds"] = bool(all(mine[l] >= RATIO * rel[l] for l in LANGS))
        if "base" in scores:
            mg = {m: halves(scores[m], MGSM, None) for m in (cand, "base")}
            c = out["maths | candidate vs base"] = compare(mg, [cand], ["base"], ACC, len(mg[cand]))
            show(f"maths | {cand} vs base | AfriMGSM yo, ha, ig", c)
            q1["maths_holds"] = bool(c["lower95"] > -MATHS)
        decision["q1"] = q1
        decision["stage2"] = bool(all(q1.get(k) for k in ("recovery", "translation_holds", "token_saving_holds", "maths_holds")))
        print(f"decision | question 1: {q1}")

    # Secondary
    for m in [x for x in MODELS if x in scores]:
        for b in [b for b in ("release", "n-atlas", "base") if b in scores and b != m]:
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
    if "base" in scores and "lowlr" in scores:
        p = {m: pooled(scores[m]) for m in ("base", "lowlr")}
        out["seed spread | base vs lowlr | R"] = c = compare(p, ["base"], ["lowlr"], ACC, len(p["base"]))
        show("seed spread | base (seed 2) vs lowlr (seed 1) | R", c)
    dc = {m: a for m in ("n-atlas", "release", *MODELS) if (a := dc_accuracy(v9 if m in ("n-atlas", "release") else v12, v12, m)) is not None}
    out["domain-conditional"] = {m: ACC(a) for m, a in dc.items()}
    for m, a in dc.items():
        print(f"domain-conditional | {m}: AfriXNLI yo ha ig {ACC(a):.1f}")
    out["formats"] = {m: f for m, d in [("n-atlas", v9), ("release", v9), *[(m, v12) for m in MODELS]] if (f := formats(d, m))}

    # Question 2: contradiction
    con = {m: c for m, d in gen_dir.items() if (c := contradiction(d, m))}
    for m, (hits, shares) in con.items():
        out.setdefault("contradiction", {})[m] = {"recall": ACC(hits), "written": shares}
        print(f"contradiction | {m}: recall {ACC(hits):.1f} on {len(hits)} items; writes {shares}")
    for m in ("short", "anchor"):
        if m in con and "base" in con:
            h = {x: con[x][0] for x in (m, "base")}
            c = out[f"contradiction | {m} vs base"] = compare(h, [m], ["base"], ACC, len(h[m]))
            show(f"contradiction | {m} vs base | recall", c)
            decision[f"q2_restored_by_{m}"] = bool(c["ci95"][0] > 0)

    # Question 3: NFC on fresh lines
    helps = {}
    for kind, langs in (("ntrex", ["yo"]), ("flores", ["yo", "ig"])):
        for m in ("n-atlas", "release"):
            raw, nfc = v12 / f"nfc-{kind}-raw-{m}.json", v12 / f"nfc-{kind}-nfc-{m}.json"
            if not (raw.exists() and nfc.exists()):
                continue
            for l in langs:
                a, b = rows(nfc, l), rows(raw, l)
                st = {"nfc": chrf_stats(a), "raw": chrf_stats(b)}
                c = out[f"nfc | {kind} | {m} | {l}"] = compare(st, ["nfc"], ["raw"], CHRF, len(a))
                show(f"nfc | {kind} | {m} | {l} -> en chrF++ (NFC normalizer - as released)", c)
                ch = np.array([unicodedata.normalize("NFC", x["source"]) != x["source"] for x in b])
                if ch.any():
                    idx = np.flatnonzero(ch)
                    sub = {k: v[idx] for k, v in st.items()}
                    cc = out[f"nfc | {kind} | {m} | {l} | changed sources"] = compare(sub, ["nfc"], ["raw"], CHRF, len(idx))
                    show(f"nfc | {kind} | {m} | {l} | {len(idx)} changed sources", cc)
                ca, cb = nfc.with_name(f"comet-{nfc.stem}.json"), raw.with_name(f"comet-{raw.stem}.json")
                if ca.exists() and cb.exists():
                    s = {"nfc": np.array([x["comet"] for x in rows(ca, l)]), "raw": np.array([x["comet"] for x in rows(cb, l)])}
                    out[f"nfc | {kind} | {m} | {l} | AfriCOMET"] = c2 = compare(s, ["nfc"], ["raw"], ACC, len(a))
                    show(f"nfc | {kind} | {m} | {l} | AfriCOMET", c2, ".2f")
                if m == "release" and l == "yo":
                    helps[kind] = bool(c["ci95"][0] > 0)
    if len(helps) == 2:
        decision["q3_nfc_helps"] = bool(all(helps.values()))
        print(f"decision | question 3: NFC helps: {decision['q3_nfc_helps']} {helps}")

    # Stage 2
    seeds = [m for m in (cand, f"{cand}-s3", f"{cand}-s4") if cand and m in scores]
    if len(seeds) == 3 and "long-s1" in scores:
        p = {m: pooled(scores[m]) for m in (*seeds, "release", "long-s1")}
        c = out["stage 2 | candidate (3 seeds) vs release (2 seeds) | R"] = compare(p, seeds, ["release", "long-s1"], ACC, len(p["release"]))
        show("stage 2 | candidate (3 seeds) vs release (2 seeds) | R", c)

    for name in ("short-report",):
        if (v12 / f"{name}.json").exists():
            out[name] = json.loads((v12 / f"{name}.json").read_text())
    out["decision"] = decision
    print(f"decision: {decision}")
    if v12.exists():
        (v12 / "analysis.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item()) + "\n")
        (v12 / "decision.json").write_text(json.dumps(decision, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "check-dc":
        sys.exit(check_dc(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]))
    sys.exit(main(Path(sys.argv[1])))

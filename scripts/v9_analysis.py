"""v9 comparisons from docs/METHOD.md, section 4.6, on the pulled results. Sections run
only when their files have arrived, so it can be rerun as the pod works.
The release's held-out samples, AfriCOMET scores and bits per character are
v7's; every IrokoBench score is v9's own. Paired over items or sentences,
5,000 bootstrap rounds, the v7 analysis code.
    uv run --project wazocore --with sacrebleu python scripts/v9_analysis.py results/vocab/v7 results/vocab/v9
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "wazocore"))
from v7_analysis import LANGS, chrf_stats, compare, rows  # noqa: E402
from v8_analysis import ACC, CHRF, R_TASKS  # noqa: E402
from v9_pick import grid  # noqa: E402

MARGIN, SPREAD = 1.0, 2.0  # translation margin (v6 to v8); release seed spread on R (v8)
ROUNDS = 5000
CODE = {"yo": "yor", "ha": "hau", "ig": "ibo"}


def items(r: Path, model: str) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    """Per task: doc ids and scores, AfriMGSM by flexible-extract."""
    out = {}
    for f in r.glob(f"iroko-{model}/*/samples_*.jsonl"):
        task = f.name.split("_prompt_1")[0].removeprefix("samples_")
        xs = sorted((x for x in map(json.loads, f.read_text(encoding="utf-8").splitlines())
                     if x["filter"] in ("none", "flexible-extract")), key=lambda x: x["doc_id"])
        out[task] = (np.array([x["doc_id"] for x in xs]), np.array([x["exact_match" if "afrimgsm" in task else "acc"] for x in xs], dtype=float))
    return out or None


def pooled(scores: dict, which: str | None = None) -> np.ndarray:
    """R items in a fixed order; `which` keeps one half (even or odd doc ids)."""
    keep = {"selection": 0, "confirmation": 1}
    parts = []
    for t in R_TASKS:
        ids, s = scores[t]
        parts.append(s if which is None else s[ids % 2 == keep[which]])
    return np.concatenate(parts)


def boot(f, n: int) -> dict:
    """Point estimate and 95% interval of f(index array) over item resamples."""
    rng = np.random.default_rng(0)
    d = np.sort([f(rng.integers(0, n, n)) for _ in range(ROUNDS)])
    return {"delta": float(f(np.arange(n))), "ci95": [float(d[int(0.025 * ROUNDS)]), float(d[int(0.975 * ROUNDS) - 1])],
            "lower95": float(d[int(0.05 * ROUNDS)])}


def show(key: str, c: dict, f: str = ".1f") -> None:
    print(f"{key}: {c['delta']:+{f}} [{c['ci95'][0]:+{f}}, {c['ci95'][1]:+{f}}]  lower95 {c['lower95']:+{f}}", flush=True)


def translation(out: dict, files: dict[str, Path], pairs, way: str, label: str) -> None:
    """chrF++ (and AfriCOMET where scored) for each pair, per language."""
    for lang in LANGS:
        samples = {m: rows(p, lang) for m, p in files.items()}
        ids = [x["i"] for x in next(iter(samples.values()))]
        assert all([x["i"] for x in v] == ids for v in samples.values()), f"{label} {lang}: sentences differ"
        chrf = {m: chrf_stats(v) for m, v in samples.items()}
        com_files = {m: p.with_name(f"comet-{p.stem}.json") for m, p in files.items()}
        com = {m: np.array([x["comet"] for x in rows(p, lang)]) for m, p in com_files.items()} if all(p.exists() for p in com_files.values()) else None
        for a, b in pairs:
            key = f"{label} | {a} vs {b} | {way} | {lang}"
            out[key] = {"chrf": compare(chrf, [a], [b], CHRF, len(ids))}
            show(f"{key} chrF++", out[key]["chrf"], ".2f")
            if com is not None:
                out[key]["comet"] = compare(com, [a], [b], ACC, len(ids))
                show(f"{key} AfriCOMET", out[key]["comet"], ".2f")


def main(v7: Path, v9: Path) -> int:
    out: dict = {}
    names = {**grid(), **{p.replace("rel-", "ctl-").replace("release", "control-s0"): bt for p, bt in grid().items()}}
    scores = {m: s for m in ["n-atlas", *names] if (s := items(v9, m)) and all(t in s for t in R_TASKS)}

    # The grid on both halves (stage A read only the selection half)
    out["grid"] = {m: {"b_t": names.get(m), **{h: ACC(pooled(s, h)) for h in ("selection", "confirmation")}, "R": ACC(pooled(s))}
                   for m, s in scores.items()}
    for m, g in out["grid"].items():
        print(f"R {m:>14} {str(g['b_t']):>12}: selection {g['selection']:.1f}  confirmation {g['confirmation']:.1f}  all {g['R']:.1f}")

    # Reproduction: the retrained control-s0 against v7's
    if "control-s0" in scores and (old := items(v7, "control-s0")):
        n = len(pooled(old))
        out["reproduction | control-s0 v9 vs v7"] = c = compare({"v9": pooled(scores["control-s0"]), "v7": pooled(old)}, ["v9"], ["v7"], ACC, n)
        show("reproduction | control-s0 v9 vs v7 on R", c)
        if abs(c["delta"]) > SPREAD:
            print("  flagged: differs from v7 by more than 2.0; the v9 file is used (preregistered)")

    # Question 1: the candidate on the confirmation half
    pick = json.loads((v9 / "pick.json").read_text()) if (v9 / "pick.json").exists() else None
    cand = pick and pick["candidate"]
    out["pick"] = pick
    if cand and cand != "release" and cand in scores:
        half = {m: pooled(scores[m], "confirmation") for m in (cand, "release", "n-atlas")}
        n = len(half[cand])
        c = out["primary | candidate vs release, confirmation half"] = compare(half, [cand], ["release"], ACC, n)
        show(f"primary | {cand} vs release (confirmation half)", c)
        share = lambda i: (ACC(half[cand][i]) - ACC(half["release"][i])) / (ACC(half["n-atlas"][i]) - ACC(half["release"][i]))
        s = out["primary | recovered share"] = boot(share, n)
        print(f"primary | recovered share: {s['delta']:+.2f} [{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}]")
        recovery = c["ci95"][0] > 0 and c["delta"] > SPREAD
        holds = None
        if all((v9 / f"heldout-{w}-{cand}.json").exists() for w in ("en-xx", "xx-en")):
            lows = []
            for way in ("en-xx", "xx-en"):
                files = {cand: v9 / f"heldout-{way}-{cand}.json", "release": v7 / f"heldout-{way}-release.json",
                         "n-atlas": v7 / f"heldout-{way}-n-atlas.json"}
                translation(out, files, [(cand, "release"), (cand, "n-atlas")], way, "held-out")
                lows += [out[f"held-out | {cand} vs release | {way} | {l}"]["chrf"]["lower95"] for l in LANGS]
            holds = all(x > -MARGIN for x in lows)
        for t in sorted(scores[cand]):
            s3 = {m: scores[m][t][1] for m in (cand, "release", "n-atlas")}
            for b in ("release", "n-atlas"):
                out[f"secondary | {cand} vs {b} | {t}"] = c2 = compare(s3, [cand], [b], ACC, len(s3[cand])); show(f"secondary | {cand} vs {b} | {t}", c2)
        mm = {m: items(v9, f"{m}-mmlu") for m in (cand, "release", "n-atlas")}
        if all(mm.values()):
            for t in sorted(mm[cand]):
                s3 = {m: mm[m][t][1] for m in mm}
                for b in ("release", "n-atlas"):
                    out[f"secondary | {cand} vs {b} | {t}"] = c2 = compare(s3, [cand], [b], ACC, len(s3[cand])); show(f"secondary | {cand} vs {b} | {t}", c2)
        out["decision | question 1"] = {"recovery": recovery, "translation_holds": holds}
        print(f"decision | question 1: recovery {recovery}, translation holds {holds}")
    elif pick:
        out["decision | question 1"] = {"recovery": False, "translation_holds": None, "note": f"candidate {cand}"}
        print(f"decision | question 1: no candidate beyond the release ({cand})")

    # Question 2: the format probe, all R items
    probed = {m: s for m in ("n-atlas", "release", "control-s0") if (s := items(v9, f"probe-{m}"))}
    if len(probed) == 3 and all(m in scores for m in probed):
        p = {m: pooled(s) for m, s in scores.items() if m in probed} | {f"probe-{m}": pooled(s) for m, s in probed.items()}
        n = len(p["n-atlas"])
        for m in probed:
            out[f"probe | {m}"] = c = compare(p, [f"probe-{m}"], [m], ACC, n); show(f"probe | probe-{m} vs {m}", c)
        for m in ("release", "control-s0"):
            gap = lambda i, m=m: ACC(p["n-atlas"][i]) - ACC(p[m][i])
            closed = lambda i, m=m: gap(i) - (ACC(p["probe-n-atlas"][i]) - ACC(p[f"probe-{m}"][i]))
            c = out[f"probe | gap closed | {m}"] = boot(closed, n)
            show(f"probe | gap to the original closed | {m} (gap {gap(np.arange(n)):.1f})", c)
            out[f"decision | rebound | {m}"] = rebound = c["delta"] >= 0.5 * gap(np.arange(n)) and c["ci95"][0] > 0
            print(f"decision | rebound | {m}: {rebound}")

    # Question 3: unmarked input
    have = lambda m, s: (v9 / f"unmarked-l{s}-xx-en-{m}.json").exists()
    models = [m for m in ("n-atlas", "release", "control-s0") if all(have(m, s) for s in (0, 1, 2))]
    if "n-atlas" in models and len(models) > 1:
        for lang in LANGS:
            st = {(m, s): chrf_stats(rows(v9 / f"unmarked-l{s}-xx-en-{m}.json", lang)) for m in models for s in (0, 1, 2)}
            n = len(st[("n-atlas", 0)])
            drop = lambda m, s, i: CHRF(st[(m, s)][i]) - CHRF(st[(m, 0)][i])
            for s in (1, 2):
                for m in models:
                    out[f"unmarked | {m} | l{s} | xx-en | {lang}"] = c = boot(lambda i: drop(m, s, i), n)
                    show(f"unmarked | {m} | l{s} - marked | xx-en | {lang} chrF++", c, ".2f")
                for m in models[1:]:
                    out[f"unmarked | extra drop | {m} vs n-atlas | l{s} | {lang}"] = c = boot(lambda i: drop(m, s, i) - drop("n-atlas", s, i), n)
                    show(f"unmarked | extra drop | {m} vs n-atlas | l{s} | {lang} chrF++", c, ".2f")
    xnli = {m: (items(v9, f"unmarked-{m}"), scores.get(m)) for m in ("n-atlas", "release", "control-s0")}
    xnli = {m: v for m, v in xnli.items() if v[0] and v[1]}
    if "n-atlas" in xnli and len(xnli) > 1:
        for lang in LANGS:
            t = f"afrixnli_{CODE[lang]}"
            d = {m: u[t][1] - k[t][1] for m, (u, k) in xnli.items()}
            n = len(d["n-atlas"])
            for m in xnli:
                out[f"unmarked | {m} | afrixnli | {lang}"] = c = boot(lambda i, m=m: ACC(d[m][i]), n); show(f"unmarked | {m} | AfriXNLI l2 - marked | {lang}", c)
            for m in [x for x in xnli if x != "n-atlas"]:
                out[f"unmarked | extra drop | {m} vs n-atlas | afrixnli | {lang}"] = c = boot(lambda i, m=m: ACC(d[m][i]) - ACC(d["n-atlas"][i]), n)
                show(f"unmarked | extra drop | {m} vs n-atlas | AfriXNLI | {lang}", c)
    extra = {k: v for k, v in out.items() if k.startswith("unmarked | extra drop | release")}
    if extra:
        big = [k for k, v in extra.items() if v["ci95"][1] < 0 and -v["delta"] > (3.0 if "afrixnli" in k else 2.0)]
        out["decision | question 3"] = {"v10_adds_unmarked_text": bool(big), "triggered_by": big}
        print(f"decision | question 3: v10 adds unmarked text: {bool(big)} {big}")

    for name in ("audit", "drift-release", "drift-control-s0", "probe-report"):
        if (v9 / f"{name}.json").exists():
            out[name] = json.loads((v9 / f"{name}.json").read_text())
    (v9 / "analysis.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item()) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))

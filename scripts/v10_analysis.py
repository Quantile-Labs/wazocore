"""v10 comparisons from docs/METHOD.md, section 4.6, on the pulled results. Sections run
only when their files have arrived, so it can be rerun as the pod works.
IrokoBench for the release, the original and control-s0 is v9's (batch 16,
the same lm-eval); held-out samples, AfriCOMET and bits per character for
the release and the original are v7's. Paired over items or sentences,
5,000 bootstrap rounds, the v7 analysis code.

    uv run --project wazocore --with sacrebleu python scripts/v10_analysis.py results/vocab
    python -I v10_analysis.py check-null RESULTS_V10 RESULTS_V9 MODEL   # on the pod: the bug check

Writes analysis.json and decision.json into the v10 folder; the pod reads
decision.json to decide whether stage 2 runs.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "wazocore"))
sys.path.insert(0, "/workspace/wazocore")
from v7_analysis import LANGS, chrf_stats, compare, rows  # noqa: E402
from v8_analysis import ACC, R_TASKS  # noqa: E402
from v9_analysis import boot, items, pooled, show, translation  # noqa: E402
from wazocore.vx_core import calibrated_choice, last_number, loops, one_sentence  # noqa: E402

MARGIN, SPREAD, RATIO, SHIFT = 1.0, 2.0, 0.95, 2.0  # translation margin; seed spread on R; token saving; calibration margin
CODE = {"yo": "yor", "ha": "hau", "ig": "ibo"}
LABELS = ["entailment", "neutral", "contradiction"]
TOLERANCE = 0.25  # bf16 batch noise on a label log-probability (a logged deviation; 0.01 was preregistered)
ARMS = ("sdft", "para", "control-sdft")


def samples(r: Path, model: str, task: str) -> list[dict] | None:
    """lm-eval sample lines for one task, AfriMGSM by flexible-extract, by doc id."""
    f = list(r.glob(f"iroko-{model}/*/samples_{task}_prompt_1*.jsonl"))
    if len(f) != 1:
        return None
    xs = [x for x in map(json.loads, filter(None, f[0].read_text(encoding="utf-8").split("\n"))) if x["filter"] in ("none", "flexible-extract")]
    return sorted(xs, key=lambda x: x["doc_id"])


def logprobs(x: dict) -> list[float]:
    return [float(x["resps"][i][0][0]) for i in range(len(x["resps"]))]


def null_prior(v10: Path, model: str, code: str) -> tuple[list[float], list[float]] | None:
    """(the real first item's log-probabilities, the content-free ones) from the null task."""
    xs = samples(v10, f"null-{model}", f"afrixnli_{code}")
    if not xs:
        return None
    assert [x["doc_id"] for x in xs] == [0, 1] and xs[1]["doc"]["premise"] == "N/A", f"null-{model} {code}: unexpected documents"
    return logprobs(xs[0]), logprobs(xs[1])


def check_null(v10: Path, main_dir: Path, model: str) -> int:
    """The bug check: lm-eval's null task sends the first real item exactly
    as the model's own AfriXNLI run did (same document, same request
    strings), and scores it within bf16 batch noise (a logged deviation)."""
    worst, same = 0.0, True
    for code in ("yor", "hau", "ibo", "eng"):
        f = list(v10.glob(f"iroko-null-{model}/*/samples_afrixnli_{code}_prompt_1*.jsonl"))
        xs = samples(v10, f"null-{model}", f"afrixnli_{code}")
        ref = samples(main_dir, model, f"afrixnli_{code}")
        if not f or not xs or not ref:
            print(f"check-null {model} {code}: files missing", file=sys.stderr)
            return 1
        same &= xs[0]["arguments"] == ref[0]["arguments"] and xs[0]["doc"] == ref[0]["doc"]
        worst = max(worst, max(abs(a - b) for a, b in zip(logprobs(xs[0]), logprobs(ref[0]))))
    print(f"check-null {model}: requests identical {same}; largest difference {worst:.4f} (bf16 noise bound {TOLERANCE})", file=sys.stderr)
    return 0 if same and worst <= TOLERANCE else 1


def calibrated(r: Path, v10: Path, model: str, code: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Raw and content-free calibrated correctness per AfriXNLI item."""
    xs, p = samples(r, model, f"afrixnli_{code}"), null_prior(v10, model, code)
    if not xs or not p:
        return None
    gold = [LABELS.index(x["target"]) if isinstance(x["target"], str) else int(x["target"]) for x in xs]
    raw = np.array([x["acc"] for x in xs], dtype=float)
    cal = np.array([calibrated_choice(logprobs(x), p[1]) == g for x, g in zip(xs, gold)], dtype=float)
    return raw, cal


def formats(r: Path, model: str) -> dict | None:
    """v10's format diagnostics on AfriMGSM replies, per language."""
    out = {}
    for code in ("yor", "hau", "ibo", "eng"):
        xs = samples(r, model, f"afrimgsm_{code}")
        if not xs:
            return None
        reply = [x["resps"][0][0] for x in xs]
        out[code] = {"n": len(xs), "one_sentence": sum(map(one_sentence, reply)), "loops": sum(map(loops, reply)),
                     "no_answer": sum(x["filtered_resps"][0] in ("[invalid]", "") for x in xs),
                     "no_digit_number": sum(last_number(t) is None for t in reply)}
    return out


def chars_per_token(path: Path) -> dict[str, float]:
    return {l: sum(len(x["output"]) for x in rows(path, l)) / max(1, sum(x["tokens"] for x in rows(path, l))) for l in LANGS}


def baseline(dirs: dict, way: str, model: str) -> Path:
    """The release's or the original's held-out samples. X -> English comes
    from v9's level-0 unmarked run: the same lines, with sources in NFC as
    v9 and v10 give them. v7's files fed 829 Yoruba sources in another
    normalisation, worth about 6 chrF++ (a logged deviation)."""
    return dirs[9] / f"unmarked-l0-xx-en-{model}.json" if way == "xx-en" else dirs[7] / f"heldout-{way}-{model}.json"


def arm(out: dict, dirs: dict, a: str, ref: str, scores: dict) -> dict:
    """Recovery, translation and token saving for arm `a` against `ref`."""
    v7, v10 = dirs[7], dirs[10]
    res: dict = {}
    p = {m: pooled(scores[m]) for m in (a, ref, "n-atlas")}
    n = len(p[a])
    c = out[f"{a} vs {ref} | R"] = compare(p, [a], [ref], ACC, n)
    show(f"{a} vs {ref} | R", c)
    share = lambda i: (ACC(p[a][i]) - ACC(p[ref][i])) / (ACC(p["n-atlas"][i]) - ACC(p[ref][i]))
    s = out[f"{a} vs {ref} | recovered share"] = boot(share, n)
    print(f"{a} vs {ref} | recovered share: {s['delta']:+.2f} [{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}]")
    res["recovery"] = bool(c["ci95"][0] > 0 and c["delta"] > SPREAD)
    if ref == "release" and all((v10 / f"heldout-{w}-{a}.json").exists() for w in ("en-xx", "xx-en")):
        lows = []
        for way in ("en-xx", "xx-en"):
            files = {a: v10 / f"heldout-{way}-{a}.json", **{m: baseline(dirs, way, m) for m in ("release", "n-atlas")}}
            translation(out, files, [(a, "release"), (a, "n-atlas")], way, "held-out")
            lows += [out[f"held-out | {a} vs release | {way} | {l}"]["chrf"]["lower95"] for l in LANGS]
        res["translation_holds"] = bool(all(x > -MARGIN for x in lows))
        mine, rel = chars_per_token(v10 / f"heldout-en-xx-{a}.json"), chars_per_token(v7 / "heldout-en-xx-release.json")
        out[f"{a} | chars per token vs release"] = {l: mine[l] / rel[l] for l in LANGS}
        print(f"{a} | chars per token / release's: " + ", ".join(f"{l} {mine[l] / rel[l]:.3f}" for l in LANGS))
        res["token_saving_holds"] = bool(all(mine[l] >= RATIO * rel[l] for l in LANGS))
    out[f"decision | {a}"] = res
    print(f"decision | {a} vs {ref}: {res}")
    return res


def folders(base: Path) -> dict[int, Path]:
    """results/vocab/vK on the laptop, /workspace/work/results-vK on the pod."""
    return {k: base / f"v{k}" if (base / f"v{k}").exists() else base / f"results-v{k}" for k in (7, 8, 9, 10)}


def main(base: Path) -> int:
    dirs = folders(base)
    v7, v8, v9, v10 = (dirs[k] for k in (7, 8, 9, 10))
    out: dict = {}
    where = [("n-atlas", v9), ("release", v9), ("control-s0", v9), ("release-v10", v10), ("inst", v8), ("long-s1", v8),
             *[(a, v10) for a in ARMS], ("sdft-s1", v10), ("sdft-s2", v10), ("sdft-b05-t1", v10)]
    scores = {m: s for m, d in where if (s := items(d, m)) and all(t in s for t in R_TASKS)}
    for m, s in scores.items():
        print(f"R {m:>14}: {ACC(pooled(s)):.1f}")
    out["R"] = {m: ACC(pooled(s)) for m, s in scores.items()}

    # The release rescored in v10 must equal v9 item for item (the pod stops otherwise)
    if "release-v10" in scores:
        same = all(np.array_equal(scores["release-v10"][t][1], scores["release"][t][1]) for t in R_TASKS)
        out["reproduction | release v10 equals v9"] = bool(same)
        print(f"reproduction | release v10 equals v9 item for item: {same}")

    decision: dict = {"stage2": False}
    # Question 1, the primary endpoint
    if "sdft" in scores:
        q1 = arm(out, dirs, "sdft", "release", scores)
        decision["q1"] = q1
        decision["stage2"] = bool(q1.get("recovery") and q1.get("translation_holds") and q1.get("token_saving_holds"))
    # Question 2
    if "para" in scores:
        decision["q2"] = arm(out, dirs, "para", "release", scores)
        if "sdft" in scores:
            c = out["para vs sdft | R"] = compare({m: pooled(scores[m]) for m in ("para", "sdft")}, ["para"], ["sdft"], ACC, len(pooled(scores["para"])))
            show("para vs sdft | R", c)
    # Question 3
    if "control-sdft" in scores:
        decision["q3"] = arm(out, dirs, "control-sdft", "control-s0", scores)
        if "sdft" in scores:
            c = out["sdft vs control-sdft | R"] = compare({m: pooled(scores[m]) for m in ("sdft", "control-sdft")}, ["sdft"], ["control-sdft"], ACC, len(pooled(scores["sdft"])))
            show("vocabulary | sdft vs control-sdft | R (v8: inst vs control-inst -4.2)", c)
        if all((v10 / f"heldout-{w}-control-sdft.json").exists() for w in ("en-xx", "xx-en")) and all((v10 / f"heldout-{w}-sdft.json").exists() for w in ("en-xx", "xx-en")):
            for way in ("en-xx", "xx-en"):
                translation(out, {m: v10 / f"heldout-{way}-{m}.json" for m in ("sdft", "control-sdft")}, [("sdft", "control-sdft")], way, "vocabulary")

    # Question 4: content-free calibration, pooled over yo, ha and ig
    cal = {}
    for m, d in [("n-atlas", v9), ("release", v9), ("control-s0", v9), *[(a, v10) for a in ARMS]]:
        per = [calibrated(d, v10, m, CODE[l]) for l in LANGS]
        if all(per):
            cal[m] = (np.concatenate([x[0] for x in per]), np.concatenate([x[1] for x in per]))
            print(f"calibration | {m}: raw {ACC(cal[m][0]):.1f}, calibrated {ACC(cal[m][1]):.1f}")
    out["calibration"] = {m: {"raw": ACC(r), "calibrated": ACC(c)} for m, (r, c) in cal.items()}
    if "n-atlas" in cal and "release" in cal:
        n = len(cal["n-atlas"][0])
        raw = out["calibration | release vs n-atlas | raw"] = compare({m: cal[m][0] for m in cal}, ["release"], ["n-atlas"], ACC, n)
        c = out["calibration | release vs n-atlas | calibrated"] = compare({m: cal[m][1] for m in cal}, ["release"], ["n-atlas"], ACC, n)
        show("calibration | release vs n-atlas | raw", raw)
        show("calibration | release vs n-atlas | calibrated", c)
        shift = bool(c["lower95"] > -SHIFT and raw["ci95"][1] < 0)
        decision["q4_preference_shift"] = shift
        print(f"decision | question 4: preference shift {shift}")
        for m in [x for x in cal if x not in ("n-atlas", "release")]:
            for k, i in (("raw", 0), ("calibrated", 1)):
                out[f"calibration | {m} vs n-atlas | {k}"] = c = compare({x: cal[x][i] for x in cal}, [m], ["n-atlas"], ACC, n)
                show(f"calibration | {m} vs n-atlas | {k}", c)

    # Secondary: per task, AfriMMLU, format diagnostics, bits per character, against v8's inst
    for a in [x for x in ARMS if x in scores]:
        refs = ["release", "n-atlas"] + (["control-s0"] if a == "control-sdft" else []) + (["inst"] if a == "sdft" and "inst" in scores else [])
        for t in sorted(scores[a]):
            for b in [b for b in refs if t in scores[b]]:
                s = {m: scores[m][t][1] for m in (a, b)}
                out[f"secondary | {a} vs {b} | {t}"] = c = compare(s, [a], [b], ACC, len(s[a]))
                show(f"secondary | {a} vs {b} | {t}", c)
        if "inst" in refs:
            c = out["secondary | sdft vs inst | R"] = compare({m: pooled(scores[m]) for m in ("sdft", "inst")}, ["sdft"], ["inst"], ACC, len(pooled(scores["sdft"])))
            show("secondary | sdft vs inst | R (different runs)", c)
    mmlu = {m: s for m in ("n-atlas", "release", *ARMS) if (s := items(v10, f"{m}-mmlu") if m in ("n-atlas", "release") else items(v10, m))}
    for a in [x for x in ARMS if x in mmlu]:
        for t in [t for t in sorted(mmlu[a]) if "afrimmlu" in t]:
            for b in [b for b in ("release", "n-atlas") if b in mmlu and t in mmlu[b]]:
                s = {a: mmlu[a][t][1], b: mmlu[b][t][1]}
                out[f"secondary | {a} vs {b} | {t}"] = c = compare(s, [a], [b], ACC, len(s[a]))
                show(f"secondary | {a} vs {b} | {t}", c)
    out["formats"] = {m: f for m, d in [("n-atlas", v9), ("release", v9), ("control-s0", v9), *[(a, v10) for a in ARMS]] if (f := formats(d, m))}
    for m, f in out["formats"].items():
        print(f"formats | {m}: " + "; ".join(f"{c} one-sentence {v['one_sentence']}, no answer {v['no_answer']}, loops {v['loops']}" for c, v in f.items()))
    for m in ARMS:
        if (v10 / f"bpc-{m}.json").exists():
            out[f"bpc | {m}"] = json.loads((v10 / f"bpc-{m}.json").read_text())

    # Exploratory: sdft with its bottom layers halfway back
    if "sdft-b05-t1" in scores and "sdft" in scores:
        c = out["explore | sdft-b05-t1 vs sdft | R"] = compare({m: pooled(scores[m]) for m in ("sdft-b05-t1", "sdft")}, ["sdft-b05-t1"], ["sdft"], ACC, len(pooled(scores["sdft"])))
        show("explore | sdft-b05-t1 vs sdft | R", c)

    # Stage 2: three sdft seeds against release seeds 0 and 1
    seeds = [m for m in ("sdft", "sdft-s1", "sdft-s2") if m in scores]
    if len(seeds) == 3 and "long-s1" in scores:
        p = {m: pooled(scores[m]) for m in (*seeds, "release", "long-s1")}
        c = out["stage 2 | sdft (3 seeds) vs release (2 seeds) | R"] = compare(p, seeds, ["release", "long-s1"], ACC, len(p["sdft"]))
        show("stage 2 | sdft (3 seeds) vs release (2 seeds) | R", c)

    for name in ("sdft-report", "para-report", "aya-check"):
        if (v10 / f"{name}.json").exists():
            out[name] = json.loads((v10 / f"{name}.json").read_text())
    out["decision"] = decision
    print(f"decision: {decision}")
    if v10.exists():
        (v10 / "analysis.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item()) + "\n")
        (v10 / "decision.json").write_text(json.dumps(decision, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "check-null":
        sys.exit(check_null(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]))
    sys.exit(main(Path(sys.argv[1])))

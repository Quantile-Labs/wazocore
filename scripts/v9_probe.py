"""v9 format-probe data, as fixed in docs/METHOD.md, section 4.6: 200 MNLI and 200 GSM8K
training items in English, in IrokoBench's prompt-1 format, drawn in a fixed
shuffled order (seed 0) after dropping any item that shares a 13-word
sequence with an IrokoBench item in any of the four languages.

On the pod, with venv-eval:
  python -I scripts/v9_probe.py --out probe.jsonl --report probe-report.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "wazocore"))
sys.path.insert(0, "/workspace/wazocore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/workspace")
from wazocore.vx_core import probe_messages, word_ngrams  # noqa: E402
from v8_aya import IROKO, LANGUAGES, N, strings  # noqa: E402

SOURCES = {"nli": ("nyu-mll/multi_nli", None, "da70db2af9d09693783c3320c4249840212ee221"),
           "math": ("openai/gsm8k", "main", "740312add88f781978c0658806c59bc2815b9866")}
PER_KIND = 200


def iroko_ngrams() -> set:
    from datasets import load_dataset

    return {g for name, rev in IROKO.items() for config in LANGUAGES.values()
            for split in load_dataset(name, config, revision=rev).values()
            for row in split for text in strings(row) for g in word_ngrams(text, N)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args(argv)

    from datasets import load_dataset

    grams, convs = iroko_ngrams(), []
    report: dict = {"sources": SOURCES, "irokobench_revisions": IROKO, "ngram": N, "per_kind": PER_KIND}
    for kind, (name, config, rev) in SOURCES.items():
        rows = list(load_dataset(name, config, revision=rev, split="train"))
        order = list(range(len(rows)))
        random.Random(0).shuffle(order)
        dropped, kept = Counter(), []
        for i in order:
            m = probe_messages(kind, rows[i])
            if m is None:
                dropped["unlabelled"] += 1
            elif any(word_ngrams(x["content"], N) & grams for x in m):
                dropped["irokobench"] += 1
            else:
                kept.append(m)
            if len(kept) == PER_KIND:
                break
        report[kind] = {"rows": len(rows), "examined": sum(dropped.values()) + len(kept), "dropped": dict(dropped), "kept": len(kept)}
        convs += kept
    random.Random(0).shuffle(convs)
    tmp = args.out.with_suffix(".part")
    tmp.write_text("".join(json.dumps({"messages": c}, ensure_ascii=False) + "\n" for c in convs), encoding="utf-8")
    tmp.replace(args.out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

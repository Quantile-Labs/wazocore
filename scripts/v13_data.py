"""v13 data, as fixed in docs/METHOD.md, section 4.6: v10's self-distilled file split
into its maths and chat halves. A row is translation if it is in v6's gold
file, maths if its user turn is one of v10's maths prompts, chat if it is
one of v10's Aya prompts; a row matching none or more than one stops the
build. Each output keeps the translation rows and one half, in v10's
order, byte for byte.

On the pod, with venv-eval:
  python -I scripts/v13_data.py --sdft chat-v10-sdft.jsonl --gold chat-v6-gold.jsonl --v10 V10_DATA_DIR \
      --vx VX --base N_ATLAS --out DIR --report split-report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "wazocore"))
sys.path.insert(0, "/workspace/wazocore")
from v8_aya import chat_ids, consumed_share  # noqa: E402
from v10_data import jsonl, lines  # noqa: E402


def classify(rows: list[str], gold: set[str], maths: set[str], chat: set[str]) -> list[str]:
    """'gold', 'maths' or 'chat' for each row of the self-distilled file."""
    kinds = []
    for row in rows:
        if row in gold:
            kinds.append("gold")
            continue
        prompt = json.loads(row)["messages"][0]["content"]
        hit = [k for k, s in (("maths", maths), ("chat", chat)) if prompt in s]
        if len(hit) != 1:
            raise SystemExit(f"row matches {hit or 'no'} self-distilled source: {prompt[:80]!r}")
        kinds.append(hit[0])
    return kinds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for k in ("sdft", "gold", "v10", "vx", "base", "out", "report"):
        ap.add_argument(f"--{k}", type=Path, required=True)
    args = ap.parse_args(argv)
    from transformers import AutoTokenizer

    rows = lines(args.sdft)
    maths = {p["prompt"] for p in jsonl(args.v10 / "answer.jsonl")}
    chat = {p["prompt"] for p in jsonl(args.v10 / "aya.jsonl")}
    kinds = classify(rows, set(lines(args.gold)), maths, chat)
    report: dict = {"rows": dict(Counter(kinds)), "distinct": {k: len({r for r, x in zip(rows, kinds) if x == k}) for k in ("gold", "maths", "chat")}}
    vx, orig = AutoTokenizer.from_pretrained(args.vx), AutoTokenizer.from_pretrained(args.base)
    tok = Counter()
    for r, k in zip(rows, kinds):
        tok[k] += len(chat_ids(vx, json.loads(r)["messages"]))
    report["tokens_release_tokenizer"] = dict(tok)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, drop in (("no-chat", "chat"), ("no-math", "maths")):
        kept = [(r, k) for r, k in zip(rows, kinds) if k != drop]
        tmp = args.out / f"chat-v13-{name}.part"
        tmp.write_text("\n".join(r for r, _ in kept) + "\n", encoding="utf-8")
        tmp.replace(args.out / f"chat-v13-{name}.jsonl")
        convs, flags = [json.loads(r)["messages"] for r, _ in kept], [k != "gold" for _, k in kept]
        report[name] = {"rows": len(kept), "consumed": {"release_tokenizer": consumed_share(vx, convs, flags),
                                                        "original_tokenizer": consumed_share(orig, convs, flags)}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

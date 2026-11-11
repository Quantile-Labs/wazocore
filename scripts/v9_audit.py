"""v9 exploratory audit (docs/METHOD.md, section 4.6): what the 15,000 new tokens are
made of, the share of new-token use each group carries on the release's
held-out replies, and tokens per word on FLORES-200 devtest when marks are
stripped, for both tokenizers. CPU only.

  python -I scripts/v9_audit.py --vx VX_TOK --base N_ATLAS --flores F \
      --replies results-v7/heldout-en-xx-release.json --out audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "wazocore"))
sys.path.insert(0, "/workspace/wazocore")
from wazocore.vx_core import byte_decoder, strip_marks, token_groups  # noqa: E402

FLORES = {"yo": "yor_Latn", "ha": "hau_Latn", "ig": "ibo_Latn"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vx", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--flores", type=Path, required=True)
    ap.add_argument("--replies", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    from tokenizers import Tokenizer

    new = json.loads((args.vx / "new_tokens.json").read_text(encoding="utf-8"))
    bd = byte_decoder()
    groups = {t["id"]: token_groups(bytes(bd[c] for c in t["token"])) for t in new["tokens"]}
    names = list(next(iter(groups.values())))
    vx, base = (Tokenizer.from_file(str(d / "tokenizer.json")) for d in (args.vx, args.base))
    use = Counter(i for r in json.loads(args.replies.read_text(encoding="utf-8"))["rows"]
                  for i in vx.encode(r["output"], add_special_tokens=False).ids if i in groups)
    total = sum(use.values())
    report = {"new_tokens": len(groups), "uses_on_replies": total,
              "groups": {g: {"tokens": sum(v[g] for v in groups.values()),
                             "share_of_use": sum(n for i, n in use.items() if groups[i][g]) / max(1, total)} for g in names}}
    tpw = {}
    for lang, code in FLORES.items():
        lines = (args.flores / "devtest" / f"{code}.devtest").read_text(encoding="utf-8").splitlines()
        for level in (0, 1, 2):
            text = [strip_marks(l, lang, level) for l in lines]
            words = sum(len(l.split()) for l in text)
            tpw[f"{lang}-l{level}"] = {name: sum(len(t.encode(l, add_special_tokens=False).ids) for l in text) / words
                                       for name, t in (("n-atlas", base), ("wazocore", vx))}
    report["tokens_per_word"] = tpw
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

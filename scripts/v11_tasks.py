"""v11 evaluation inputs (docs/METHOD.md, section 4.6).

  tasks      lm-eval task files that include the installed prompt-1 tasks
             unchanged and override one thing each:
               afrixnli_<code>_prompt_1_gen   answered by generation (32 tokens,
                                              greedy) and scored by the label the
                                              reply writes (vx_core.written_label)
               <task>_nfc                     documents in NFC (Yoruba, Igbo)
  tokenizer  a copy of a model folder whose tokenizer normalises to NFC;
             the weights are linked, not copied

On the pod, with venv-eval:
  python -I scripts/v11_tasks.py tasks --out TASKS_DIR --wazocore /workspace/wazocore
  python -I scripts/v11_tasks.py tokenizer --model RELEASE --out RELEASE_NFC
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GEN = ("yor", "hau", "ibo", "eng")
NFC = ("yor", "ibo")
SCORE = '''import sys

sys.path.insert(0, {wazocore!r})
from wazocore.vx_core import nfc_strings, written_label


def process(doc, results):
    gold = {{0: "entailment", 1: "neutral", 2: "contradiction"}}[doc["label"]]
    label = written_label(results[0])
    return {{"acc": float(label == gold), "no_label": float(label is None)}}


def nfc(ds):
    # no cache: a cached map is keyed by this function, not by nfc_strings
    return ds.map(nfc_strings, load_from_cache_file=False)
'''


def source(root: Path, family: str, code: str) -> tuple[Path, str]:
    """The installed prompt-1 task file for one family and language, and its task name."""
    d = root / family / "direct" / "prompt_1"
    name = f"afrimmlu_direct_{code}" if family == "afrimmlu" else f"{family}_{code}"
    return d / f"{name}.yaml", f"{name}_prompt_1"


def no_process_docs(task: Path) -> None:
    texts = [task.read_text(), *(p.read_text() for p in task.parent.iterdir() if p.suffix == "" and p.is_file())]
    if any("process_docs" in t for t in texts):
        raise SystemExit(f"{task} or its base has its own process_docs: v11's override would replace it")


def step_tasks(args) -> int:
    import lm_eval

    root = Path(lm_eval.__file__).parent / "tasks"
    args.out.mkdir(parents=True, exist_ok=True)
    n = 0
    for code in GEN:
        task, name = source(root, "afrixnli", code)
        (args.out / f"{name}_gen.yaml").write_text(
            f"include: {task}\ntask: {name}_gen\ntag: v11_gen\noutput_type: generate_until\ndoc_to_choice: null\n"
            "generation_kwargs:\n  do_sample: false\n  max_gen_toks: 32\n  until:\n    - 'Premise:'\n"
            "filter_list:\n  - name: none\n    filter:\n      - function: take_first\n"
            "process_results: !function score_v11.process\n"
            "metric_list:\n  - metric: acc\n    aggregation: mean\n    higher_is_better: true\n"
            "  - metric: no_label\n    aggregation: mean\n    higher_is_better: false\n")
        n += 1
    for family in ("afrixnli", "afrimgsm", "afrimmlu"):
        for code in NFC:
            task, name = source(root, family, code)
            no_process_docs(task)
            (args.out / f"{name}_nfc.yaml").write_text(
                f"include: {task}\ntask: {name}_nfc\ntag: v11_nfc\nprocess_docs: !function score_v11.nfc\n")
            n += 1
    (args.out / "score_v11.py").write_text(SCORE.format(wazocore=args.wazocore))
    print(f"wrote {n} tasks to {args.out}", file=sys.stderr)
    return 0


def step_tokenizer(args) -> int:
    """The model folder with tokenizer.json's normalizer set to NFC (or NFC
    before an existing one); every other file is linked."""
    args.out.mkdir(parents=True, exist_ok=True)
    for f in args.model.iterdir():
        if f.is_file() and f.name != "tokenizer.json" and not (args.out / f.name).exists():
            (args.out / f.name).symlink_to(f.resolve())
    tok = json.loads((args.model / "tokenizer.json").read_text(encoding="utf-8"))
    old = tok.get("normalizer")
    tok["normalizer"] = {"type": "NFC"} if old is None else {"type": "Sequence", "normalizers": [{"type": "NFC"}, old]}
    (args.out / "tokenizer.json").write_text(json.dumps(tok, ensure_ascii=False), encoding="utf-8")
    print(f"{args.out}: normalizer {json.dumps(tok['normalizer'])}", file=sys.stderr)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="step", required=True)
    t = sub.add_parser("tasks")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--wazocore", required=True, help="folder holding the wazocore package")
    k = sub.add_parser("tokenizer")
    k.add_argument("--model", type=Path, required=True)
    k.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    return {"tasks": step_tasks, "tokenizer": step_tokenizer}[args.step](args)


if __name__ == "__main__":
    sys.exit(main())

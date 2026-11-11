"""v10 content-free AfriXNLI (docs/METHOD.md, section 4.6, question 4): lm-eval task
files that include the installed prompt-1 task unchanged and keep two
documents, the first test item as it is and the same item with premise
and hypothesis both "N/A". lm-eval scores both exactly as it scores the
real task. The first must match the model's own AfriXNLI run (the bug
check); the second gives each label's content-free log-probability.

On the pod, with venv-eval:
  python -I scripts/v10_tasks.py --out TASKS_DIR
then lm_eval --include_path TASKS_DIR --tasks afrixnli_yor_prompt_1_null,...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODES = ("yor", "hau", "ibo", "eng")
NULL = '''from datasets import concatenate_datasets


def _null(ds):
    first = ds.select([0])
    return concatenate_datasets([first, first.map(lambda d: {"premise": "N/A", "hypothesis": "N/A"})])


yor = hau = ibo = eng = _null
'''


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    import lm_eval

    src = Path(lm_eval.__file__).parent / "tasks/afrixnli/direct/prompt_1"
    args.out.mkdir(parents=True, exist_ok=True)
    for code in CODES:
        task = src / f"afrixnli_{code}.yaml"
        if "process_docs" in task.read_text() or "process_docs" in (src / "afrixnli_yaml").read_text():
            raise SystemExit(f"{task} has its own process_docs: v10's override would replace it")
        (args.out / f"afrixnli_{code}_null.yaml").write_text(
            f"include: {task}\ntask: afrixnli_{code}_prompt_1_null\ntag: v10_null\n"
            f"process_docs: !function null_v10.{code}\n")
    (args.out / "null_v10.py").write_text(NULL)
    print(f"wrote {len(CODES)} tasks to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

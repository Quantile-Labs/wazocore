"""v12 domain-conditional AfriXNLI (docs/METHOD.md, section 4.6): lm-eval task files that
include the installed prompt-1 task unchanged and keep two documents, the
first test item as it is and the same item with empty premise and
hypothesis, the domain premise of domain-conditional PMI (Surface Form
Competition, EMNLP 2021). The first must score as the model's own AfriXNLI
run did (v10's bug check); the second gives each label's domain
log-probability. Generative AfriXNLI and the NFC tokenizer come from
scripts/v11_tasks.py.

On the pod, with venv-eval:
  python -I scripts/v12_tasks.py --out TASKS_DIR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODES = ("yor", "hau", "ibo", "eng")
DC = '''from datasets import concatenate_datasets


def _dc(ds):
    first = ds.select([0])
    return concatenate_datasets([first, first.map(lambda d: {"premise": "", "hypothesis": ""}, load_from_cache_file=False)])


yor = hau = ibo = eng = _dc
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
            raise SystemExit(f"{task} has its own process_docs: v12's override would replace it")
        (args.out / f"afrixnli_{code}_dc.yaml").write_text(
            f"include: {task}\ntask: afrixnli_{code}_prompt_1_dc\ntag: v12_dc\nprocess_docs: !function dc_v12.{code}\n")
    (args.out / "dc_v12.py").write_text(DC)
    print(f"wrote {len(CODES)} tasks to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

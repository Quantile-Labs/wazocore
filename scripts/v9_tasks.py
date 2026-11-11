"""v9 unmarked AfriXNLI: lm-eval task files that include the installed
prompt-1 task unchanged and pass each document's premise and hypothesis
through vx_core.strip_marks at level 2 (docs/METHOD.md, section 4.6, question 3).

On the pod, with venv-eval:
  python -I scripts/v9_tasks.py --out TASKS_DIR --wazocore /workspace/wazocore
then lm_eval --include_path TASKS_DIR --tasks afrixnli_yor_prompt_1_unmarked,...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

LANGS = {"yor": "yo", "hau": "ha", "ibo": "ig"}
STRIP = '''import sys

sys.path.insert(0, {wazocore!r})
from wazocore.vx_core import strip_marks


def _strip(lang):
    return lambda ds: ds.map(lambda d: {{"premise": strip_marks(d["premise"], lang, 2),
                                         "hypothesis": strip_marks(d["hypothesis"], lang, 2)}})


{assign}
'''


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--wazocore", required=True, help="folder holding the wazocore package")
    args = ap.parse_args(argv)

    import lm_eval

    src = Path(lm_eval.__file__).parent / "tasks/afrixnli/direct/prompt_1"
    args.out.mkdir(parents=True, exist_ok=True)
    for code in LANGS:
        task = src / f"afrixnli_{code}.yaml"
        text = task.read_text()
        if "process_docs" in text or "process_docs" in (src / "afrixnli_yaml").read_text():
            raise SystemExit(f"{task} has its own process_docs: v9's override would replace it")
        (args.out / f"afrixnli_{code}_unmarked.yaml").write_text(
            f"include: {task}\ntask: afrixnli_{code}_prompt_1_unmarked\ntag: v9_unmarked\n"
            f"process_docs: !function strip_v9.{code}\n")
    assign = "\n".join(f"{c} = _strip({l!r})" for c, l in LANGS.items())
    (args.out / "strip_v9.py").write_text(STRIP.format(wazocore=args.wazocore, assign=assign))
    print(f"wrote {len(LANGS)} tasks to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

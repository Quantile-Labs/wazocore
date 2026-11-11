"""v13 stages A and B (docs/METHOD.md, section 4.6): v12's pick for `no-chat` and
`no-math`, ties to `no-chat`, margin 1.0.

  python -I scripts/v13_pick.py rank RESULTS
  python -I scripts/v13_pick.py choose RESULTS
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v12_pick  # noqa: E402

v12_pick.MODELS = ("no-chat", "no-math")

if __name__ == "__main__":
    sys.exit(v12_pick.main(sys.argv[1:]))

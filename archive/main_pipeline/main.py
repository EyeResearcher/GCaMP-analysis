
"""Legacy entry point delegating to the modular pipeline package."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.main import main


if __name__ == "__main__":  # pragma: no cover
    main()
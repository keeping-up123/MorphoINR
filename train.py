"""Convenience entry point for the MorphoINR two-stage training pipeline."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from train import main


if __name__ == "__main__":
    main()

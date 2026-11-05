import sys
from pathlib import Path

# pipeline/ scripts import each other as top-level modules.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

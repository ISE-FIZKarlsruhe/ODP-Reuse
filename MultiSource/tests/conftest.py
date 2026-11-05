import sys
from pathlib import Path

# The harvester scripts are plain modules in MultiSource/, not a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

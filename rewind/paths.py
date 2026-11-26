"""Where things live, relative to the repository root, so code and notebooks work from any folder.

Notebooks still let environment variables override these (REWIND_DATA_ROOT, REWIND_OUT_ROOT, REWIND_SELECTION)."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]      # the repository folder
OUTPUTS = REPO / "outputs"                      # git-ignored: every result
BATCH = OUTPUTS / "batch"                       # the batch run: <dataset>/<video>/, batch_summary.csv, search/
SELECTION = REPO / "selection.csv"              # which videos the batch runs on (made on the first run)
DATA_ROOT = Path("/mnt/5T_storage/clipsExtraction/datasets")   # MEVA + UCA on the team VM

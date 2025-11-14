import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "labels"

fn = PROJECT_ROOT / "data" / "labels" / "labels_manifest_1000_1.csv"


out = PROJECT_ROOT / "data" / "labels" / "labels_manifest_1000.csv"
df = pd.read_csv(fn)
df["filepath"] = [str(i).replace("/d", "d") for i in df["filepath"]]
df["filepath"] = [str(i).replace("\\", "/") for i in df["filepath"]]

df.to_csv(out, index=False)
print("Donezo!", out)
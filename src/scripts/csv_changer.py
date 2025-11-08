import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "labels"

fn = PROJECT_ROOT / "data" / "processed" / "manifest_with_labels.csv"


out = PROJECT_ROOT / "data" / "processed" / "manifest_with_labels.csv"
df = pd.read_csv(fn)

df["filepath"] = [str(i).replace("[redacted]", "") for i in df["filepath_PC"]]
df.drop("filepath_PC", axis=1, inplace=True)
df.drop("filepath_laptop", axis=1, inplace=True)

df.to_csv(out, index=False)
print("Donezo!", out)
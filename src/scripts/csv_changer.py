import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "labels"

fn = DATA_PATH / "labels_manifest_1000.csv"


out = DATA_PATH / "labels_manifest_1000_new.csv"
df = pd.read_csv(fn)

#df["filepath"] = [i.replace("[redacted]", "") for i in df["filepath_PC"]]
#df.drop("filepath_PC", axis=1, inplace=True)
#df.drop("filepath_laptop", axis=1, inplace=True)

df.to_csv(out, index=False)
print("Donezo!", out)
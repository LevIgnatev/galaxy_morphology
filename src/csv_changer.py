import pandas as pd
fn = r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\processed\manifest_with_labels.csv"
df = pd.read_csv(fn)
df["filepath_PC"] = df["filepath"]
df["filepath_laptop"] = df["filepath"].str.replace(r"user\PycharmProjects\galaxy_morphology_ml_captioning", r"79263\galaxy_morphology_ml_captioning", regex=False)
out = fn
df.to_csv(out, index=False)
print("Wrote", out)
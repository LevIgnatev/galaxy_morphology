import pandas as pd
fn = r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\processed\manifest_with_labels.csv"
df = pd.read_csv(fn)
#df["filepath_PC"] = df["filepath"]
df.drop('filepath', axis=1, inplace=True)
out = fn
df.to_csv(out, index=False)
print("Wrote", out)
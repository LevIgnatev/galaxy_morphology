import pandas as pd
from sklearn.model_selection import train_test_split
fn = r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\processed\manifest_with_labels.csv"
fn1 = r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\processed\manifest_train_and_val.csv"
fn2 = r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\processed\manifest_final_test.csv"
df = pd.read_csv(fn)

normal_df, test_df = train_test_split(df,
                                          test_size=0.1,
                                          stratify=df["derived_label"],
                                          random_state=37)
out = fn1
out2 = fn2
normal_df.to_csv(out, index=False)
test_df.to_csv(out2, index=False)
print("Wrote", out, out2)
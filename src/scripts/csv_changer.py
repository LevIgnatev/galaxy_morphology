import pandas as pd
from sklearn.model_selection import train_test_split
fn = r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\processed\captions_full\captions_full.csv"
df = pd.read_csv(fn)
train_val_df, test_df = train_test_split(df,
                                          train_size=0.9,
                                          test_size=0.1,
                                          random_state=37)
train_df, val_df = train_test_split(
    train_val_df,
    train_size=7/9,
    test_size=2/9,
    random_state=37
)
out1 = r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\processed\captions_full\train_captions.txt"
out2 = r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\processed\captions_full\val_captions.txt"
out3 = r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\processed\captions_full\test_captions.txt"
val_df["objid"].to_csv(out2, index=False, header=False)
test_df["objid"].to_csv(out3, index=False, header=False)
train_df["objid"].to_csv(out1, index=False, header=False)
print("Wrote", out1)

#fn = r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\processed\captions_full\captions_full.csv"
#df = pd.read_csv(fn)
#df["filepath"] = [i.replace("C:\\Users\\user\\PycharmProjects\\galaxy_morphology_ml_captioning\\", "") for i in df["filepath_PC"]]
#df.drop("filepath_PC", axis=1, inplace=True)
#out = r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\processed\captions_full\captions_new2.csv"
#df.to_csv(out, index=False)
#print("Wrote", out)
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report, confusion_matrix

from infer import predict_class

PROJECT_ROOT = Path(__file__).parents[2]

labels = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "manifest_train_and_val.csv")["derived_label"].astype(
    str).dropna().unique().tolist()

df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "manifest_final_test.csv")
predicted_labels = []
real_labels = []

for _, row in df.iterrows():
    image_path = str(PROJECT_ROOT / row["filepath"])
    predicted_labels.append(predict_class(image_path))
    real_labels.append(row["derived_label"])

acc = accuracy_score(real_labels, predicted_labels)
bal_acc = balanced_accuracy_score(real_labels, predicted_labels)
f1_acc = f1_score(real_labels, predicted_labels, average="macro")
weighted_f1 = f1_score(real_labels, predicted_labels, average="weighted")
print(f"Accuracy: {acc}")
print(f"Balanced accuracy: {bal_acc}")
print(f"F1 accuracy: {f1_acc}")
print(f"Weighted F1 accuracy: {weighted_f1}")
print("Classification report:")
print(classification_report(real_labels, predicted_labels, zero_division=0))

con_mat = confusion_matrix(real_labels, predicted_labels)
con_mat_df = pd.DataFrame(con_mat, index=labels, columns=labels)
print("Confusion matrix:")
print(con_mat_df)


plt.figure()
plt.imshow(con_mat, interpolation="nearest")
plt.xticks(range(len(labels)), labels)
plt.yticks(range(len(labels)), labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.colorbar()
plt.show()

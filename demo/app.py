import tensorflow as tf
import streamlit as st
from PIL import Image
import os, sys
import numpy as np, pandas as pd
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.gradcam_visualizer_AI_gen import generate_gradcam_overlay_from_array

st.set_page_config(page_title="Galaxy morphology demo", layout="centered")
st.title("Galaxy morphology demo")

ROOT = Path(__file__).resolve().parents[1]
classifier_path = Path(os.getenv("model_path", ROOT / "checkpoints" / "ckpt_classifier_full.keras"))
manifest_path = Path(os.getenv("manifest_path", ROOT / "data" / "labels" / "labels_manifest_1000.csv"))

@st.cache_resource
def load_model_and_classes(model_path):
    class_names = pd.read_csv(manifest_path)["derived_label"].unique().tolist()
    model = tf.keras.models.load_model(model_path, compile=False)
    return model, class_names

def predict_and_gradcam(model_path, image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.array(img)

    heatmap, probabilities = generate_gradcam_overlay_from_array(model_path, arr)

    ordered_probs = np.argsort(-probabilities)
    probs_list = [(class_names[i], probabilities[i]) for i in ordered_probs]

    return heatmap, probs_list

with st.spinner("Loading model..."):
    classifier_model, class_names = load_model_and_classes(classifier_path)

st.success(f"Model loaded successfully!")

images_list = ["None",
                r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\raw\images\125119.jpg",
                r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\raw\images\42.jpg",
                r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\raw\images\107682.jpg",
                r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\raw\images\192211.jpg",
                r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\labels\tomato.jpg",
                r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\raw\images\192973.jpg",
                r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\raw\images\226130.jpg",
                r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\raw\images\226162.jpg"
]

true_labels = {r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\raw\images\125119.jpg": "merger",
                r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\raw\images\42.jpg": "spiral",
                r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\raw\images\107682.jpg": "spiral",
                r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\raw\images\192211.jpg": "elliptical",
                r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\labels\tomato.jpg": ":)",
                r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\raw\images\192973.jpg": "edge-on",
                r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\raw\images\226130.jpg": "edge-on",
                r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\raw\images\226162.jpg": "spiral"
}

images_list_choice = ["None",
                      "1. A couple merging galaxies",
                      "2. A Beautiful spiral galaxy",
                      "3. A galaxy with distinctive arms",
                      "4. Blurry elliptical galaxy",
                      "5. A tomato",
                      "6. Edge-on galaxy",
                      "7. Blue edge-on galaxy",
                      "8. Galaxy with a huge bar"
]

col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
with col2:
    choice = images_list[images_list_choice.index(st.selectbox("Or choose from one of the sample galaxies:", images_list_choice, index=0))]

image_path = None
if uploaded is not None:
    temp_images_dir = r"C:\Users\user\PycharmProjects\galaxy_morphology_ml_captioning\data\labels\uploaded_images"
    os.makedirs(temp_images_dir, exist_ok=True)
    image_path = os.path.join(temp_images_dir, uploaded.name)
    with open(image_path, "wb") as f:
        f.write(uploaded.getbuffer())
elif choice != "None":
    image_path = choice

if image_path:
    st.image(image_path, use_container_width=True, caption="Selected image")

    if st.button("Predict and show attention map"):
        with st.spinner("Running model and Grad-CAM..."):
            heatmap, probs_list = predict_and_gradcam(classifier_path, image_path)
        st.write("#### Predictions:")
        for predicted_class, probability in probs_list:
            st.write(f"- {predicted_class}: {probability:.4f}")

        st.write(f"### **Verdict:** {probs_list[0][0]}")

        if image_path in true_labels:
            st.write(f"### **True label:** {true_labels[image_path]}")

        st.write("---")
        st.write("**Grad-CAM overlay:**")
        st.image(heatmap, caption="Grad-CAM heat map (redder - more attention)", use_container_width=True)
else:
    st.info("Select a sample or upload an image to begin.")
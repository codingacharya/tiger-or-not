import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image

# ---------------------------
# Load trained model
# ---------------------------
MODEL_PATH = "tiger_footprint_model.h5"
model = load_model(MODEL_PATH)

# ---------------------------
# App title
# ---------------------------
st.set_page_config(page_title="Tiger Footprint Classifier", page_icon="🐅")
st.title("🐾 Tiger Footprint Classifier")
st.write("Upload a footprint image to check if it belongs to a **Tiger**.")

# ---------------------------
# Prediction function
# ---------------------------
def predict_footprint(img: Image.Image) -> float:
    img = img.resize((224, 224))  # same size as training
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    return prediction

# ---------------------------
# File uploader
# ---------------------------
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Run prediction
    with st.spinner("🔍 Analyzing footprint..."):
        score = predict_footprint(img)

        st.subheader("Prediction Result")
        if score > 0.5:
            st.success(f"✅ This looks like a **Tiger footprint** (confidence: {score:.2f})")
        else:
            st.error(f"❌ This is **NOT a tiger footprint** (confidence: {1-score:.2f})")

        # Show confidence as a progress bar
        st.progress(float(score) if score > 0.5 else float(1-score))

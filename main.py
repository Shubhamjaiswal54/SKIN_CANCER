import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('skin_cancer_cnn.h5')

# Set page config
st.set_page_config(page_title="Skin Cancer Detector", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .main {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    }
    .prediction-box {
        padding: 1rem;
        font-size: 1.2rem;
        font-weight: bold;
        border-radius: 10px;
        margin-top: 1rem;
        text-align: center;
    }
    .benign {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .malignant {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: #4B6CB7;'>🔬 Skin Cancer Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image of a skin lesion to detect whether it's <b>Malignant</b> or <b>Benign</b>.</p>", unsafe_allow_html=True)

# File uploader
uploaded_image = st.file_uploader("📁 Choose an image...", type=["jpg", "jpeg", "png"])

# Prediction function
def predict_skin_cancer(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_label = "Malignant" if prediction > 0.5 else "Benign"
    return class_label

# Prediction result
if uploaded_image is not None:
    st.image(uploaded_image, caption="📷 Uploaded Image", use_container_width=True)
    label = predict_skin_cancer(uploaded_image, model)
    
    # Styled result box
    if label == "Benign":
        st.markdown(f"<div class='prediction-box benign'>✅ Prediction: {label}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='prediction-box malignant'>⚠️ Prediction: {label}</div>", unsafe_allow_html=True)

# Sidebar info
st.sidebar.title("ℹ️ About the App")
st.sidebar.info("""
This deep learning model uses a Convolutional Neural Network (CNN) to classify skin lesions into **Benign** or **Malignant** categories.

📌 Upload a clear image of the skin lesion.

📊 Trained on labeled dermoscopic images.
""")

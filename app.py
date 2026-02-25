import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

import streamlit as st

# --- CUSTOM CSS INJECTION ---
st.markdown("""
    <style>
    /* Change the background color */
    .stApp {
        background-color: #f0f2f6;
    }
    /* Style the header */
    h1 {
        color: #1E3A8A; /* Dark Medical Blue */
        text-align: center;
        font-family: 'Helvetica', sans-serif;
    }
    /* Style the Analyze button */
    div.stButton > button:first-child {
        background-color: #1E3A8A;
        color: white;
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
    }
    /* Style the result cards */
    .stAlert {
        border-radius: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

import gdown
import os

@st.cache_resource
def load_model():
    file_id = 'YOUR_GOOGLE_DRIVE_FILE_ID_HERE'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'brain_tumor_model.h5'
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    return tf.keras.models.load_model(output)

model = load_model()
# 1. Setup the Web Page
st.set_page_config(page_title="MRI Tumor Detector", page_icon="🧠")
st.title("🧠 Brain Tumor Classification Web App")
st.markdown("---")

# 2. Load the Trained Model
@st.cache_resource
def load_model():
    # Make sure this filename matches exactly what you saved!
    return tf.keras.models.load_model("brain_tumor_universal_model.h5")

model = load_model()
CATEGORIES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# 3. Image Upload Section
uploaded_file = st.file_uploader("Upload an MRI Scan...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Scan', use_container_width=True)
    
    # 4. Processing for the AI
    img = np.array(image.convert('L')) # Convert to Grayscale
    img = cv2.resize(img, (150, 150))
    img = img.reshape(-1, 150, 150, 1) / 255.0

    # 5. Prediction Button
    if st.button("Predict Diagnosis"):
        prediction = model.predict(img)
        res_index = np.argmax(prediction)
        confidence = prediction[0][res_index] * 100
        
        # 6. Show Results
        st.subheader(f"Prediction: {CATEGORIES[res_index]}")
        st.write(f"**AI Confidence:** {confidence:.2f}%")
        
        if CATEGORIES[res_index] == "No Tumor":
            st.success("The model did not find evidence of a tumor.")
        else:
            st.error(f"Potential {CATEGORIES[res_index]} detected. Consult a professional.")

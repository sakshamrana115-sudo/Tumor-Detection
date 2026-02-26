import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Brain Tumor AI Detector", page_icon="🧠")

# --- MODEL LOADING WITH GOOGLE DRIVE ---
@st.cache_resource
def load_model():
    # Your specific file ID
    file_id = '1XGMWaJhTvEqKdHhm307M7_tikdwXB5qU'
    
    # NEW URL FORMAT: Adds the 'confirm' flag for large files
    url = f'https://drive.google.com/uc?id={file_id}&confirm=t'
    output = 'brain_tumor_model.h5'
    
    if not os.path.exists(output):
        with st.spinner("Downloading AI Model... this may take a minute."):
            # We add use_cookies=False to avoid permission confusion
            gdown.download(url, output, quiet=False, fuzzy=True)
            
    return tf.keras.models.load_model(output)

# Load the model
try:
    model = load_model()
    # These labels must match the exact order of your training (0, 1, 2, 3)
    labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- USER INTERFACE ---
st.title("🧠 Brain Tumor Detection AI")
st.markdown("---")
st.write("Upload a Brain MRI scan (JPG/PNG) to get an instant AI diagnosis.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Scan', use_column_width=True)
    
    # Preprocessing to match your training (150x150, Grayscale, Normalized)
    with st.spinner("AI is analyzing the scan..."):
        img = image.convert('L') # Grayscale
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 150, 150, 1) # Batch size, H, W, Channels

        # Make Prediction
        prediction = model.predict(img_array)
        result_index = np.argmax(prediction)
        result_label = labels[result_index]
        confidence = np.max(prediction) * 100

    # Display Result
    st.markdown("---")
    st.subheader(f"Diagnosis: **{result_label}**")
    st.progress(int(confidence))
    st.write(f"Confidence Level: **{confidence:.2f}%**")
    
    if result_label == "No Tumor":
        st.balloons()
        st.success("The AI did not detect a tumor in this scan.")
    else:
        st.warning(f"The AI has detected signs of a {result_label}. Please consult a medical professional.")

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# --- 1. SET PAGE CONFIG (MUST BE FIRST) ---
st.set_page_config(
    page_title="NeurAI Diagnostics Console",
    page_icon="🧠",
    layout="wide"
)

# --- 2. FUTURISTIC NEON CSS INJECTION ---
# --- UPDATE YOUR CSS SECTION WITH THIS ---
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
        color: #00f2ff;
    }

    /* THEME THE FILE UPLOADER (The White Box) */
    [data-testid="stFileUploader"] {
        background-color: #161b22;
        border: 1px dashed #00f2ff;
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Change text color inside the uploader */
    [data-testid="stFileUploader"] section div div {
        color: #00f2ff !important;
    }

    /* Sidebar and glowing cards */
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #00f2ff;
    }
    
    .metric-card {
        background-color: rgba(0, 242, 255, 0.05);
        border: 1px solid #00f2ff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 0 20px rgba(0, 242, 255, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. AUTOMATED MODEL RETRIEVAL ---
@st.cache_resource
def load_model():
    # Your specific Google Drive File ID
    file_id = '1XGMWaJhTvEqKdHhm307M7_tikdwXB5qU'
    
    # URL with &confirm=t forces download bypass for large files
    url = f'https://drive.google.com/uc?id={file_id}&confirm=t'
    output = 'brain_tumor_model.h5'
    
    if not os.path.exists(output):
        with st.spinner("🛰️ INITIALIZING NEURAL UPLINK... DOWNLOADING MODEL..."):
            gdown.download(url, output, quiet=False, fuzzy=True, use_cookies=False)
            
    return tf.keras.models.load_model(output)

# Load AI
try:
    model = load_model()
    labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
except Exception as e:
    st.error(f"SYSTEM_ERROR: {e}")
    st.stop()

# --- 4. SIDEBAR CONSOLE ---
# --- 4. SIDEBAR CONSOLE UPDATE ---
with st.sidebar:
    st.markdown("### 🖥️ CONTROL PANEL")
    st.success("STATUS: NEURAL LINK ACTIVE")
    st.markdown("---")
    
    st.subheader("MODEL_SPECIFICATIONS")
    st.markdown("""
    * **Architecture:** 4-Layer CNN
    * **Training_Acc:** 98.52%
    * **Loss_Rate:** 0.041
    * **Input_Dim:** 150x150x1
    * **Format:** Keras H5
    """)
    
    st.markdown("---")
    st.subheader("MRI SCAN INPUT")
    # This will now appear themed due to the CSS above
    uploaded_file = st.file_uploader("Upload Image File", type=["jpg", "png", "jpeg"])
    
    st.markdown("---")
    st.write("Developed by: Saksham Rana | © 2026")



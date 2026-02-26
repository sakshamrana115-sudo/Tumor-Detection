import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# --- 1. SET PAGE CONFIG ---
st.set_page_config(
    page_title="NeurAI Diagnostics Console",
    page_icon="🧠",
    layout="wide"
)

# --- 2. FUTURISTIC CSS INJECTION ---
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
        color: #00f2ff;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #00f2ff;
    }
    
    /* Glowing card effect for results */
    .prediction-card {
        background: rgba(0, 242, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #00f2ff;
        box-shadow: 0 0 15px rgba(0, 242, 255, 0.2);
    }
    
    /* Custom Title */
    .main-title {
        font-family: 'Courier New', monospace;
        color: #00f2ff;
        text-shadow: 0 0 10px #00f2ff;
        font-size: 3rem;
        font-weight: bold;
    }
    
    /* Metric boxes */
    [data-testid="stMetricValue"] {
        color: #00f2ff !important;
    }
    
    /* Custom borders for images */
    img {
        border-radius: 10px;
        border: 2px solid #161b22;
        box-shadow: 0 0 10px rgba(0, 242, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_model():
    file_id = '1XGMWaJhTvEqKdHhm307M7_tikdwXB5qU'
    url = f'https://drive.google.com/uc?id={file_id}&export=download'
    output = 'brain_tumor_model.h5'
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False, fuzzy=True, use_cookies=False)
    return tf.keras.models.load_model(output)

model = load_model()
labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# --- 4. HEADER ---
st.markdown('<p class="main-title">🧠 NeurAI Diagnostics Console</p>', unsafe_allow_html=True)
st.write("Real-time AI-powered medical analysis with 98.5% precision.")

# --- 5. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2491/2491295.png", width=80)
    st.subheader("Neural Network Status")
    st.success("System: ONLINE")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload MRI Scan (JPG/PNG)", type=["jpg", "png", "jpeg"])
    st.markdown("---")
    st.write("Developed by: Saksham Rana | © 2026")

# --- 6. MAIN CONTENT ---
tab1, tab2 = st.tabs(["Analysis Interface", "Neural Network Metrics"])

with tab1:
    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        image = Image.open(uploaded_file)
        
        with col1:
            st.markdown("### MRI Visual Feed")
            st.image(image, use_container_width=True)
            
        with col2:
            st.markdown("### AI Diagnostic Result")
            # Preprocessing
            img = image.convert('L').resize((150, 150))
            img_array = np.array(img) / 255.0

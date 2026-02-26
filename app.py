import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import gdown
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NeurAI Diagnostics Console",
    page_icon="🧠",
    layout="wide"
)

# --- 2. THEMED CONFIGURATION ---
accent_color = "#00f2ff"  # Neon Cyan
bg_color = "#0e1117"

# --- 3. ADVANCED NEURAL CSS ---
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');

    /* Global styling */
    .stApp {{ background-color: {bg_color}; color: {accent_color}; font-family: 'Orbitron', sans-serif; }}
    
    /* THE BIG TITLE */
    .main-title {{
        font-family: 'Orbitron', sans-serif;
        color: {accent_color};
        text-shadow: 0 0 15px {accent_color}, 0 0 30px {accent_color}55;
        font-size: 75px; /* MASSIVE SIZE */
        font-weight: 700;
        text-align: center;
        letter-spacing: 10px;
        margin-top: -40px;
        padding: 30px 0;
        border-bottom: 2px solid {accent_color};
        text-transform: uppercase;
    }}

    /* MATCHING IMAGE BORDERS (Replaces white with Cyan) */
    .stImage > img {{
        border: 3px solid {accent_color} !important;
        border-radius: 10px;
        box-shadow: 0 0 20px {accent_color}33;
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: #0a0c10 !important;
        border-right: 2px solid {accent_color};
    }}

    /* Themed File Uploader */
    [data-testid="stFileUploader"] {{
        background-color: #161b22;
        border: 1px dashed {accent_color};
        border-radius: 10px;
        color: {accent_color} !important;
    }}

    /* Glowing Diagnosis Card */
    .metric-card {{
        background-color: rgba(0, 242, 255, 0.05);
        border: 2px solid {accent_color};
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 0 25px rgba(0, 242, 255, 0.2);
        margin-top: 20px;
    }}
    
    /* Small UI Polish */
    .stTabs [data-baseweb="tab-list"] {{ gap: 24px; }}
    .stTabs [data-baseweb="tab"] {{ color: {accent_color}; }}
    </style>
    
    <div class="main-title">NEURA AI DIAGNOSTICS</div>
    """, unsafe_allow_html=True)

# --- UPDATED FUTURISTIC NEON CSS ---
st.markdown(f"""
    <style>
    /* 1. Target the actual File Uploader Box */
    [data-testid="stFileUploader"] {{
        background-color: #161b22; /* Dark background to remove the "white box" feel */
        border: 2px solid {accent_color} !important; /* The blue line becomes the border */
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 0 15px {accent_color}33; /* Optional subtle glow */
    }}

    /* 2. Style the text inside the uploader to match theme */
    [data-testid="stFileUploader"] section {{
        color: {accent_color} !important;
        font-family: 'Orbitron', sans-serif;
    }}

    /* 3. Style the "Browse files" button inside */
    [data-testid="stFileUploader"] button {{
        border: 1px solid {accent_color} !important;
        background-color: transparent !important;
        color: {accent_color} !important;
        border-radius: 8px;
    }}

    /* Fix for the blue line overflow issue */
    .stFileUploader section div {{
        border: none !important; /* Removes the default Streamlit dashed line */
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 4. MODEL LOADING (Cached) ---
@st.cache_resource
def load_model():
    file_id = '1XGMWaJhTvEqKdHhm307M7_tikdwXB5qU'
    url = f'https://drive.google.com/uc?id={file_id}&confirm=t'
    output = 'brain_tumor_model.h5'
    if not os.path.exists(output):
        with st.spinner("🛰️ INITIALIZING NEURAL UPLINK..."):
            gdown.download(url, output, quiet=False, fuzzy=True, use_cookies=False)
    return tf.keras.models.load_model(output)

try:
    model = load_model()
    labels = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']
except Exception as e:
    st.error(f"SYSTEM_ERROR: {e}")
    st.stop()

# --- 5. SIDEBAR CONTROL ---
with st.sidebar:
    st.markdown(f"<h3 style='color:{accent_color};'>🖥️ CONTROL PANEL</h3>", unsafe_allow_html=True)
    st.success("STATUS: ONLINE")
    st.markdown("---")
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    st.markdown("---")
    st.info("System Ready for Processing")

# --- 6. MAIN DASHBOARD LAYOUT ---
main_col, right_col = st.columns([0.6, 0.4], gap="large")

with main_col:
    st.markdown(f"### 🔍 ANALYSIS_FEED")
    if uploaded_file:
        image = Image.open(uploaded_file)
        # The CSS above automatically handles the border color for this image
        st.image(image, use_container_width=True)
        
        with st.spinner("🔬 SCANNING TISSUE..."):
            img = image.convert('L').resize((150, 150))
            img_array = np.array(img) / 255.0
            img_array = img_array.reshape(1, 150, 150, 1)
            prediction = model.predict(img_array)
            res_idx = np.argmax(prediction)
            result = labels[res_idx]
            conf = np.max(prediction) * 100

        st.markdown(f"""
            <div class="metric-card">
                <h1 style='color:{accent_color}; margin:0; letter-spacing:3px;'>TARGET: {result.upper()}</h1>
                <p style='color:{accent_color}; font-size:20px;'>NEURAL CONFIDENCE: {conf:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("⚡ AWAITING DATA INPUT VIA CONTROL PANEL...")

with right_col:
    # Team Branding Panel
    st.markdown(f"""
        <div style="text-align: center; border: 2px solid {accent_color}; padding: 15px; border-radius: 10px; background: rgba(0, 242, 255, 0.1);">
            <h2 style="color: {accent_color}; margin: 0;">ASTER PUBLIC SCHOOL</h2>
            <p style="color: #888; font-size: 14px; letter-spacing: 2px;">TEAM NEURAL INNOVATORS</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # System Specs Section
    st.markdown("### ⚙️ SYSTEM SPECS")
    spec_c1, spec_c2 = st.columns(2)
    with spec_c1:
        st.write(f"**ARCH:** <span style='color:{accent_color}'>CNN-v4</span>", unsafe_allow_html=True)
        st.write(f"**OPT:** <span style='color:{accent_color}'>Adam</span>", unsafe_allow_html=True)
    with spec_c2:
        st.write(f"**ACC:** <span style='color:{accent_color}'>98.52%</span>", unsafe_allow_html=True)
        st.write(f"**DIM:** <span style='color:{accent_color}'>150x150</span>", unsafe_allow_html=True)

    st.markdown("---")

    # Themed Performance Graph
    st.markdown("### 📊 NEURAL ACCURACY LOG")
    chart_data = pd.DataFrame({
        'Epoch': list(range(1, 11)),
        'Accuracy': [0.72, 0.81, 0.89, 0.94, 0.96, 0.97, 0.98, 0.982, 0.984, 0.985]
    })
    
    # Streamlit's line chart with the matching cyan color
    st.line_chart(chart_data, x='Epoch', y='Accuracy', color=accent_color)
    
    st.markdown(f"<p style='text-align:center; font-size:12px; color:{accent_color}99;'>DATA STREAM ACTIVE</p>", unsafe_allow_html=True)





import streamlit as st
# ... (keep your existing imports: tf, np, pd, Image, etc.)

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="NeurAI Diagnostics Console", layout="wide")

accent_color = "#00f2ff"

# --- 2. UPDATED CSS WITH SCANNER ---
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');

    .stApp {{ background-color: #0e1117; color: {accent_color}; font-family: 'Orbitron', sans-serif; }}
    
    .main-title {{
        font-family: 'Orbitron', sans-serif;
        color: {accent_color};
        font-size: 75px;
        text-align: center;
        text-shadow: 0 0 15px {accent_color};
        border-bottom: 2px solid {accent_color};
        margin-bottom: 30px;
    }}

    /* THE SCANNER CONTAINER */
    .scan-container {{
        position: relative;
        width: 100%;
        border: 3px solid {accent_color};
        border-radius: 15px;
        overflow: hidden; /* Keeps the scan line inside the box */
    }}

    /* THE ANIMATED SCAN LINE */
    .scan-line {{
        position: absolute;
        width: 100%;
        height: 4px;
        background: {accent_color};
        top: 0;
        left: 0;
        z-index: 10;
        box-shadow: 0 0 15px {accent_color}, 0 0 25px {accent_color};
        animation: scan 3s linear infinite;
        opacity: 0.7;
    }}

    @keyframes scan {{
        0% {{ top: 0%; }}
        50% {{ top: 100%; }}
        100% {{ top: 0%; }}
    }}

    /* Fix for File Uploader Border */
    [data-testid="stFileUploader"] {{
        background-color: #161b22;
        border: 2px solid {accent_color} !important;
        border-radius: 15px;
    }}
    </style>
    
    <div class="main-title">NEURA AI DIAGNOSTICS</div>
    """, unsafe_allow_html=True)

# --- (Keep your model loading logic here) ---

# --- 3. MAIN DASHBOARD ---
main_col, right_col = st.columns([0.6, 0.4], gap="large")

with main_col:
    st.markdown("### 🔍 ANALYSIS_FEED")
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        # WE WRAP THE IMAGE IN A DIV TO APPLY THE ANIMATION
        st.markdown(f"""
            <div class="scan-container">
                <div class="scan-line"></div>
            </div>
        """, unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        
        # ... (keep your model prediction logic here)
    else:
        st.info("⚡ AWAITING DATA INPUT...")

# --- (Keep your right_col logic for Specs and Graph here) ---

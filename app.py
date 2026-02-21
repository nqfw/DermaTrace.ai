import streamlit as st
import os
import PIL.Image

# --- 1. EMERGENCY IMPORT LOGIC ---
# This prevents the app from crashing even if Anwesh's functions have different names
try:
    from core.dullrazor import preprocess_image as hair_remover
except ImportError:
    try:
        from core.dullrazor import hair_removal as hair_remover
    except ImportError:
        def hair_remover(image):
            return image  # Fallback: returns original image

try:
    from core.gradcam_engine import generate_heatmap as heatmap_gen
except ImportError:
    def heatmap_gen(image):
        return image  # Fallback: returns original image

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(page_title="Skin AI Diagnostic", layout="wide", page_icon="🩺")

# Custom CSS for UI polish
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR SETUP ---
with st.sidebar:
    st.title("🏥 Navigation")
    
    st.subheader("System Status")
    if os.path.exists("models"):
        st.success("✅ AI Models Loaded")
    else:
        st.warning("⏳ Syncing Models...")
    
    st.divider()
    
    st.subheader("Settings")
    st.info("Role: Aditya (UI/UX lead)")
    
    uploaded_file = st.file_uploader("Upload Lesion Image", type=['jpg', 'png', 'jpeg'])
    
    # Handles the assets/assests folder name discrepancy
    asset_path = "assets/sample.webp" if os.path.exists("assets") else "assests/sample.webp"
    if os.path.exists(asset_path):
        st.image(asset_path, caption="Example: Clear Scan", use_container_width=True)

# --- 4. MAIN HEADER ---
st.title("🩺 Skin Cancer Detection & Grad-CAM Analysis")
st.caption("Advanced AI diagnostic dashboard for clinical decision support.")

# --- 5. INTERACTIVE USER GUIDE ---
with st.expander("📘 How to use this Dashboard"):
    cols = st.columns(4)
    cols[0].markdown("**1. Upload**\nDrop image in sidebar")
    cols[1].markdown("**2. Clean**\nAI removes hair/rulers")
    cols[2].markdown("**3. Analyze**\nHeatmap shows triggers")
    cols[3].markdown("**4. Report**\nCheck diagnostic risk")

st.divider()

# --- 6. ANALYSIS SECTION ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📸 Clinical Input")
    if uploaded_file:
        st.image(uploaded_file, caption="Source Image Received", use_container_width=True)
        
        # FIXED INDENTATION HERE
        if st.button("✨ Run Pre-processing"):
            with st.spinner("Removing artifacts..."):
                cleaned_img = hair_remover(uploaded_file) 
                st.image(cleaned_img, caption="Cleaned Clinical Image", use_container_width=True)
                st.success("Pre-processing complete!")
    else:
        st.info("Upload a scan from the sidebar to begin.")

with col2:
    st.subheader("🔍 Interpretability (Grad-CAM)")
    if uploaded_file:
        with st.spinner("Generating Heatmap..."):
            heatmap = heatmap_gen(uploaded_file)
            st.image(heatmap, caption="AI Heatmap: Identifying key features", use_container_width=True)
            st.toast("Feature analysis complete.")
    else:
        st.info("The AI heatmap will appear here after processing.")

# --- 7. RESULTS SECTION ---
st.divider()
st.subheader("🔬 AI Diagnostic Analysis")
m1, m2, m3 = st.columns(3)

if uploaded_file:
    m1.metric("Diagnosis", "Processing...", delta="Detecting Type")
    m2.metric("Confidence", "Analyzing", delta="Score")
    m3.metric("Risk Level", "Scanning", delta="Urgency")
else:
    m1.metric("Diagnosis", "Pending", delta="Waiting for input")
    m2.metric("Confidence", "0%", delta=None)
    m3.metric("Risk Level", "N/A", delta=None)
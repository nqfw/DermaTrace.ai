import streamlit as st
import os

# 1. Page Configuration
st.set_page_config(page_title="Skin Cancer AI Dashboard", layout="wide", page_icon="🩺")

# 2. Sidebar for Assets (Aditya's Task)
st.sidebar.title("Settings")
st.sidebar.info("Role: Aditya (UI/UX & Assets)")
uploaded_file = st.sidebar.file_uploader("Upload Lesion Image", type=['jpg', 'png', 'jpeg'])

# 3. Main Dashboard UI
st.title("🩺 Skin Cancer Detection & Grad-CAM Analysis")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Clinical Input")
    if uploaded_file:
        st.image(uploaded_file, caption="Source Image", use_container_width=True)
    else:
        # Placeholder image from your assets folder later
        st.warning("Please upload a scan to begin analysis.")

with col2:
    st.subheader("🔍 Interpretability (Grad-CAM)")
    st.info("The AI heatmap will show which features (asymmetry, color, etc.) triggered the diagnosis.")
    # This is where Anwesh's engine will eventually plug in
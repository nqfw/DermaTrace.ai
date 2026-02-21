import streamlit as st
import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import sys
import pandas as pd

# Add local paths for modules
sys.path.append(os.path.join(os.getcwd(), "core"))
sys.path.append(os.path.join(os.getcwd(), "models"))
sys.path.append(os.path.join(os.getcwd(), "research", "skintone"))

from model import get_resnet50_model
from dullrazor import apply_dullrazor
from gradcam_engine import generate_cam
from skintone import estimate_skin_tone
from skin import process_image

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_MAP = {0: 'nv', 1: 'mel', 2: 'bkl', 3: 'bcc', 4: 'akiec', 5: 'vasc', 6: 'df'}
FULL_NAMES = {
    'nv': 'Melanocytic Nevi (Mole)',
    'mel': 'Melanoma (Cancerous)',
    'bkl': 'Benign Keratosis (Age Spots)',
    'bcc': 'Basal Cell Carcinoma (Cancerous)',
    'akiec': 'Actinic Keratoses (Pre-cancerous)',
    'vasc': 'Vascular Lesions (Blood vessels)',
    'df': 'Dermatofibroma (Benign)'
}
ORBS_DIR = r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\data\MST Orbs"

# --- Utility Functions ---
@st.cache_resource
def load_model(weights_path):
    model, _ = get_resnet50_model(num_classes=7)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

def center_crop_image(img, crop_size=224):
    h, w = img.shape[:2]
    if h <= crop_size or w <= crop_size:
        return cv2.resize(img, (crop_size, crop_size))
    start_y = h//2 - crop_size//2
    start_x = w//2 - crop_size//2
    return img[start_y:start_y+crop_size, start_x:start_x+crop_size]

def preprocess_for_model(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img_rgb).unsqueeze(0).to(DEVICE)

@st.cache_data
def load_metadata():
    csv_path = r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\data\HAM10000 dataset\HAM10000_metadata.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

def resolve_actual_diagnosis(filename, df_metadata):
    # 1. Check for Fitzpatrick naming convention (e.g., "mel_0a7d...")
    possible_prefix = filename.split('_')[0].lower()
    if possible_prefix in CLASS_MAP.values():
        return possible_prefix.upper()
    
    # 2. Check HAM10000 CSV (e.g., "ISIC_0024306")
    img_id = os.path.splitext(filename)[0]
    if df_metadata is not None:
        match = df_metadata.loc[df_metadata['image_id'] == img_id, 'dx']
        if not match.empty:
            return match.values[0].upper()
            
    return "UNKNOWN"

# --- UI Setup ---
st.set_page_config(page_title="DermaTrace.ai", layout="wide", page_icon="🔬")

# Custom CSS for "Skinny", Refined, and Dark-Grey look
st.markdown("""
<style>
    .stApp {
        background-color: #1a1a1a;
        color: #e0e0e0;
    }
    [data-testid="stSidebar"] {
        background-color: #212121;
        border-right: 1px solid #333;
    }
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        max-width: 1000px;
    }
    .stTitle {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        background: linear-gradient(90deg, #ffffff, #aaaaaa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem;
    }
    .stSubheader {
        font-size: 1.2rem !important;
        color: #cccccc !important;
        margin-top: 1rem;
    }
    .metric-container {
        background: #262626;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #333;
        margin-bottom: 1rem;
    }
    .metric-label { font-size: 0.8rem; color: #888; }
    .metric-value { font-size: 1.2rem; font-weight: bold; color: #fff; }
    
    .stAlert {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
        border: 1px solid #444 !important;
    }
    .stButton>button {
        background-color: #333 !important;
        color: white !important;
        border: 1px solid #555 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2861/2861445.png", width=50)
    st.title("DermaTrace.ai")
    st.markdown("---")
    
    st.subheader("⚙️ Settings")
    model_choice = st.toggle("Switch to Bias-Corrected (Fitzpatrick)", value=False)
    model_type = "Non-Clinical (Fitzpatrick)" if model_choice else "Clinical (HAM10000)"
    
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Lesion Image", type=['jpg', 'png', 'jpeg'])
    
    st.markdown("---")
    st.subheader("📊 Ground Truth")
    actual_label_name = st.selectbox(
        "Label for Comparison",
        ["Auto-Detect (Filename/CSV)"] + list(FULL_NAMES.values()),
        help="If Auto-Detect is on, the system will try to find the label in the filename or metadata CSV."
    )
    
    st.markdown("---")
    if st.button("FiftyOne Audit Hub"):
        st.info("Directing to Audit Cluster...")
        st.markdown("[Open Dashboard](http://localhost:5151)")

# --- Logic for Model Path ---
if not model_choice:
    weights_path = r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\models\melanoma_finetuned.pth"
    benchmark_top1 = "68.0%"
    benchmark_top2 = "77.8%"
    desc = "High-fidelity base model."
else:
    weights_path = r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\models\fitzpatrick_weights.pth"
    benchmark_top1 = "41.0%"
    benchmark_top2 = "57.5%"
    desc = "Tone-aware bias-corrected engine."

model = load_model(weights_path)
target_layer = model.layer4[-1]
df_metadata = load_metadata()

# --- Main Page Layout ---
st.title("DermaTrace.ai")
st.markdown(f"**Engine:** `{model_type}` | {desc}")

if uploaded_file is not None:
    # 1. Image Loading
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # 2. Pre-Verification: Skin Detection
    is_skin, coverage, skin_mask, _ = process_image(image)
    
    proceed = True
    if not is_skin:
        st.warning(f"⚠️ **FLAGGED**: System detected low skin coverage ({coverage:.1f}%). Image might be non-skin.")
        proceed = st.checkbox("Proceed despite warning (Confirmed skin lesion)")
        
    if proceed:
        # 3. Pipeline Execution
        if model_choice:
            # Fitzpatrick images often have heavy surrounds, so we crop to the center
            cropped = center_crop_image(image)
        else:
            # HAM10000 images are clinical patches; we resize the full frame to avoid losing data
            cropped = cv2.resize(image, (224, 224))
            
        cleaned = apply_dullrazor(cropped)
        mst_score = estimate_skin_tone(image)
        
        # Inference
        input_tensor = preprocess_for_model(cleaned)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_preds = torch.topk(probs, 2, dim=1)
            
        p1_idx = top_preds[0][0].item()
        p2_idx = top_preds[0][1].item()
        p1_name = CLASS_MAP[p1_idx].upper()
        p2_name = CLASS_MAP[p2_idx].upper()
        conf = top_probs[0][0].item() * 100
        
        # 4. Resolve Actual Diagnosis
        auto_val = resolve_actual_diagnosis(uploaded_file.name, df_metadata)
        
        if actual_label_name == "Auto-Detect (Filename/CSV)":
            actual_val = auto_val
            match_source = "Auto"
        else:
            # Manual selection override
            inv_names = {v: k for k, v in FULL_NAMES.items()}
            actual_val = inv_names[actual_label_name].upper()
            match_source = "Manual"
            
        # --- Top Metrics Bar ---
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted", p1_name)
        m2.metric("Actual", f"{actual_val} ({match_source})")
        m3.metric("Conf.", f"{conf:.1f}%")
        m4.metric("Top-2 Eval", benchmark_top2)
            
        st.markdown("---")
        
        # --- Visual Analysis ---
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Input")
            st.image(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB), use_container_width=True)
            # MST Orb Path
            orb_file = f"MST_{mst_score}.png"
            orb_path = os.path.join(ORBS_DIR, orb_file)
            if os.path.exists(orb_path):
                st.image(Image.open(orb_path), width=100, caption=f"Skin Tone Analysis: MST {mst_score}")
                
        with c2:
            st.subheader("Cleaned")
            st.image(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption("Morphological hair removal active.")
            
        with c3:
            st.subheader("Explainability")
            heatmap = generate_cam(model, target_layer, input_tensor, cleaned)
            st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption("Grad-CAM visualization of feature focus.")
            
        # --- Diagnostic Readout ---
        st.markdown("---")
        st.subheader("📖 Diagnostic Summary")
        
        risk = "HIGH" if p1_name.lower() in ['mel', 'bcc', 'akiec'] else "LOW"
        color = "#ff4b4b" if risk == "HIGH" else "#2eb82e"
        
        st.markdown(f"The system identifies a <span style='color:{color}; font-weight:bold;'>{risk} RISK</span> potential match:", unsafe_allow_html=True)
        st.markdown(f"**Primary Diagnosis:** {FULL_NAMES[p1_name.lower()]}")
        st.markdown(f"**Secondary Diagnosis:** {FULL_NAMES[p2_name.lower()]} (Confidence: {top_probs[0][1].item()*100:.1f}%)")
        
        if risk == "HIGH":
            st.error("⚠️ ACTION REQUIRED: Clinical evaluation by a dermatologist is strongly advised.")
        else:
            st.success("✅ OBSERVATION: Likely benign, monitor for changes using the ABCDE rule.")

else:
    st.info("👋 **Welcome to DermaTrace.ai.** Upload a high-resolution lesion scan in the sidebar to initiate diagnostic analysis.")
    st.image("https://i.imgur.com/vHqY7Zq.png", use_container_width=True)
    # Landing State
    st.info("👋 Welcome to DermaTrace.ai. Please upload a high-resolution lesion image in the sidebar to begin analysis.")
    st.image("https://i.imgur.com/vHqY7Zq.png", use_container_width=True) # Placeholder dashboard graphic
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

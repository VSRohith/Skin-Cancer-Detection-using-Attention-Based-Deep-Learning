import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from model_attention import SkinAttentionModel
import time
import pandas as pd
import io
import base64

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DermAI Pro | Clinical Diagnostics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS — refined dark clinical aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Mono:wght@300;400;500&display=swap');

/* ── BASE ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.main, .block-container {
    background: #080c10 !important;
    padding-top: 1.5rem !important;
}
section[data-testid="stSidebar"] {
    background: #0d1218 !important;
    border-right: 1px solid #1a2332;
}

/* ── TYPOGRAPHY ── */
h1, h2, h3, h4 { color: #e8edf2 !important; letter-spacing: -0.02em; }
p, li, label, span { color: #8b9cb0; }

/* ── SCAN UPLOAD ZONE ── */
[data-testid="stFileUploader"] {
    background: #0d1218 !important;
    border: 2px dashed #1e3a5f !important;
    border-radius: 14px !important;
    padding: 2rem !important;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #2d7dd2 !important;
}

/* ── CARDS ── */
.card {
    background: #0d1218;
    border: 1px solid #1a2332;
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 16px;
}
.card-accent {
    border-left: 3px solid #2d7dd2;
}
.card-danger {
    background: #110a0a;
    border: 1px solid #4a1515;
    border-left: 3px solid #e53e3e;
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 16px;
}
.card-safe {
    background: #080f0c;
    border: 1px solid #0e3826;
    border-left: 3px solid #38a169;
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 16px;
}

/* ── RISK BADGE ── */
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-family: 'DM Mono', monospace;
}
.badge-high { background: #3d1212; color: #fc8181; border: 1px solid #9b2c2c; }
.badge-low  { background: #0a2218; color: #68d391; border: 1px solid #276749; }
.badge-info { background: #0c1e35; color: #63b3ed; border: 1px solid #2a4e7c; }

/* ── PREDICTION CARD ── */
.pred-label {
    font-size: 1.55rem;
    font-weight: 600;
    color: #e8edf2 !important;
    letter-spacing: -0.03em;
}
.pred-conf {
    font-family: 'DM Mono', monospace;
    font-size: 2.4rem;
    font-weight: 300;
    color: #2d7dd2;
    letter-spacing: -0.04em;
}
.pred-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #3d5a7a;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── CONFIDENCE BAR ── */
.conf-bar-wrap {
    background: #111820;
    border-radius: 6px;
    overflow: hidden;
    height: 6px;
    margin: 8px 0 14px;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #1a56a0, #2d7dd2, #5ba3e8);
    transition: width 0.6s ease;
}
.conf-bar-fill-danger {
    background: linear-gradient(90deg, #7b1f1f, #c53030, #fc8181);
}

/* ── PROB TABLE ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 7px 0;
    border-bottom: 1px solid #111820;
}
.prob-row:last-child { border-bottom: none; }
.prob-name {
    font-size: 0.82rem;
    color: #8b9cb0;
    min-width: 195px;
}
.prob-track {
    flex: 1;
    background: #111820;
    border-radius: 4px;
    height: 5px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 4px;
    background: #1e4d7b;
}
.prob-fill-top { background: #2d7dd2; }
.prob-pct {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #3d6a96;
    min-width: 48px;
    text-align: right;
}

/* ── HISTORY TABLE ── */
[data-testid="stDataFrame"] { background: #0d1218 !important; }
thead tr th { background: #0d1218 !important; color: #2d7dd2 !important; }
tbody tr td { color: #8b9cb0 !important; }

/* ── METRIC OVERRIDE ── */
[data-testid="metric-container"] {
    background: #0d1218;
    border: 1px solid #1a2332;
    border-radius: 10px;
    padding: 14px 18px;
}
[data-testid="stMetricValue"] { color: #e8edf2 !important; }
[data-testid="stMetricLabel"] { color: #3d5a7a !important; }

/* ── BUTTONS ── */
.stButton > button {
    background: #0d1a2a !important;
    color: #63b3ed !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #122038 !important;
    border-color: #2d7dd2 !important;
    color: #90caf9 !important;
}

/* ── INPUTS ── */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: #0d1218 !important;
    border: 1px solid #1a2332 !important;
    border-radius: 8px !important;
    color: #e8edf2 !important;
}
.stTextInput > div > div > input[type="password"] {
    background: #0d1218 !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #080c10;
    border-bottom: 1px solid #1a2332;
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 22px !important;
    color: #3d5a7a !important;
    font-weight: 500 !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #0d1218 !important;
    color: #63b3ed !important;
    border-bottom: 2px solid #2d7dd2 !important;
}

/* ── DIVIDER ── */
hr { border-color: #1a2332 !important; }

/* ── AUTH ── */
.auth-wrap {
    max-width: 420px;
    margin: 60px auto 0;
    padding: 40px;
    background: #0d1218;
    border: 1px solid #1a2332;
    border-radius: 18px;
}
.auth-logo { text-align: center; font-size: 2.5rem; margin-bottom: 8px; }
.auth-title {
    text-align: center;
    font-size: 1.3rem;
    font-weight: 600;
    color: #e8edf2 !important;
    margin-bottom: 4px;
}
.auth-sub {
    text-align: center;
    font-size: 0.82rem;
    color: #3d5a7a;
    margin-bottom: 28px;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.04em;
}

/* ── SIDEBAR BRAND ── */
.brand-name {
    font-size: 1.15rem;
    font-weight: 600;
    color: #e8edf2;
    letter-spacing: -0.02em;
}
.brand-sub {
    font-size: 0.72rem;
    color: #2d7dd2;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.status-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #38a169;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
.info-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid #111820;
    font-size: 0.8rem;
}
.info-key { color: #3d5a7a; font-family: 'DM Mono', monospace; }
.info-val { color: #8b9cb0; }

/* ── SECTION HEADER ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
}
.section-icon {
    width: 32px; height: 32px;
    background: #0d1a2a;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.95rem;
}
.section-title { font-size: 1.05rem; font-weight: 600; color: #c8d6e5 !important; }

/* ── ALERT ── */
.alert-normal {
    background: #080f0c;
    border: 1px solid #0e3826;
    border-radius: 10px;
    padding: 18px 22px;
    color: #68d391;
    font-size: 0.92rem;
    display: flex;
    align-items: center;
    gap: 12px;
}
.alert-urgent {
    background: #110a0a;
    border: 1px solid #4a1515;
    border-radius: 10px;
    padding: 18px 22px;
    color: #fc8181;
    font-size: 0.92rem;
}

/* ── SCAN METADATA ── */
.scan-meta-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 12px;
}
.scan-meta-item {
    background: #0a1016;
    border: 1px solid #141e2b;
    border-radius: 8px;
    padding: 10px 14px;
}
.scan-meta-label { font-size: 0.7rem; color: #2d5a7a; font-family: 'DM Mono', monospace; text-transform: uppercase; letter-spacing: 0.05em; }
.scan-meta-value { font-size: 0.88rem; color: #8b9cb0; margin-top: 2px; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #080c10; }
::-webkit-scrollbar-thumb { background: #1a2332; border-radius: 3px; }

/* ── EXPANDER ── */
.streamlit-expanderHeader {
    background: #0d1218 !important;
    border: 1px solid #1a2332 !important;
    border-radius: 8px !important;
    color: #8b9cb0 !important;
}

/* ── CAPTION ── */
.caption-bar {
    text-align: center;
    font-size: 0.72rem;
    color: #1e3a5f;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.06em;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #0d1420;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if 'users' not in st.session_state:
    st.session_state.users = {"doctor1": "password123"}
if 'user_data' not in st.session_state:
    st.session_state.user_data = {
        "doctor1": [
            {"Date": "2026-01-10", "Patient ID": "PX-001", "Consultant": "doctor1",
             "Findings": "Stable Nevus", "Risk": "Low", "Action": "Routine Follow-up"},
            {"Date": "2026-02-14", "Patient ID": "PX-002", "Consultant": "doctor1",
             "Findings": "Melanoma", "Risk": "High", "Action": "Biopsy Required"},
        ]
    }
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'scan_history' not in st.session_state:
    st.session_state.scan_history = []


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def generate_gradcam(model, input_tensor, target_class):
    feature_layer = model.features[-1]
    gradients, activations = [], []

    def save_gradient(grad): gradients.append(grad)
    def save_activation(module, inp, output): activations.append(output)

    handle_grad = feature_layer.register_full_backward_hook(lambda m, i, o: save_gradient(o[0]))
    handle_act  = feature_layer.register_forward_hook(save_activation)
    model.zero_grad()
    out = model(input_tensor)
    out[0, target_class].backward()
    pooled = torch.mean(gradients[0], dim=[0, 2, 3])
    for i in range(activations[0].shape[1]):
        activations[0][:, i, :, :] *= pooled[i]
    heatmap = torch.mean(activations[0], dim=1).squeeze()
    heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
    heatmap /= np.max(heatmap) + 1e-10
    handle_grad.remove(); handle_act.remove()
    return heatmap


@st.cache_resource
def load_clinical_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SkinAttentionModel(num_classes=7)
    model.load_state_dict(torch.load('skin_attention_model.pth', map_location=device))
    model.to(device); model.eval()
    return model, device


LABELS = [
    'Actinic keratoses',
    'Basal cell carcinoma',
    'Benign keratosis',
    'Dermatofibroma',
    'Melanoma',
    'Nevus',
    'Vascular lesions',
]
URGENT = {'Melanoma', 'Basal cell carcinoma', 'Actinic keratoses'}

LABEL_INFO = {
    'Actinic keratoses':    ("Pre-cancerous lesion caused by UV exposure.",       "Dermatology referral within 2 weeks."),
    'Basal cell carcinoma': ("Most common skin cancer; locally invasive.",         "Surgical excision or Mohs surgery recommended."),
    'Benign keratosis':     ("Non-cancerous skin growth; seborrheic origin.",      "No treatment required; monitor annually."),
    'Dermatofibroma':       ("Benign fibrous nodule; commonly on extremities.",    "Observation; excision if symptomatic."),
    'Melanoma':             ("Aggressive malignant tumour; high metastatic risk.", "Immediate excisional biopsy & oncology referral."),
    'Nevus':                ("Common mole; benign melanocytic growth.",            "Routine dermoscopy follow-up every 12 months."),
    'Vascular lesions':     ("Abnormal blood vessel proliferation.",               "Laser therapy or sclerotherapy as appropriate."),
}


def confidence_bar_html(pct: float, danger: bool = False) -> str:
    cls = "conf-bar-fill-danger" if danger else "conf-bar-fill"
    return f"""
    <div class='conf-bar-wrap'>
        <div class='{cls}' style='width:{pct:.1f}%'></div>
    </div>"""


def prob_distribution_html(probs_list):
    rows = ""
    top_idx = int(np.argmax(probs_list))
    for i, (label, p) in enumerate(zip(LABELS, probs_list)):
        pct = p * 100
        is_top = (i == top_idx)
        fill_cls = "prob-fill-top" if is_top else "prob-fill"
        label_col = "#e8edf2" if is_top else "#8b9cb0"
        rows += f"""
        <div class='prob-row'>
            <span class='prob-name' style='color:{label_col};font-weight:{"600" if is_top else "400"}'>{label}</span>
            <div class='prob-track'><div class='{fill_cls}' style='width:{pct:.1f}%'></div></div>
            <span class='prob-pct' style='color:{"#2d7dd2" if is_top else "#3d6a96"}'>{pct:.1f}%</span>
        </div>"""
    return f"<div style='padding:4px 0'>{rows}</div>"


model, device = load_clinical_model()


# ─────────────────────────────────────────────
#  AUTH SCREEN
# ─────────────────────────────────────────────
if not st.session_state.authenticated:
    st.markdown("<div class='auth-head'>Skin Cancer Detection</div>", unsafe_allow_html=True)
    st.markdown("<div class='auth-logo'>🔬</div>", unsafe_allow_html=True)
    st.markdown("<div class='auth-title'>DermAI Pro</div>", unsafe_allow_html=True)
    st.markdown("<div class='auth-sub'>CLINICAL ACCESS PORTAL · v3.2</div>", unsafe_allow_html=True)

    tab_login, tab_signup = st.tabs(["Sign In", "Register"])

    with tab_login:
        user_in = st.text_input("Username", placeholder="clinician_id", key="li_user")
        pw_in   = st.text_input("Password", type="password", placeholder="••••••••", key="li_pw")
        if st.button("Sign In →", use_container_width=True):
            if user_in in st.session_state.users and st.session_state.users[user_in] == pw_in:
                st.session_state.authenticated = True
                st.session_state.current_user  = user_in
                if user_in not in st.session_state.user_data:
                    st.session_state.user_data[user_in] = []
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")

    with tab_signup:
        new_u  = st.text_input("Choose Username", placeholder="dr_yourname", key="su_user")
        new_p  = st.text_input("Choose Password", type="password", placeholder="min. 8 characters", key="su_pw")
        new_p2 = st.text_input("Confirm Password", type="password", placeholder="repeat password", key="su_pw2")
        if st.button("Create Account →", use_container_width=True):
            if not new_u or not new_p:
                st.warning("Please fill all fields.")
            elif new_p != new_p2:
                st.error("Passwords do not match.")
            elif new_u in st.session_state.users:
                st.warning("Username already taken.")
            else:
                st.session_state.users[new_u] = new_p
                st.session_state.user_data[new_u] = []
                st.success("Account created. You may now sign in.")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='caption-bar'>
    FOR CLINICAL DECISION SUPPORT ONLY · NOT A SUBSTITUTE FOR PROFESSIONAL MEDICAL JUDGMENT
    </div>""", unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='padding:14px 0 20px'>
        <div class='brand-name'>🔬 DermAI Pro</div>
        <div class='brand-sub'>Clinical Diagnostics Suite</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:0 0 16px'>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='card' style='padding:16px 18px;margin-bottom:12px'>
        <div style='font-size:0.72rem;color:#2d5a7a;font-family:DM Mono,monospace;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px'>Active Session</div>
        <div style='font-size:1rem;font-weight:600;color:#e8edf2'>Dr. {(st.session_state.current_user or "User").capitalize()}</div>
        <div style='margin-top:10px'>
            <span class='status-dot'></span>
            <span style='font-size:0.75rem;color:#38a169'>System Online</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Patient Context**")
    p_id   = st.text_input("Patient ID",   value="PX-001", label_visibility="collapsed",
                            placeholder="Patient ID e.g. PX-001")
    p_age  = st.text_input("Age / Sex",    value="",        placeholder="e.g. 45 / M")
    p_site = st.selectbox("Lesion Site",
                          ["", "Back", "Chest", "Face", "Arm", "Leg", "Hand", "Scalp", "Neck", "Other"],
                          label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='info-row'><span class='info-key'>Device</span><span class='info-val'>{device.type.upper()}</span></div>
    <div class='info-row'><span class='info-key'>XAI Engine</span><span class='info-val'>Grad-CAM v2</span></div>
    <div class='info-row'><span class='info-key'>Model</span><span class='info-val'>Attention ResNet-50</span></div>
    <div class='info-row'><span class='info-key'>Classes</span><span class='info-val'>7 Dermatological</span></div>
    <div class='info-row' style='border:none'><span class='info-key'>Build</span><span class='info-val'>3.2-stable</span></div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔓  Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.current_user  = None
        st.rerun()

    st.markdown("""
    <div style='margin-top:auto;padding-top:24px;font-size:0.68rem;color:#1a2e42;font-family:DM Mono,monospace;
                text-align:center;line-height:1.6'>
        FOR CLINICAL SUPPORT ONLY<br>NOT A DIAGNOSTIC SUBSTITUTE
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown("""
<div style='display:flex;align-items:baseline;gap:14px;margin-bottom:4px'>
    <h1 style='font-size:1.7rem;font-weight:600;margin:0'>DermAI Pro</h1>
    <span class='badge badge-info'>Explainable Clinical Suite</span>
</div>
<p style='color:#3d5a7a;font-size:0.85rem;margin-bottom:24px;font-family:DM Mono,monospace;
          letter-spacing:.03em'>
    ATTENTION-BASED MELANOMA DETECTION · HAM10000 · 7-CLASS CLASSIFICATION
</p>
""", unsafe_allow_html=True)

tab_diag, tab_xai, tab_hist, tab_about = st.tabs([
    "  Diagnostic Hub  ",
    "  XAI Visualiser  ",
    "  Patient History  ",
    "  About & Methods  ",
])


# ═══════════════════════════════════════════
#  TAB 1 — DIAGNOSTIC HUB
# ═══════════════════════════════════════════
with tab_diag:

    upload_col, gap, info_col = st.columns([3, 0.15, 1.4])

    with upload_col:
        st.markdown("""
        <div class='section-header'>
            <div class='section-icon'>📂</div>
            <span class='section-title'>Upload Dermoscopic Scan</span>
        </div>""", unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Drop scan here or click to browse",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded_file:
            image   = Image.open(uploaded_file).convert('RGB')
            img_np  = np.array(image)
            gray    = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            hsv     = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            h, w    = img_np.shape[:2]

            # ── smart gate ──
            texture_val = cv2.Laplacian(gray, cv2.CV_64F).var()
            is_lesion   = (texture_val > 150) and (np.mean(hsv[:, :, 1]) > 35)

            # ── source image ──
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.image(image, use_container_width=True, caption=f"Source — {uploaded_file.name}")

            # ── scan metadata ──
            st.markdown(f"""
            <div class='scan-meta-grid'>
                <div class='scan-meta-item'>
                    <div class='scan-meta-label'>Dimensions</div>
                    <div class='scan-meta-value'>{w} × {h} px</div>
                </div>
                <div class='scan-meta-item'>
                    <div class='scan-meta-label'>File</div>
                    <div class='scan-meta-value'>{uploaded_file.name}</div>
                </div>
                <div class='scan-meta-item'>
                    <div class='scan-meta-label'>Texture Score</div>
                    <div class='scan-meta-value'>{texture_val:.1f}</div>
                </div>
                <div class='scan-meta-item'>
                    <div class='scan-meta-label'>Saturation (mean)</div>
                    <div class='scan-meta-value'>{np.mean(hsv[:,:,1]):.1f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with info_col:
        st.markdown("""
        <div class='section-header'>
            <div class='section-icon'>ℹ️</div>
            <span class='section-title'>Guidelines</span>
        </div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class='card' style='font-size:0.82rem;line-height:1.75;'>
            <p style='color:#3d5a7a;margin-bottom:8px;font-size:0.7rem;letter-spacing:.06em;font-family:DM Mono,monospace;text-transform:uppercase'>Input Requirements</p>
            <p>• Dermoscopic / clinical photo</p>
            <p>• JPG or PNG format</p>
            <p>• Min. 200 × 200 px</p>
            <p>• Single lesion, well-lit</p>
            <p>• No watermarks or annotations</p>
        </div>
        <div class='card' style='font-size:0.82rem;line-height:1.75;margin-top:0'>
            <p style='color:#3d5a7a;margin-bottom:8px;font-size:0.7rem;letter-spacing:.06em;font-family:DM Mono,monospace;text-transform:uppercase'>Risk Tiers</p>
            <p><span class='badge badge-high'>HIGH</span>&nbsp; Biopsy recommended</p>
            <p style='margin-top:8px'><span class='badge badge-low'>LOW</span>&nbsp; Routine follow-up</p>
        </div>
        """, unsafe_allow_html=True)

    # ── RESULTS SECTION ──
    if uploaded_file:
        st.markdown("<hr>", unsafe_allow_html=True)

        if not is_lesion:
            st.markdown("""
            <div class='alert-normal'>
                <span style='font-size:1.4rem'>✅</span>
                <div>
                    <strong>No significant lesion detected</strong><br>
                    <span style='font-size:0.8rem;color:#38a169;opacity:.75'>
                    Low texture complexity and saturation — image appears to show normal skin or non-lesional tissue.
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            # ── inference ──
            tfm = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            input_t  = tfm(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_t)
            probs    = F.softmax(output, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            conf     = float(probs[pred_idx])
            res      = LABELS[pred_idx]
            is_urgent = res in URGENT
            desc, action = LABEL_INFO[res]

            # ── grad-cam ──
            heatmap  = generate_gradcam(model, input_t, pred_idx)
            hmap_res = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
            hmap_col = cv2.applyColorMap(np.uint8(255 * hmap_res), cv2.COLORMAP_JET)
            overlay  = cv2.addWeighted(img_np, 0.55, cv2.cvtColor(hmap_col, cv2.COLOR_BGR2RGB), 0.45, 0)

            r1, r2, r3 = st.columns([1.6, 1.6, 1.3])

            with r1:
                st.markdown("""
                <div class='section-header'>
                    <div class='section-icon'>🎯</div>
                    <span class='section-title'>Primary Prediction</span>
                </div>""", unsafe_allow_html=True)

                badge_html = f"<span class='badge badge-{'high' if is_urgent else 'low'}'>{'HIGH RISK' if is_urgent else 'LOW RISK'}</span>"
                card_cls   = "card-danger" if is_urgent else "card-safe"

                st.markdown(f"""
                <div class='{card_cls}'>
                    <div class='pred-meta'>Top-1 Classification</div>
                    <div class='pred-label'>{res}</div>
                    {confidence_bar_html(conf * 100, danger=is_urgent)}
                    <div style='display:flex;align-items:center;gap:12px;margin-bottom:14px'>
                        <div class='pred-conf'>{conf*100:.1f}%</div>
                        <div>
                            <div style='font-size:0.72rem;color:#3d5a7a;font-family:DM Mono,monospace;letter-spacing:.04em'>CONFIDENCE</div>
                            <div style='margin-top:4px'>{badge_html}</div>
                        </div>
                    </div>
                    <div style='font-size:0.82rem;color:#8b9cb0;line-height:1.6;border-top:1px solid #1a2332;padding-top:12px'>
                        <strong style='color:#c8d6e5'>Description:</strong> {desc}
                    </div>
                </div>

                <div class='{"card-danger" if is_urgent else "card"}' style='padding:18px 20px'>
                    <div style='font-size:0.7rem;color:#3d5a7a;font-family:DM Mono,monospace;letter-spacing:.06em;text-transform:uppercase;margin-bottom:8px'>
                        Clinical Recommendation
                    </div>
                    <div style='font-size:0.9rem;color:{"#fc8181" if is_urgent else "#68d391"}'>
                        {"⚠️ " if is_urgent else "✔️ "}{action}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with r2:
                st.markdown("""
                <div class='section-header'>
                    <div class='section-icon'>📊</div>
                    <span class='section-title'>Class Probabilities</span>
                </div>""", unsafe_allow_html=True)
                st.markdown(f"<div class='card'>{prob_distribution_html(probs)}</div>", unsafe_allow_html=True)

            with r3:
                st.markdown("""
                <div class='section-header'>
                    <div class='section-icon'>🌡️</div>
                    <span class='section-title'>XAI Heatmap</span>
                </div>""", unsafe_allow_html=True)
                st.image(overlay, use_container_width=True, caption="Grad-CAM Overlay")

                # ── top-3 ──
                top3_idx = np.argsort(probs)[::-1][:3]
                st.markdown("""
                <div class='card' style='margin-top:12px;padding:16px 18px'>
                    <div style='font-size:0.7rem;color:#3d5a7a;font-family:DM Mono,monospace;letter-spacing:.06em;text-transform:uppercase;margin-bottom:10px'>Top-3 Differential</div>
                """, unsafe_allow_html=True)
                for rank, i in enumerate(top3_idx):
                    st.markdown(f"""
                    <div style='display:flex;justify-content:space-between;padding:5px 0;
                                border-bottom:1px solid #111820;font-size:0.8rem'>
                        <span style='color:{"#e8edf2" if rank==0 else "#8b9cb0"}'>{LABELS[i]}</span>
                        <span style='font-family:DM Mono,monospace;color:{"#2d7dd2" if rank==0 else "#3d5a7a"}'>{probs[i]*100:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # ── save to history ──
            new_entry = {
                "Date": time.strftime("%Y-%m-%d %H:%M"),
                "Patient ID": p_id,
                "Consultant": st.session_state.current_user,
                "Findings": res,
                "Confidence": f"{conf*100:.1f}%",
                "Risk": "High" if is_urgent else "Low",
                "Action": action,
            }
            if st.session_state.current_user not in st.session_state.user_data:
                st.session_state.user_data[st.session_state.current_user] = []

            # avoid duplicate on rerun
            existing = st.session_state.user_data[st.session_state.current_user]
            if not existing or existing[-1].get("Findings") != res or existing[-1].get("Patient ID") != p_id:
                st.session_state.user_data[st.session_state.current_user].append(new_entry)


# ═══════════════════════════════════════════
#  TAB 2 — XAI VISUALISER
# ═══════════════════════════════════════════
with tab_xai:
    st.markdown("""
    <div class='section-header' style='margin-top:8px'>
        <div class='section-icon'>🔍</div>
        <span class='section-title'>Explainability Engine</span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='card card-accent'>
            <p style='color:#2d7dd2;font-size:0.7rem;font-family:DM Mono,monospace;letter-spacing:.06em;text-transform:uppercase;margin-bottom:10px'>Grad-CAM</p>
            <p style='font-size:0.9rem;color:#c8d6e5;font-weight:500;margin-bottom:8px'>Gradient-weighted Class Activation Mapping</p>
            <p style='font-size:0.82rem;line-height:1.75;'>
            Grad-CAM computes the gradient of the class score with respect to the final
            convolutional feature maps. These gradients are global-average-pooled to produce
            channel weights, which are then combined with the activations to produce a spatial
            saliency map — highlighting which regions of the input most influenced the prediction.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='card card-accent'>
            <p style='color:#2d7dd2;font-size:0.7rem;font-family:DM Mono,monospace;letter-spacing:.06em;text-transform:uppercase;margin-bottom:10px'>Attention Mechanism</p>
            <p style='font-size:0.9rem;color:#c8d6e5;font-weight:500;margin-bottom:8px'>CBAM — Convolutional Block Attention Module</p>
            <p style='font-size:0.82rem;line-height:1.75;'>
            CBAM applies both channel-wise and spatial attention sequentially. The channel
            attention focuses on "what" is meaningful, while spatial attention focuses on "where."
            This dual-attention mechanism forces the model to attend to diagnostically relevant
            lesion features, suppressing skin background noise and improving reliability.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='section-header'>
        <div class='section-icon'>🏗️</div>
        <span class='section-title'>Model Architecture</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
    <p style='color:#3d5a7a;font-size:0.7rem;font-family:DM Mono,monospace;letter-spacing:.06em;text-transform:uppercase;margin-bottom:14px'>
    Pipeline: Input → Backbone → Attention → Classifier → XAI
    </p>
    """, unsafe_allow_html=True)

    arch_cols = st.columns(5)
    stages = [
        ("Input", "224×224 RGB\nDermoscopic Image", "📷"),
        ("Backbone", "ResNet-50\nPre-trained ImageNet", "🔧"),
        ("Attention", "CBAM Module\nChannel + Spatial", "🎯"),
        ("Head", "Global Avg Pool\nFC → 7 classes", "🧮"),
        ("XAI", "Grad-CAM\nSaliency Overlay", "🌡️"),
    ]
    for col, (title, desc, icon) in zip(arch_cols, stages):
        with col:
            st.markdown(f"""
            <div class='card' style='text-align:center;padding:18px 12px'>
                <div style='font-size:1.6rem'>{icon}</div>
                <div style='font-size:0.82rem;font-weight:600;color:#c8d6e5;margin:8px 0 4px'>{title}</div>
                <div style='font-size:0.72rem;color:#3d5a7a;line-height:1.5;white-space:pre-line'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='section-header'>
        <div class='section-icon'>📈</div>
        <span class='section-title'>Dataset & Performance</span>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        ("Dataset",    "HAM10000",  "10,015 dermoscopic images"),
        ("Classes",    "7",         "Skin lesion categories"),
        ("Backbone",   "ResNet-50", "Pretrained + fine-tuned"),
        ("XAI Method", "Grad-CAM",  "Layer-wise saliency"),
    ]
    for col, (label, val, sub) in zip([m1, m2, m3, m4], metrics):
        with col:
            st.markdown(f"""
            <div class='card' style='text-align:center;padding:18px'>
                <div style='font-size:1.4rem;font-weight:600;color:#2d7dd2;font-family:DM Mono,monospace'>{val}</div>
                <div style='font-size:0.75rem;color:#c8d6e5;margin:4px 0 2px;font-weight:500'>{label}</div>
                <div style='font-size:0.7rem;color:#3d5a7a'>{sub}</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════
#  TAB 3 — PATIENT HISTORY
# ═══════════════════════════════════════════
with tab_hist:
    st.markdown("""
    <div class='section-header' style='margin-top:8px'>
        <div class='section-icon'>📋</div>
        <span class='section-title'>Longitudinal Patient Records</span>
    </div>
    """, unsafe_allow_html=True)

    data = st.session_state.user_data.get(st.session_state.current_user, [])

    if data:
        df = pd.DataFrame(data)

        # ── summary metrics ──
        sm1, sm2, sm3, sm4 = st.columns(4)
        total  = len(df)
        high   = int((df["Risk"] == "High").sum()) if "Risk" in df.columns else 0
        low    = int((df["Risk"] == "Low").sum())  if "Risk" in df.columns else 0
        unique = df["Patient ID"].nunique()         if "Patient ID" in df.columns else "-"

        for col, (val, lbl) in zip([sm1, sm2, sm3, sm4],
                                   [(total,"Total Scans"),(high,"High Risk"),
                                    (low,"Low Risk"),(unique,"Patients")]):
            with col:
                st.markdown(f"""
                <div class='card' style='text-align:center;padding:16px'>
                    <div style='font-size:1.5rem;font-weight:600;color:#2d7dd2;font-family:DM Mono,monospace'>{val}</div>
                    <div style='font-size:0.75rem;color:#8b9cb0;margin-top:4px'>{lbl}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── filter ──
        fc1, fc2 = st.columns([2, 1])
        with fc1:
            search = st.text_input("Filter by Patient ID or Findings", placeholder="Search...", key="hist_search")
        with fc2:
            risk_filter = st.selectbox("Risk Level", ["All", "High", "Low"], key="risk_sel")

        filtered = df.copy()
        if search:
            mask = filtered.apply(lambda r: search.lower() in str(r).lower(), axis=1)
            filtered = filtered[mask]
        if risk_filter != "All":
            filtered = filtered[filtered["Risk"] == risk_filter]

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Risk": st.column_config.TextColumn("Risk", width="small"),
                "Confidence": st.column_config.TextColumn("Confidence", width="small"),
            }
        )

        # ── export ──
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇  Export Filtered Records as CSV",
            data=csv,
            file_name=f"dermAI_{st.session_state.current_user}_{time.strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    else:
        st.markdown("""
        <div class='card' style='text-align:center;padding:48px;color:#3d5a7a'>
            <div style='font-size:2rem;margin-bottom:12px'>📭</div>
            No diagnostic records found for this account.<br>
            <span style='font-size:0.82rem'>Upload a scan in the Diagnostic Hub to begin.</span>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
#  TAB 4 — ABOUT & METHODS
# ═══════════════════════════════════════════
with tab_about:
    a1, a2 = st.columns([2, 1.2])

    with a1:
        st.markdown("""
        <div class='section-header' style='margin-top:8px'>
            <div class='section-icon'>🧬</div>
            <span class='section-title'>Project Overview</span>
        </div>
        <div class='card card-accent'>
            <p style='font-size:0.88rem;line-height:1.85;color:#adbac7'>
            <strong style='color:#e8edf2'>DermAI Pro</strong> is an end-to-end web-deployed
            Explainable AI (XAI) system for dermoscopic lesion classification, developed as a
            final-year research project in the Health domain.
            </p>
            <p style='font-size:0.88rem;line-height:1.85;color:#adbac7;margin-top:12px'>
            Melanoma accounts for the majority of skin cancer deaths despite representing a
            minority of cases. Early and accurate detection is paramount. This system addresses
            the critical <em>trust gap</em> in medical AI by providing not only a classification
            but a transparent, spatially grounded explanation via Grad-CAM heatmaps.
            </p>
            <p style='font-size:0.88rem;line-height:1.85;color:#adbac7;margin-top:12px'>
            The core innovation is the <strong style='color:#e8edf2'>Attention-Based Hybrid Network</strong>:
            a ResNet-50 backbone augmented with a CBAM attention module that forces the network
            to focus on diagnostically salient lesion features, reducing background noise and
            improving reliability on imbalanced clinical datasets.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='section-header' style='margin-top:4px'>
            <div class='section-icon'>⚠️</div>
            <span class='section-title'>Limitations & Disclaimer</span>
        </div>
        <div class='card-danger' style='font-size:0.85rem;line-height:1.75'>
            This system is intended as a <strong>clinical decision support tool only</strong>.
            It is not a replacement for professional dermatological assessment or biopsy.
            All outputs must be reviewed and validated by a qualified clinician before
            any diagnostic or treatment decision is made. Model performance may vary
            across patient populations, image acquisition devices, and lesion subtypes
            not represented in the HAM10000 training set.
        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div class='section-header' style='margin-top:8px'>
            <div class='section-icon'>🏷️</div>
            <span class='section-title'>Supported Classes</span>
        </div>
        """, unsafe_allow_html=True)

        for label, (desc, action) in LABEL_INFO.items():
            risk_tag = "badge-high" if label in URGENT else "badge-low"
            risk_txt = "HIGH" if label in URGENT else "LOW"
            st.markdown(f"""
            <div class='card' style='padding:14px 16px;margin-bottom:8px'>
                <div style='display:flex;justify-content:space-between;align-items:center'>
                    <span style='font-size:0.85rem;font-weight:500;color:#c8d6e5'>{label}</span>
                    <span class='badge {risk_tag}'>{risk_txt}</span>
                </div>
                <div style='font-size:0.75rem;color:#3d5a7a;margin-top:5px;line-height:1.5'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class='caption-bar'>
    DermAI Pro v3.2 · Attention-Based Melanoma Detection ·
    HAM10000 Dataset · For Clinical Decision Support Only ·
    Not a Substitute for Professional Medical Judgment
</div>
""", unsafe_allow_html=True)
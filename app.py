# Developed by Alexandra de Almeida Ferreira

import os
import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F
import io
import time

# =============================
# OPTIONAL PDF (SAFE IMPORT)
# =============================
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except:
    REPORTLAB_AVAILABLE = False

from src.model import SimpleCNN
from src.gradcam import GradCAM, show_cam_on_image

st.set_page_config(page_title="Grad-CAM Explorer", layout="wide")

# =============================
# STYLE (UNCHANGED)
# =============================
st.markdown("""
<style>
.stApp { background:#020617; color:#e2e8f0; }
.left-panel { border-right:1px solid #1f2231; padding-right:12px; }
.right-panel { background:#050a18; padding:20px; border-radius:16px; }

.stButton>button {
    width:100%;
    background:linear-gradient(90deg,#6366f1,#8b5cf6);
    border-radius:10px;
    transition: all 0.25s ease;
}
.stButton>button:hover {
    box-shadow:0 0 20px rgba(139,92,246,0.8);
}
.stButton>button:active {
    box-shadow:0 0 30px rgba(99,102,241,1);
}

.pipe { border:1px solid #1f2231; border-radius:12px; padding:12px; text-align:center; }
.active { border:1px solid #6366f1; box-shadow:0 0 20px rgba(99,102,241,0.6); }
.stFileUploader { border:1px solid #1f2231; border-radius:12px; padding:12px; background:transparent; }
.stFileUploader section { background:transparent; border:0; }
.card { border:1px solid #1f2231; border-radius:14px; padding:16px; margin-top:20px; background:#020617; }
.footer { text-align:center; opacity:0.6; margin-top:40px; }
</style>
""", unsafe_allow_html=True)

# =============================
# HEADER
# =============================
st.title("🔥 Grad-CAM Explorer")
st.caption("Visualizing Deep Learning Interpretability")

# =============================
# MODEL
# =============================
@st.cache_resource
def load_model():
    model_path = "models/model.pth"
    if not os.path.exists(model_path):
        return None
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

model = load_model()

if model is None:
    st.error("Model not found")
    st.stop()

gradcam = GradCAM(model, model.conv2)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

class_names = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

# =============================
# UTILS
# =============================
def pil_to_buffer(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

# =============================
# STATE
# =============================
if "stage" not in st.session_state:
    st.session_state.stage = "upload"

if "results" not in st.session_state:
    st.session_state.results = None

if "upload_key" not in st.session_state:
    st.session_state.upload_key = "upload_0"

# =============================
# LAYOUT
# =============================
left, right = st.columns([0.2,0.8])

# =============================
# LEFT PANEL
# =============================
with left:
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"], key=st.session_state.upload_key)

    alpha = st.slider("Blend intensity", 0.0, 1.0, 0.5)

    mode = st.radio(
        "Explainability Mode",
        ["Grad-CAM", "Grad-CAM++", "Saliency", "Integrated Gradients"]
    )

    compare_all = st.toggle("Compare all methods", False)

    run = st.button("Run Analysis")

    st.subheader("System")
    st.write("🟢 Ready" if uploaded else "🟡 Waiting input")

    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# RIGHT PANEL
# =============================
with right:
    st.markdown('<div class="right-panel">', unsafe_allow_html=True)

    stage = st.session_state.stage

    p1,p2,p3,p4 = st.columns(4)

    def pipe(col, icon, title, desc, key):
        active = stage == key
        with col:
            st.markdown(f"""
            <div class="pipe {'active' if active else ''}">
            {icon}<br>
            <b>{title}</b><br>
            <span style="opacity:0.6;font-size:11px;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    pipe(p1,"📤","UPLOAD","Provide image","upload")
    pipe(p2,"🧠","MODEL","Run CNN","model")
    pipe(p3,"🔥","ATTENTION","Explainability","cam")
    pipe(p4,"📊","RESULT","Visualization","result")

    st.markdown("""
    <div class="card">
    <h3>🚀 Explainability Lab</h3>
    Compare multiple interpretability methods:
    </div>
    """, unsafe_allow_html=True)

    # =============================
    # RUN
    # =============================
    if run and uploaded:
        st.session_state.stage = "model"
        st.rerun()

    if st.session_state.stage == "model" and uploaded:

        image = Image.open(uploaded).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)

        probs = F.softmax(outputs, dim=1)[0]
        top3_prob, top3_idx = torch.topk(probs, 3)

        st.session_state.stage = "cam"
        st.session_state._temp = (image, input_tensor, probs, top3_idx, top3_prob)
        st.rerun()

    if st.session_state.stage == "cam":

        image, input_tensor, probs, top3_idx, top3_prob = st.session_state._temp

        img_np = np.array(image.resize((32,32))) / 255.0

        cam_base = gradcam.generate(input_tensor, top3_idx[0].item())
        cam_pp = cam_base**1.5
        sal = np.max(np.abs(input_tensor.numpy()), axis=1)[0]
        ig = cam_base

        results = {
            "Grad-CAM": show_cam_on_image(img_np, cam_base),
            "Grad-CAM++": show_cam_on_image(img_np, cam_pp),
            "Saliency": sal,
            "Integrated Gradients": ig
        }

        st.session_state.results = {
            "image": image,
            "results": results,
            "pred": class_names[top3_idx[0]],
            "confidence": top3_prob[0].item(),
            "top3_idx": top3_idx,
            "top3_prob": top3_prob
        }

        st.session_state.stage = "result"
        st.rerun()

    # =============================
    # RESULTS
    # =============================
    if st.session_state.results:

        data = st.session_state.results

        st.markdown("## 🧠 Results")

        c1,c2 = st.columns(2)
        with c1:
            st.image(data["image"], use_container_width=True)
        with c2:
            st.image(data["results"][mode], use_container_width=True)

        if compare_all:
            cols = st.columns(4)
            for col, (name, img) in zip(cols, data["results"].items()):
                col.image(img, caption=name, use_container_width=True)

        st.markdown(f"### 🎯 Prediction: **{data['pred']}**")
        st.progress(float(data["confidence"]))
        st.write(f"{data['confidence']*100:.2f}% confidence")

        st.markdown("### Top 3 Predictions")
        for i in range(3):
            p = data["top3_prob"][i].item()
            label = class_names[data["top3_idx"][i]]
            st.write(f"{label} → {p*100:.2f}% | uncertainty {(1-p)*100:.2f}%")

        # =============================
        # BUTTONS
        # =============================
        c1, c2 = st.columns(2)

        with c1:
            if REPORTLAB_AVAILABLE:

                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=A4)
                styles = getSampleStyleSheet()

                elements = [
                    Paragraph("AI Explainability Report", styles["Title"]),
                    Spacer(1,12),
                    Paragraph(f"Prediction: {data['pred']}", styles["Normal"]),
                    Paragraph(f"Confidence: {data['confidence']*100:.2f}%", styles["Normal"]),
                ]

                for i in range(3):
                    p = data["top3_prob"][i].item()
                    label = class_names[data["top3_idx"][i]]
                    elements.append(Paragraph(f"{label}: {p*100:.2f}%", styles["Normal"]))

                elements.append(Spacer(1,12))

                # ORIGINAL
                elements.append(RLImage(pil_to_buffer(data["image"]), width=6*cm, height=6*cm))

                # RESULT
                res_img = data["results"][mode]
                buf = io.BytesIO()
                Image.fromarray((res_img * 255).astype(np.uint8)).save(buf, format="PNG")
                buf.seek(0)

                elements.append(Spacer(1,12))
                elements.append(RLImage(buf, width=6*cm, height=6*cm))

                elements.append(Spacer(1,20))
                elements.append(Paragraph("Developed by Alexandra de Almeida Ferreira", styles["Normal"]))
                elements.append(Paragraph("GitHub: github.com/dealmeidaferreiraAlexandra", styles["Normal"]))
                elements.append(Paragraph("LinkedIn: linkedin.com/in/dealmeidaferreira", styles["Normal"]))

                doc.build(elements)

                st.download_button("📄 PDF", buffer.getvalue(), "report.pdf", use_container_width=True)

        with c2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.upload_key = f"upload_{time.time()}"
                st.session_state.results = None
                st.session_state.stage = "upload"
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# FOOTER
# =============================
st.markdown("""
<div class='footer'>
Developed by <b>Alexandra de Almeida Ferreira</b><br><br>
🔗 <a href="https://github.com/dealmeidaferreiraAlexandra" target="_blank">GitHub</a> | 
💼 <a href="https://www.linkedin.com/in/dealmeidaferreira" target="_blank">LinkedIn</a>
</div>
""", unsafe_allow_html=True)

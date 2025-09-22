import streamlit as st
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import cv2
import os
import base64

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="SafeVision",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# ----------------- BACKGROUND IMAGE -----------------
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

BASE_DIR = Path(__file__).parent
bg_path = BASE_DIR / "assets" / "bg.png"  
if bg_path.exists():
    bg_base64 = get_base64_image(bg_path)
    st.markdown(
        f"""
        <style>
        /* Full background */
        .stAppViewContainer {{
            background: url("data:image/png;base64,{bg_base64}") no-repeat center center fixed !important;
            background-size: cover !important;
        }}

        

        /* Content container */
        .stAppViewBlockContainer {{
            background: rgba(255,255,255,0.85);
            border-radius: 12px;
            padding: 2rem;
            z-index: 1;
            position: relative;
        }}

        /* Text colors */
        h1, h2, h3, h4, h5, h6, p, label {{
            color: white !important;
        }}

        /* Buttons */
        div.stButton > button {{
            background-color: #238636;
            color: white !important;
            border-radius: 8px;
            border: none;
        }}
        div.stButton > button:hover {{
            background-color: #2ea043;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# ----------------- HEADER -----------------
st.markdown('<div class="main">', unsafe_allow_html=True)
st.title("🛡️ SafeVision")
st.markdown("Upload an image or choose a sample to see predictions from the trained YOLO model.")
st.markdown("---")

# ----------------- PATHS -----------------
this_dir = Path(__file__).parent
uploads_dir = Path(r"D:\HackathonApp\uploads")
predictions_dir = Path(r"D:\HackathonApp\predictions")
sample_images_dir = this_dir / "sample_images"

uploads_dir.mkdir(parents=True, exist_ok=True)
predictions_dir.mkdir(parents=True, exist_ok=True)

# ----------------- LOAD MODEL -----------------
BASE_DIR = Path(__file__).parent
model_path = BASE_DIR / "train4" / "weights" / "best.pt"
model = YOLO(model_path)

# ----------------- PREDICTION FUNCTION -----------------
def predict_and_save(model, image_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_img_path = output_dir / image_path.name
    output_label_path = output_dir / image_path.with_suffix('.txt').name

    results = model.predict(image_path, conf=0.5)
    result = results[0]

    # Save image with boxes
    img = result.plot()
    cv2.imwrite(str(output_img_path), img)

    # Save labels
    with open(output_label_path, 'w') as f:
        for box in result.boxes:
            cls_id = int(box.cls)
            x_center, y_center, width, height = box.xywhn[0].tolist()
            f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")
    
    return output_img_path, output_label_path

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("Instructions")
    st.write("""
    1. Upload an image or select a sample from the dropdown.  
    2. Click **Predict** to see the model predictions.  
    3. View original and predicted images side by side.  
    4. Download predicted image and labels if needed.
    """)

# ----------------- INPUT SELECTION -----------------
st.subheader("Choose input method:")
input_method = st.radio(
    "Choose input method:",
    ["Upload Image", "Use Sample Image"],
    label_visibility="collapsed"
)

image_path = None

# --- Upload Image ---
if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        save_path = uploads_dir / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        image_path = save_path

# --- Use Sample Image ---
elif input_method == "Use Sample Image":
    if sample_images_dir.exists():
        sample_images = sorted([f.name for f in sample_images_dir.glob("*.png")])
        if sample_images:
            selected_image_name = st.selectbox(
                "Choose a sample image",
                ["Select a sample image"] + sample_images
            )
            if selected_image_name != "Select a sample image":
                image_path = sample_images_dir / selected_image_name
        else:
            st.warning("No sample images found in the sample_images directory.")
    else:
        st.warning(f"Sample images directory {sample_images_dir} does not exist.")

# ----------------- PREDICTION BUTTON -----------------
if image_path:
    if st.button("Predict"):
        st.markdown(f"<div class='main'><h4>Predicting for: {image_path.name}</h4></div>", unsafe_allow_html=True)
        output_img_path, output_label_path = predict_and_save(model, image_path, predictions_dir)

        # Side-by-side display
        col1, col2 = st.columns(2)
        with col1:
            st.image(Image.open(image_path), caption="Original Image", width=400)
        with col2:
            st.image(Image.open(output_img_path), caption="Prediction Result", width=400)

        # Optional download links
        st.markdown("---")
        st.markdown("**Download Results:**")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "Download Predicted Image",
                data=open(output_img_path, "rb").read(),
                file_name=output_img_path.name,
                mime="image/png"
            )
        with col_dl2:
            st.download_button(
                "Download Labels",
                data=open(output_label_path, "r").read(),
                file_name=output_label_path.name,
                mime="text/plain"
            )

st.markdown('</div>', unsafe_allow_html=True)

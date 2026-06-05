import os
# Load local .env file manually if it exists
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"').strip("'")

import time
import glob
import cv2
import numpy as np
import torch
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from openai import OpenAI
import base64
import io

# ==========================================
# 🌟 Premium Custom styling (Glassmorphism & Neon accents)
# ==========================================
st.set_page_config(
    page_title="DeepPCB Inference & VLM Diagnostics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono&display=swap" rel="stylesheet">
    <style>
        /* Base styles */
        * {
            font-family: 'Outfit', sans-serif;
        }
        .stApp {
            background-color: #0F1115;
            color: #E2E8F0;
        }
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #161920 !important;
            border-right: 1px solid rgba(102, 252, 241, 0.1) !important;
        }
        
        /* Titles and Headers */
        h1, h2, h3 {
            color: #66FCF1 !important;
            font-weight: 600 !important;
        }
        .header-title {
            font-size: 2.5rem;
            font-weight: 800 !important;
            background: linear-gradient(90deg, #66FCF1 0%, #45A29E 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .header-subtitle {
            font-size: 1.1rem;
            color: #8A9Aad;
            margin-bottom: 2rem;
        }
        
        /* Card Panel Styling */
        .glass-card {
            background: rgba(30, 41, 59, 0.45);
            backdrop-filter: blur(12px);
            border-radius: 12px;
            border: 1px solid rgba(102, 252, 241, 0.15);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        
        /* Metrics styles */
        .metric-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 2.2rem;
            font-weight: 800;
            color: #66FCF1;
        }
        
        /* RAG & AI report panels */
        .manual-box {
            background: rgba(15, 23, 42, 0.7);
            border-left: 4px solid #45A29E;
            padding: 1rem;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            max-height: 250px;
            overflow-y: auto;
            color: #A0AEC0;
        }
        .report-box {
            background: rgba(22, 25, 32, 0.85);
            border-left: 4px solid #66FCF1;
            padding: 1.2rem;
            border-radius: 8px;
            font-size: 0.95rem;
            line-height: 1.6;
            color: #E2E8F0;
            border: 1px solid rgba(102, 252, 241, 0.1);
        }
        .report-box a {
            color: #00D2FF !important;
            font-weight: bold;
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)

# Defect class mapping
CLASS_NAMES = {
    0: "Open",
    1: "Short",
    2: "Mousebite",
    3: "Spur",
    4: "Spurious Copper",
    5: "Pin-hole"
}

# ==========================================
# 📊 System Utilities & Helper Functions
# ==========================================
@st.cache_resource
def setup_vector_db():
    """Initializes local ChromaDB vector database containing AOI manuals."""
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(ROOT_DIR, "chroma_db")
    client = chromadb.PersistentClient(path=db_path)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name="aoi_service_manuals",
        embedding_function=embedding_func
    )
    
    # Auto-ingest if empty
    if collection.count() == 0:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        manuals_dir = os.path.join(ROOT_DIR, "mock_manuals")
        if os.path.exists(manuals_dir):
            doc_id = 0
            for filename in os.listdir(manuals_dir):
                if filename.endswith(".txt"):
                    filepath = os.path.join(manuals_dir, filename)
                    with open(filepath, 'r') as f:
                        content = f.read()
                    sections = content.split("SECTION ")
                    for i, sec in enumerate(sections):
                        if not sec.strip():
                            continue
                        sec_text = f"SECTION {sec.strip()}"
                        collection.add(
                            documents=[sec_text],
                            metadatas=[{"source": filename, "section": i}],
                            ids=[f"doc_{doc_id}"]
                        )
                        doc_id += 1
    return collection

def query_manual(collection, defect_class):
    """Retrieves context matching defect_class from Vector DB along with metadata sources."""
    if collection is None or collection.count() == 0:
        return "No manual context found.", []
    results = collection.query(query_texts=[defect_class], n_results=2)
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    
    formatted_manuals = []
    sources = []
    for doc, meta in zip(documents, metadatas):
        src_name = meta.get("source", "Unknown Source")
        sec_num = meta.get("section", 0)
        formatted_manuals.append(f"[{src_name} - Section {sec_num}]:\n{doc}")
        sources.append({
            "source": src_name,
            "section": sec_num,
            "text": doc
        })
        
    return "\n\n".join(formatted_manuals), sources

import re

def linkify_citations(report_text):
    """Parses bracket citations like [manual_open.txt - Section 1] and turns them into Markdown links to anchors."""
    # Pattern matching brackets e.g. [manual_open_circuits.txt - Section 1]
    pattern = r'\[([a-zA-Z0-9_\-\.\s]+)\s*-\s*[sS]ection\s*(\d+)\]'
    
    def repl(match):
        full_match = match.group(0)
        filename = match.group(1).strip()
        section = match.group(2).strip()
        # Generate target HTML anchor ID
        anchor_id = f"{filename}_section_{section}".replace(".", "_").replace(" ", "_").lower()
        # Return standard markdown link (compiled natively to clean HTML links by Streamlit)
        return f"[{full_match}](#{anchor_id})"
        
    return re.sub(pattern, repl, report_text)

def get_available_cameras():
    """Detects active webcam device indexes on the local Linux platform."""
    cameras = []
    # Method 1: Check Linux device endpoints directly
    video_devices = glob.glob("/dev/video*")
    for dev in sorted(video_devices):
        try:
            num = int(dev.replace("/dev/video", ""))
            cap = cv2.VideoCapture(num)
            if cap.isOpened():
                cameras.append(num)
                cap.release()
        except Exception:
            pass
    # Method 2: Fallback scan if empty
    if not cameras:
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append(i)
                cap.release()
    return cameras

@st.cache_resource
def load_yolo_model(model_format):
    """Dynamically loads weights based on runtime format selection."""
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(ROOT_DIR, "runs/detect/pcb_inspection/yolov8s_deeppcb/weights")
    
    cuda_available = torch.cuda.is_available()
    
    if model_format == "TensorRT (GPU Compiled)":
        if cuda_available:
            path = os.path.join(weights_dir, "best.engine")
        else:
            path = os.path.join(weights_dir, "best.pt")
    elif model_format == "ONNX Runtime (FP16 Optimized)":
        path = os.path.join(weights_dir, "best.onnx")
    else:
        path = os.path.join(weights_dir, "best.pt")
        
    # Check if selected format weights exist; fallback to PyTorch if missing
    if not os.path.exists(path):
        path = os.path.join(weights_dir, "best.pt")
        
    if not os.path.exists(path):
        return None, f"Model file not found at: {path}"
    try:
        model = YOLO(path)
        return model, None
    except Exception as e:
        return None, str(e)

def get_base64_image(image_path_or_array):
    """Converts image path or numpy array into base64 string for multimodal LLMs."""
    if isinstance(image_path_or_array, str):
        with open(image_path_or_array, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    else:
        # Numpy array
        img = Image.fromarray(cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2RGB))
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

# ==========================================
# 🧠 Multimodal Reasoning LLM Clients
# ==========================================
def generate_local_vllm_report(vllm_url, model_name, defect_class, bbox_coords, confidence_pct, confidence_level, all_defects_summary, retrieved_manuals, base64_crop):
    """Queries local vLLM endpoint for offline diagnostic report."""
    try:
        client = OpenAI(base_url=vllm_url, api_key="placeholder")
        prompt = f"""
You are an expert Senior AOI Field Service Engineer diagnostic system.
Inspect this microscopic crop of a '{defect_class}' defect located at bounding box {bbox_coords}.

YOLO DETECTION TELEMETRY:
- Target Defect Class: {defect_class}
- Target Coordinates: {bbox_coords}
- Detection Confidence Score: {confidence_pct:.1f}%
- Detection Confidence Level: {confidence_level}

ALL DETECTED SUBSTRATE DEFECTS:
{all_defects_summary}

Below is retrieved context from AOI Service Manuals:
----------------------------------------
{retrieved_manuals}
----------------------------------------

PCB / SUBSTRATE VALIDATION RULE:
If the attached image is NOT a Printed Circuit Board (PCB) or a semiconductor/wafer substrate (e.g. if it contains a human, a face, furniture, general room background, or outdoor scenes), you MUST prefix your report with this exact warning block:
"⚠️ **WARNING**: The provided image does not appear to be a Printed Circuit Board (PCB) or semiconductor substrate. The diagnostic reasoning below is based strictly on the detected visual features but may be a false positive due to non-PCB input."
Then continue to write the report, emphasizing that it might be an invalid detection.

Generate a detailed engineering report with:
1. DEFECT SUMMARY: Briefly summarize the class, position, and physical appearance. Explicitly list the YOLO detection confidence percentage ({confidence_pct:.1f}%), its confidence level ({confidence_level}), and summarize any other co-located defects detected on the substrate.
2. ROOT CAUSE DIAGNOSIS: Explain the likely physical/chemical cause. Incorporate details about how the confidence level affects diagnostic priority, and how the presence of other detected defects (if any) points to cumulative system/process line issues.
3. CORRECTIVE ACTION STEP-BY-STEP: Outline exact, numbered service actions the field technician must perform. Be highly specific.

IMPORTANT CITATION RULES:
1. You MUST include inline citations whenever you reference guidelines, steps, or definitions. For example:
   * "According to [manual_open_circuits.txt - Section 1], the process requires..."
   * "The suggested corrective action is to calibrate the nozzle [manual_mousebites.txt - Section 3]."
2. At the very bottom of your report, create a section titled "### 📖 SOURCES REFERENCED" and list each unique file name and section index.
"""
        # If model name has 'vision' or we have base64, attempt multimodal query
        if "vision" in model_name.lower() or "llava" in model_name.lower():
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_crop}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=800
            )
        else:
            # Text-only local vLLM fallback
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800
            )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error connecting to local vLLM endpoint: {e}"

def generate_gemini_report(api_key, defect_class, bbox_coords, confidence_pct, confidence_level, all_defects_summary, retrieved_manuals, crop_image_np):
    """Queries online Gemini 3.5 Flash API for diagnostic report."""
    try:
        genai.configure(api_key=api_key)
        # Convert numpy crop to PIL Image
        img = Image.fromarray(cv2.cvtColor(crop_image_np, cv2.COLOR_BGR2RGB))
        model = genai.GenerativeModel('gemini-3.5-flash')
        prompt = f"""
You are an expert Senior AOI Field Service Engineer diagnostic system. 

YOLO DETECTION TELEMETRY:
- Target Defect Class: {defect_class}
- Target Coordinates: {bbox_coords}
- Detection Confidence Score: {confidence_pct:.1f}%
- Detection Confidence Level: {confidence_level}

ALL DETECTED SUBSTRATE DEFECTS:
{all_defects_summary}

Below is the retrieved context from AOI Technical Service Manuals:
----------------------------------------
{retrieved_manuals}
----------------------------------------

PCB / SUBSTRATE VALIDATION RULE:
If the attached image is NOT a Printed Circuit Board (PCB) or a semiconductor/wafer substrate (e.g. if it contains a human, a face, furniture, general room background, or outdoor scenes), you MUST prefix your report with this exact warning block:
"⚠️ **WARNING**: The provided image does not appear to be a Printed Circuit Board (PCB) or semiconductor substrate. The diagnostic reasoning below is based strictly on the detected visual features but may be a false positive due to non-PCB input."
Then continue to write the report, emphasizing that it might be an invalid detection.

Please review the attached microscopic image of the defect and combine it with the retrieved manual to generate a highly professional diagnostic report.
Provide the report in the following format:
1. **DEFECT SUMMARY**: Briefly summarize the class, position, and physical appearance. Explicitly list the YOLO detection confidence percentage ({confidence_pct:.1f}%), its confidence level ({confidence_level}), and summarize any other co-located defects detected on the substrate.
2. **ROOT CAUSE DIAGNOSIS**: Explain the likely physical/chemical cause. Incorporate details about how the confidence level affects diagnostic priority, and how the presence of other detected defects (if any) points to cumulative system/process line issues.
3. **CORRECTIVE ACTION STEP-BY-STEP**: Outline exact, numbered service actions the field technician must perform. Be highly specific.

IMPORTANT CITATION RULES:
1. You MUST include inline citations whenever you reference guidelines, steps, or definitions. For example:
   * "According to [manual_open_circuits.txt - Section 1], the process requires..."
   * "The suggested corrective action is to calibrate the nozzle [manual_mousebites.txt - Section 3]."
2. At the very bottom of your report, create a section titled "### 📖 SOURCES REFERENCED" and list each unique file name and section index.
"""
        response = model.generate_content([prompt, img])
        return response.text
    except Exception as e:
        return f"Error during Gemini generation: {e}"

# ==========================================
# 📊 Main Application Header
# ==========================================
st.markdown('<div class="header-title">🔍 DeepPCB Inference & VLM Diagnostics</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Edge-Accelerated Vision Engine & Local Vector Manual Retrieval Pipeline</div>', unsafe_allow_html=True)

# Setup database
vector_db = setup_vector_db()

# ==========================================
# 🛠️ Sidebar Configuration Panel
# ==========================================
st.sidebar.markdown("### ⚙️ Engine Configurations")

# 1. Model Format Selector
model_formats = ["PyTorch (FP32 Baseline)", "ONNX Runtime (FP16 Optimized)", "TensorRT (GPU Compiled)"]
model_format = st.sidebar.selectbox(
    "Inference Model Format",
    model_formats
)

# CUDA Availability Check & Warning
cuda_available = torch.cuda.is_available()
if model_format == "TensorRT (GPU Compiled)" and not cuda_available:
    st.sidebar.warning("⚠️ **CUDA GPU Offline**: Local CUDA hardware not detected on this server host. Model will fall back to PyTorch FP32 baseline.")

# 2. Reasoning Provider Configuration
reasoning_providers = ["Llama.cpp (Offline)", "Ollama (Offline)", "Local vLLM (Offline)", "Gemini 3.5 Flash (Online)", "None (Local Fallback)"]
reasoning_provider = st.sidebar.selectbox(
    "VLM Reasoning Engine",
    reasoning_providers
)

if reasoning_provider == "Gemini 3.5 Flash (Online)":
    api_key_input = st.sidebar.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
else:
    api_key_input = None

if reasoning_provider == "Llama.cpp (Offline)":
    vllm_base_url = st.sidebar.text_input("Llama.cpp Server URL", value="http://localhost:8090/v1")
    vllm_model_name = st.sidebar.text_input("Model Alias", value="qwen-vision")
elif reasoning_provider == "Ollama (Offline)":
    vllm_base_url = st.sidebar.text_input("Ollama Base URL", value="http://localhost:11434/v1")
    vllm_model_name = st.sidebar.text_input("Ollama Model Name", value="llava")
elif reasoning_provider == "Local vLLM (Offline)":
    vllm_base_url = st.sidebar.text_input("vLLM Base URL", value="http://localhost:8000/v1")
    vllm_model_name = st.sidebar.text_input("vLLM Model Name", value="meta-llama/Llama-3.2-11B-Vision-Instruct")
else:
    vllm_base_url = None
    vllm_model_name = None

st.sidebar.markdown("---")
st.sidebar.markdown("### 📷 Select Source Image")

source_modes = ["Sample Validation Images", "Upload Custom File", "Browser Camera Snapshot", "Live Camera Feed"]
source_mode = st.sidebar.radio("Input Source Mode", source_modes)

# Clear captured image if mode changes
if "prev_source_mode" not in st.session_state:
    st.session_state["prev_source_mode"] = source_mode
if st.session_state["prev_source_mode"] != source_mode:
    st.session_state["prev_source_mode"] = source_mode
    if "captured_image" in st.session_state:
        del st.session_state["captured_image"]

# Populate validation samples
val_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset/images/val/")
if not os.path.exists(val_dir):
    val_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
sample_images = []
if os.path.exists(val_dir):
    sample_images = sorted([os.path.basename(f) for f in glob.glob(os.path.join(val_dir, "*.jpg"))])[:15]

selected_image_path = None
uploaded_image_bytes = None

if source_mode == "Sample Validation Images":
    if sample_images:
        selected_sample = st.sidebar.selectbox("Choose Validation Substrate", sample_images)
        selected_image_path = os.path.join(val_dir, selected_sample)
    else:
        st.sidebar.error("No validation images found in dataset folder.")
elif source_mode == "Upload Custom File":
    uploaded_file = st.sidebar.file_uploader("Upload Substrate Photo (.jpg/.png)", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        uploaded_image_bytes = uploaded_file.read()
elif source_mode == "Browser Camera Snapshot":
    camera_snap = st.sidebar.camera_input("Capture Substrate Image")
    if camera_snap is not None:
        uploaded_image_bytes = camera_snap.read()
elif source_mode == "Live Camera Feed":
    available_cams = get_available_cameras()
    if available_cams:
        cam_selection = st.sidebar.selectbox("Choose Camera Index", available_cams, format_func=lambda x: f"Webcam Device {x}")
    else:
        st.sidebar.warning("⚠️ **Webcam Device Offline**: No local video devices (`/dev/video*`) were detected on this server host. This feature requires physical camera connection.")
        cam_selection = st.sidebar.text_input("Custom Video Device/RTSP Stream URL", value="0")
        try:
            cam_selection = int(cam_selection)
        except ValueError:
            pass

    if "live_feed_checkbox" not in st.session_state:
        st.session_state["live_feed_checkbox"] = False

    live_active = st.sidebar.checkbox("Start Live Stream Feed", value=st.session_state["live_feed_checkbox"])
    st.session_state["live_feed_checkbox"] = live_active

    if st.sidebar.button("📸 Capture Frame & Process"):
        cap = cv2.VideoCapture(cam_selection)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()  # Release immediately to avoid resource locking before rerun!
            if ret:
                st.session_state["captured_image"] = frame
                st.session_state["live_feed_checkbox"] = False
                st.rerun()
            else:
                st.sidebar.error("Failed to capture frame from camera.")
        else:
            st.sidebar.error("Could not open camera to take snapshot.")

# ==========================================
# 🚀 Core Execution Pipeline
# ==========================================
model, err = load_yolo_model(model_format)
if err:
    st.error(f"Failed to load model: {err}")
    st.stop()

# Layout Columns
col1, col2 = st.columns([3, 2])

if source_mode == "Live Camera Feed" and live_active:
    with col1:
        st.markdown('<div class="glass-card"><h3>🎥 Live Camera View</h3></div>', unsafe_allow_html=True)
        image_placeholder = st.empty()
        fps_placeholder = st.empty()
        
        cap = cv2.VideoCapture(cam_selection)
        if not cap.isOpened():
            st.error(f"Error: Unable to access camera device {cam_selection}.")
        else:
            try:
                # Real-time inference loop
                while live_active:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Failed to fetch stream frame.")
                        break
                    
                    # Compute latency
                    t_start = time.perf_counter()
                    results = model(frame, verbose=False)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_end = time.perf_counter()
                    latency_ms = (t_end - t_start) * 1000
                    
                    annotated_frame = results[0].plot()
                    
                    # Convert to RGB for streamlit
                    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    image_placeholder.image(rgb_frame, channels="RGB", width="stretch")
                    
                    fps_placeholder.markdown(
                        f"⚡ **Inference Latency:** `{latency_ms:.2f} ms` | **Throughput:** `{1000/latency_ms:.1f} FPS`"
                    )
                    time.sleep(0.01)
            except Exception as ex:
                st.error(f"Error: {ex}")
            finally:
                cap.release()
                
    with col2:
        st.markdown('<div class="glass-card"><h3>⚙️ RAG Analysis Dashboard</h3><p>Freeze or stop camera feed to perform full interactive manual query and diagnostic report generation.</p></div>', unsafe_allow_html=True)

else:
    # Static Image Processing (Sample Uploaded, Browser Captured, or Live Captured)
    input_img = None
    if source_mode == "Sample Validation Images" and selected_image_path:
        input_img = cv2.imread(selected_image_path)
    elif source_mode == "Upload Custom File" and uploaded_image_bytes:
        file_bytes = np.frombuffer(uploaded_image_bytes, np.uint8)
        input_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif source_mode == "Browser Camera Snapshot" and uploaded_image_bytes:
        file_bytes = np.frombuffer(uploaded_image_bytes, np.uint8)
        input_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif source_mode == "Live Camera Feed" and "captured_image" in st.session_state:
        input_img = st.session_state["captured_image"]
        
    if input_img is not None:
        # Run detection
        t_start = time.perf_counter()
        results = model(input_img, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000
        
        # Render visual layout
        with col1:
            st.markdown('<div class="glass-card"><h3>🕵️ Inspection Substrate View</h3></div>', unsafe_allow_html=True)
            
            # Annotated vs Original view in tabs
            tab_annotated, tab_original = st.tabs(["Annotated View", "Original View"])
            with tab_annotated:
                annotated_img = results[0].plot()
                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), width="stretch")
            with tab_original:
                st.image(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB), width="stretch")
                
            # Render Speed Metrics
            st.markdown(f"""
                <div class="glass-card">
                    <h4>⚙️ Speed Metrics (Using {model_format})</h4>
                    <div style="display: flex; justify-content: space-around;">
                        <div>
                            <p style="margin-bottom: 0; color: #8A9Aad;">Avg Latency</p>
                            <span class="metric-value">{latency_ms:.2f} ms</span>
                        </div>
                        <div>
                            <p style="margin-bottom: 0; color: #8A9Aad;">Throughput</p>
                            <span class="metric-value">{1000/latency_ms:.1f} FPS</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="glass-card"><h3>📝 Detected Defects List</h3></div>', unsafe_allow_html=True)
            boxes = results[0].boxes
            if len(boxes) == 0:
                st.success("✅ Clean substrate! No anomalies detected.")
            else:
                defect_items = []
                for idx, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    defect_class = CLASS_NAMES.get(cls_id, f"Unknown ({cls_id})")
                    defect_items.append({
                        "label": f"Defect #{idx+1}: {defect_class} ({conf*100:.1f}%)",
                        "coords": xyxy,
                        "class": defect_class,
                        "confidence": conf
                    })
                    
                selected_defect_label = st.selectbox(
                    "Select defect to inspect & diagnose:",
                    options=[d["label"] for d in defect_items]
                )
                
                # Fetch selected defect
                selected_defect = [d for d in defect_items if d["label"] == selected_defect_label][0]
                coords = selected_defect["coords"]
                d_class = selected_defect["class"]
                d_conf = selected_defect["confidence"]
                
                # Determine confidence level
                if d_conf >= 0.8:
                    d_conf_level = "High"
                elif d_conf >= 0.5:
                    d_conf_level = "Medium"
                else:
                    d_conf_level = "Low"
                    
                # Compile other defects summary
                all_defects_summary = "\n".join([
                    f"- Defect #{i+1}: {d['class']} at coordinates {d['coords'].tolist()} with confidence {d['confidence']*100:.1f}%"
                    for i, d in enumerate(defect_items)
                ])
                
                # Crop defect region
                crop_img = input_img[coords[1]:coords[3], coords[0]:coords[2]]
                
                # Display crop side-by-side with coordinates
                st.markdown(f"**Bounding Coordinates (x1, y1, x2, y2):** `{coords.tolist()}`")
                
                crop_col1, crop_col2 = st.columns([1, 2])
                with crop_col1:
                    st.markdown("**Microscopic Zoom:**")
                    if crop_img.size > 0:
                        st.image(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB), width="stretch")
                with crop_col2:
                    # Query manual from ChromaDB
                    st.markdown("📖 **Matched AOI Service Manual:**")
                    manual_text, sources = query_manual(vector_db, d_class)
                    st.markdown(f'<div class="manual-box">{manual_text}</div>', unsafe_allow_html=True)
                    
                st.markdown("---")
                st.markdown("### 🧠 AI Field Service Report")
                
                if st.button("Generate Diagnostic Report", type="primary"):
                    with st.spinner("Analyzing defect microscopic patch & manuals..."):
                        if reasoning_provider == "Gemini 3.5 Flash (Online)":
                            if not api_key_input:
                                st.warning("Please input your Gemini API Key in the sidebar.")
                                report = None
                            else:
                                report = generate_gemini_report(
                                    api_key=api_key_input,
                                    defect_class=d_class,
                                    bbox_coords=coords.tolist(),
                                    confidence_pct=d_conf * 100,
                                    confidence_level=d_conf_level,
                                    all_defects_summary=all_defects_summary,
                                    retrieved_manuals=manual_text,
                                    crop_image_np=crop_img
                                )
                        elif reasoning_provider in ["Llama.cpp (Offline)", "Ollama (Offline)", "Local vLLM (Offline)"]:
                            b64_crop = get_base64_image(crop_img)
                            report = generate_local_vllm_report(
                                vllm_url=vllm_base_url,
                                model_name=vllm_model_name,
                                defect_class=d_class,
                                bbox_coords=coords.tolist(),
                                confidence_pct=d_conf * 100,
                                confidence_level=d_conf_level,
                                all_defects_summary=all_defects_summary,
                                retrieved_manuals=manual_text,
                                base64_crop=b64_crop
                            )
                        else:
                            report = None
                            
                        # If local fallback or error
                        if report is None:
                            st.info("No LLM provider available. Outputting template field service manual actions:")
                            sources_list = "\n".join([f"*   `{src['source']} (Section {src['section']})`" for src in sources])
                            report = f"""
### 📋 FIELD SERVICE REPORT
*   **Target Defect:** {d_class}
*   **Coordinate Box:** {coords.tolist()}
*   **Detection Confidence:** {d_conf*100:.1f}% (Level: {d_conf_level})
*   **All Co-located Defects:**
{all_defects_summary}

#### 🛠️ Actions Mandated:
{manual_text}

#### 📖 Sources Referenced:
{sources_list}
"""
                        
                        # Post-process report to add HTML anchor links to citations
                        report_linkified = linkify_citations(report)
                        st.markdown(f'<div class="report-box">{report_linkified}</div>', unsafe_allow_html=True)
                        
                        # Render matched references in interactive expanders
                        st.markdown("### 📚 Reference Sources Matched")
                        for item in sources:
                            anchor_id = f"{item['source']}_section_{item['section']}".replace(".", "_").replace(" ", "_").lower()
                            st.markdown(f'<div id="{anchor_id}"></div>', unsafe_allow_html=True)
                            with st.expander(f"📖 {item['source']} - Section {item['section']}"):
                                st.write(item['text'])
    else:
        st.info("👈 Please select a sample validation image, upload a custom file, or start the live camera feed & capture a frame in the sidebar.")

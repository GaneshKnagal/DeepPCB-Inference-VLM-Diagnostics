import os
# Load local .env file manually if it exists
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"').strip("'")

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from PIL import Image

# 1. Defect Category Names
CLASS_NAMES = ["Open", "Short", "Mousebite", "Spur", "Spurious Copper", "Pin-hole"]

def setup_vector_db():
    """
    Sets up a local ChromaDB instance, loads the generated technical manuals,
    and indexes them with sentence embeddings.
    """
    print("\n[RAG] Setting up local vector database...")
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(ROOT_DIR, "chroma_db")
    client = chromadb.PersistentClient(path=db_path)
    
    # Use a lightweight, local embedding function
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Get or create collection
    collection = client.get_or_create_collection(
        name="aoi_service_manuals",
        embedding_function=embedding_func
    )
    
    # Ingest manuals if collection is empty
    if collection.count() == 0:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        manuals_dir = os.path.join(ROOT_DIR, "mock_manuals")
        if not os.path.exists(manuals_dir):
            print(f"Error: Manuals directory {manuals_dir} not found. Please run create_mock_manuals.py first.")
            return None
            
        print("[RAG] Ingesting technical manuals into vector database...")
        doc_id = 0
        for filename in os.listdir(manuals_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(manuals_dir, filename)
                with open(filepath, 'r') as f:
                    content = f.read()
                    
                # Split manuals into sections for better retrieval granularity
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
        print(f"[RAG] Ingestion complete. Indexed {collection.count()} manual sections.")
    else:
        print(f"[RAG] Found existing database with {collection.count()} indexed sections.")
        
    return collection

def query_knowledge_base(collection, defect_class):
    """
    Queries the vector database for AOI service manual procedures
    related to the detected defect class.
    """
    if collection is None:
        return "No database available."
        
    print(f"[RAG] Querying manuals for defect: '{defect_class}'...")
    results = collection.query(
        query_texts=[defect_class],
        n_results=2
    )
    
    retrieved_context = "\n\n".join(results['documents'][0])
    return retrieved_context

def run_multimodal_reasoning(cropped_img_path, defect_class, bbox_coords, confidence_pct, confidence_level, all_defects_summary, retrieved_manuals):
    """
    Passes the cropped defect image and retrieved manual context to Gemini
    to perform a rigorous engineering root-cause diagnosis.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        print("\n[AI] GEMINI_API_KEY not found in environment variables.")
        print("[AI] Outputting local template-based engineer report instead...")
        # Graceful fallback report
        fallback_report = f"""
======================================================================
AOI TECHNICAL SERVICE REPORT (LOCAL FALLBACK)
======================================================================
[DEFECT CHARACTERIZATION]
- Class Identified: {defect_class}
- Coordinate Bounding Box: {bbox_coords}
- Detection Confidence: {confidence_pct:.1f}% (Level: {confidence_level})
- All Co-located Defects:
{all_defects_summary}

[RETIREVED SERVICE MANUAL DIRECTIVE]
{retrieved_manuals}

[RECOMMENDED ACTION PLAN]
Please configure 'GEMINI_API_KEY' in your shell to receive an AI-powered,
multimodal root-cause diagnosis. Based on the local service manual matched:
--> Refer to the corrective actions listed above for this class of defect.
======================================================================
"""
        return fallback_report
        
    print("\n[AI] Initializing Gemini Multimodal Reasoning Engine...")
    genai.configure(api_key=api_key)
    
    # Load cropped image
    img = Image.open(cropped_img_path)
    
    # Load Gemini 3.5 Flash (SOTA Multimodal model for speed & efficiency)
    model = genai.GenerativeModel('gemini-3.5-flash')
    
    prompt = f"""
You are an expert Senior AOI Field Service Engineer diagnostic system. 
You are inspecting a semiconductor/PCB trace anomaly and performing a root-cause reasoning analysis.

YOLO DETECTION TELEMETRY:
- Target Defect Class: {defect_class}
- Target Coordinates: {bbox_coords}
- Detection Confidence Score: {confidence_pct:.1f}%
- Detection Confidence Level: {confidence_level}

ALL DETECTED SUBSTRATE DEFECTS:
{all_defects_summary}

Below is the relevant retrieved context from our AOI Technical Service Manuals:
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
"""
    
    try:
        response = model.generate_content([prompt, img])
        return response.text
    except Exception as e:
        return f"Error during Gemini generation: {e}"

def diagnose_image(image_path):
    # 1. Load trained YOLOv8 model (use best weights)
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(ROOT_DIR, "runs/detect/pcb_inspection/yolov8s_deeppcb/weights/best.pt")
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}. Please run train.py first.")
        return
        
    print(f"\n[CV] Loading model from: {weights_path}")
    model = YOLO(weights_path)
    
    # 2. Run object detection
    print(f"[CV] Running defect detection on: {image_path}")
    image = cv2.imread(image_path)
    results = model(image, verbose=False)
    
    # Setup database
    collection = setup_vector_db()
    
    # Process detections
    boxes = results[0].boxes
    if len(boxes) == 0:
        print("[CV] No defects detected on this substrate. Yield is nominal!")
        return
        
    print(f"[CV] Detected {len(boxes)} defects.")
    
    # Create temp directory for crops
    os.makedirs("temp_crops", exist_ok=True)
    
    # Compile other defects summary
    all_defects_summary = "\n".join([
        f"- Defect #{i+1}: {CLASS_NAMES[int(b.cls[0])]} at coordinates {list(map(int, b.xyxy[0]))} with confidence {float(b.conf[0])*100:.1f}%"
        for i, b in enumerate(boxes)
    ])
    
    for idx, box in enumerate(boxes):
        # Extract class, conf, and bbox coordinates
        class_id = int(box.cls[0])
        defect_class = CLASS_NAMES[class_id]
        conf = float(box.conf[0])
        
        # YOLO coordinates (normalized/absolute depending on box.xyxy)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bbox_coords = [x1, y1, x2, y2]
        
        # Determine confidence level
        conf_level = "High" if conf >= 0.8 else "Medium" if conf >= 0.5 else "Low"
        
        print(f"\n--- Defect #{idx + 1}: {defect_class} ({conf*100:.1f}%) at {bbox_coords} ---")
        
        # Crop defect image
        crop = image[y1:y2, x1:x2]
        # Pad crop if it is too small (helps the VLM see it better)
        h, w, _ = crop.shape
        if h < 50 or w < 50:
            crop = cv2.resize(crop, (150, 150), interpolation=cv2.INTER_LINEAR)
            
        crop_path = f"temp_crops/defect_{idx}.jpg"
        cv2.imwrite(crop_path, crop)
        
        # 3. Retrieve relevant manuals (RAG)
        retrieved_manuals = query_knowledge_base(collection, defect_class)
        
        # 4. Perform Multimodal AI reasoning
        report = run_multimodal_reasoning(crop_path, defect_class, bbox_coords, conf * 100, conf_level, all_defects_summary, retrieved_manuals)
        print(report)

def main():
    # Test on a validation sample with defects
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    sample_val_image = os.path.join(ROOT_DIR, "samples/00041200.jpg")
    if not os.path.exists(sample_val_image):
        sample_val_image = os.path.join(ROOT_DIR, "dataset/images/val/00041200.jpg")
    diagnose_image(sample_val_image)

if __name__ == "__main__":
    main()

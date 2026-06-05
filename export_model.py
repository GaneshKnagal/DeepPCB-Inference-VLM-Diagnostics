import os
from ultralytics import YOLO

def export_trained_model():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(ROOT_DIR, "runs/detect/pcb_inspection/yolov8s_deeppcb/weights/best.pt")
    
    if not os.path.exists(model_path):
        print(f"Error: Trained weights not found at {model_path}. Please wait for training to complete.")
        return
        
    print(f"Loading trained weights from: {model_path}")
    model = YOLO(model_path)
    
    # Export to ONNX (with dynamic shapes and FP16 for edge versatility)
    print("\n[STEP 1] Exporting to ONNX (FP16 half-precision, dynamic shapes)...")
    onnx_path = model.export(format="onnx", half=True, dynamic=True, device=0)
    print(f"ONNX export successful: {onnx_path}")
    
    # Export to TensorRT Engine (compiled specifically for RTX 4050 CUDA cores)
    print("\n[STEP 2] Exporting to TensorRT Engine (FP16 optimized for local GPU)...")
    try:
        trt_path = model.export(format="engine", half=False, device=0)
        print(f"TensorRT export successful: {trt_path}")
    except Exception as e:
        print(f"Warning: TensorRT export failed with error: {e}")
        print("Note: Ensure TensorRT is correctly installed in the system PATH if this fails.")
        
    print("\nModel export operations completed!")

if __name__ == "__main__":
    export_trained_model()

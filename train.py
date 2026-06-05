import os
from ultralytics import YOLO

def train_model():
    print("Initializing YOLOv8s defect detection model...")
    # Load a pretrained YOLOv8s model (small variant, balanced for 6GB VRAM RTX 4050)
    model = YOLO("yolov8s.pt")
    
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    yaml_config_path = os.path.join(ROOT_DIR, "data_yaml.yaml")
    
    print("Starting training pipeline...")
    # Train the model with optimal configurations for resource constraints and accuracy
    results = model.train(
        data=yaml_config_path,
        epochs=50,                  # Sufficient epochs to converge on DeepPCB
        imgsz=640,                  # Maintain 640x640 resolution for tiny defects
        batch=16,                   # Safe batch size to fit within 6GB VRAM
        device=0,                   # Train on NVIDIA CUDA GPU (RTX 4050)
        workers=4,                  # Multi-threaded data loading
        amp=True,                   # Enable Automatic Mixed Precision (FP16) for VRAM conservation
        project="pcb_inspection",   # Save results in pcb_inspection project folder
        name="yolov8s_deeppcb",     # Run name
        exist_ok=True,              # Overwrite if exists
        val=True,                   # Validate after each epoch
        save=True                   # Save checkpoints and final weights
    )
    
    print("\nTraining completed successfully!")
    print(f"Results and model weights saved in: pcb_inspection/yolov8s_deeppcb")
    
if __name__ == "__main__":
    train_model()

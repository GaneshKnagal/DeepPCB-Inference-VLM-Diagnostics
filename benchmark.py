import os
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

def benchmark_inference():
    # Define model paths
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(ROOT_DIR, "runs/detect/pcb_inspection/yolov8s_deeppcb/weights")
    pt_path = os.path.join(weights_dir, "best.pt")
    onnx_path = os.path.join(weights_dir, "best.onnx")
    engine_path = os.path.join(weights_dir, "best.engine")
    
    # Load a sample image from validation set fallback
    sample_img_path = os.path.join(ROOT_DIR, "samples/00041200.jpg")
    if not os.path.exists(sample_img_path):
        sample_img_path = os.path.join(ROOT_DIR, "dataset/images/val/00041200.jpg")
    if not os.path.exists(sample_img_path):
        print(f"Error: Sample image not found.")
        return
        
    image = cv2.imread(sample_img_path)
    
    print("=" * 60)
    print("EDGE INFERENCE SPEED BENCHMARK (RTX 4050 GPU)")
    print("=" * 60)
    
    frameworks = {}
    
    # 1. PyTorch Baseline
    if os.path.exists(pt_path):
        print("\nLoading PyTorch baseline model...")
        model_pt = YOLO(pt_path)
        # Warmup
        model_pt(image, verbose=False)
        frameworks["PyTorch (FP32/AMP)"] = model_pt
        
    # 2. ONNX FP16
    if os.path.exists(onnx_path):
        print("Loading ONNX runtime optimized model...")
        model_onnx = YOLO(onnx_path)
        # Warmup
        model_onnx(image, verbose=False)
        frameworks["ONNX Runtime (FP16)"] = model_onnx
        
    # 3. TensorRT FP16 Engine
    if os.path.exists(engine_path):
        print("Loading TensorRT CUDA Engine...")
        model_engine = YOLO(engine_path)
        # Warmup
        model_engine(image, verbose=False)
        frameworks["TensorRT Engine (FP16)"] = model_engine
        
    if not frameworks:
        print("Error: No models found to benchmark. Please run export_model.py first.")
        return
        
    # Benchmark execution
    num_runs = 100
    
    for name, model in frameworks.items():
        print(f"\nRunning {num_runs} iterations for {name}...")
        
        # Sync CUDA before timer starts
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        start_time = time.perf_counter()
        
        for _ in range(num_runs):
            # Run inference (verbose=False to avoid print overhead during benchmark)
            _ = model(image, verbose=False)
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()
        
        total_time = (end_time - start_time) * 1000.0  # in ms
        avg_latency = total_time / num_runs
        fps = 1000.0 / avg_latency
        
        print(f"|-- Average Latency: {avg_latency:.2f} ms")
        print(f"|-- Throughput:      {fps:.1f} FPS")
        
    print("\nBenchmark completed!")
    print("=" * 60)

if __name__ == "__main__":
    benchmark_inference()

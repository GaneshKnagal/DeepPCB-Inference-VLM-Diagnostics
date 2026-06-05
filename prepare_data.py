import os
import shutil
from tqdm import tqdm

def convert_to_yolo(x1, y1, x2, y2, img_w=640, img_h=640):
    # Calculate width and height of bbox
    w = x2 - x1
    h = y2 - y1
    
    # Calculate center coordinates
    x_center = x1 + w / 2.0
    y_center = y1 + h / 2.0
    
    # Normalize coordinates
    x_center /= img_w
    y_center /= img_h
    w /= img_w
    h /= img_h
    
    return x_center, y_center, w, h

def process_split(split_file, output_images_dir, output_labels_dir, raw_pcb_data_dir):
    with open(split_file, 'r') as f:
        lines = f.readlines()
        
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    success_count = 0
    
    for line in tqdm(lines, desc=f"Processing {os.path.basename(split_file)}"):
        parts = line.strip().split()
        if len(parts) != 2:
            continue
            
        rel_img_path, rel_lbl_path = parts
        
        # Resolve actual filenames (DeepPCB uses _test.jpg in folders but .jpg in the text split file)
        actual_img_name = os.path.basename(rel_img_path).replace('.jpg', '_test.jpg')
        actual_img_dir = os.path.join(raw_pcb_data_dir, os.path.dirname(rel_img_path))
        src_img_path = os.path.join(actual_img_dir, actual_img_name)
        
        src_lbl_path = os.path.join(raw_pcb_data_dir, rel_lbl_path)
        
        # Verify source files exist
        if not os.path.exists(src_img_path):
            print(f"Warning: Image file not found: {src_img_path}")
            continue
        if not os.path.exists(src_lbl_path):
            print(f"Warning: Label file not found: {src_lbl_path}")
            continue
            
        base_name = os.path.splitext(os.path.basename(rel_img_path))[0]
        dest_img_path = os.path.join(output_images_dir, f"{base_name}.jpg")
        dest_lbl_path = os.path.join(output_labels_dir, f"{base_name}.txt")
        
        # Copy image file
        shutil.copy(src_img_path, dest_img_path)
        
        # Parse and convert labels
        yolo_lines = []
        with open(src_lbl_path, 'r') as lf:
            for lbl_line in lf:
                lbl_parts = lbl_line.strip().split()
                if len(lbl_parts) != 5:
                    continue
                
                # DeepPCB: x1 y1 x2 y2 type
                try:
                    x1, y1, x2, y2, defect_type = map(int, lbl_parts)
                except ValueError:
                    continue
                
                # DeepPCB defect type is 1-indexed (1-6). Convert to 0-indexed class for YOLO (0-5).
                class_id = defect_type - 1
                
                # Convert to normalized YOLO coordinates
                x_center, y_center, w, h = convert_to_yolo(x1, y1, x2, y2)
                
                # Check for boundary issues
                x_center = min(max(x_center, 0.0), 1.0)
                y_center = min(max(y_center, 0.0), 1.0)
                w = min(max(w, 0.0), 1.0)
                h = min(max(h, 0.0), 1.0)
                
                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                
        with open(dest_lbl_path, 'w') as df:
            df.writelines(yolo_lines)
            
        success_count += 1
        
    print(f"Successfully processed {success_count}/{len(lines)} files for split.")

def main():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    raw_pcb_data_dir = os.path.join(ROOT_DIR, "DeepPCB_raw/PCBData")
    train_split_file = os.path.join(raw_pcb_data_dir, "trainval.txt")
    test_split_file = os.path.join(raw_pcb_data_dir, "test.txt")
    
    output_base_dir = os.path.join(ROOT_DIR, "dataset")
    
    print("Starting DeepPCB data preparation for YOLOv8/v10...")
    
    # Process Train Split
    process_split(
        split_file=train_split_file,
        output_images_dir=os.path.join(output_base_dir, "images/train"),
        output_labels_dir=os.path.join(output_base_dir, "labels/train"),
        raw_pcb_data_dir=raw_pcb_data_dir
    )
    
    # Process Val/Test Split
    process_split(
        split_file=test_split_file,
        output_images_dir=os.path.join(output_base_dir, "images/val"),
        output_labels_dir=os.path.join(output_base_dir, "labels/val"),
        raw_pcb_data_dir=raw_pcb_data_dir
    )
    
    print("\nData preparation complete!")
    print(f"Processed dataset saved to: {output_base_dir}")

if __name__ == "__main__":
    main()

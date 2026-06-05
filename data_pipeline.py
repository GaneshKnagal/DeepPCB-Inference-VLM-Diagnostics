import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

class IndustrialPCBDataset(Dataset):
    """
    Custom PyTorch Dataset for Industrial PCB Defect Detection.
    Integrates Albumentations for robust, real-world augmentation pipelines.
    """
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Load image (convert to RGB)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resolve label path
        base_name = os.path.splitext(img_name)[0]
        lbl_path = os.path.join(self.labels_dir, f"{base_name}.txt")
        
        bboxes = []
        class_labels = []
        
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        # YOLO format: class_id, x_center, y_center, w, h
                        x_c, y_c, w, h = map(float, parts[1:])
                        
                        # Albumentations expects [x_center, y_center, width, height] format for yolo
                        bboxes.append([x_c, y_c, w, h])
                        class_labels.append(class_id)
                        
        # Apply transforms
        if self.transform:
            augmented = self.transform(
                image=image,
                bboxes=bboxes,
                category_ids=class_labels
            )
            image = augmented['image']
            bboxes = augmented['bboxes']
            class_labels = augmented['category_ids']
            
        return image, bboxes, class_labels

def get_train_transforms(img_size=640):
    """
    Returns robust Albumentations pipeline simulating factory/microscope field conditions:
    - Variable lighting / contrast (CLAHE, BrightnessContrast)
    - Sensor / camera noise (GaussNoise)
    - Camera vibrations (Blur, MotionBlur)
    - Handler shift/rotation (ShiftScaleRotate)
    """
    return A.Compose([
        # Spatial Augmentations (PCB rotation & shifts)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        
        # Real-World Factory Lighting / Sensor Conditions
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),  # Dynamic contrast adjustment
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        
        # Microscope Blur / Vibrations
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.Blur(blur_limit=5, p=1.0),
        ], p=0.4),
        
        # High ISO Sensor Noise
        A.GaussNoise(p=0.4),
        
        # Resize and Normalization (ready for PyTorch)
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'], min_area=4, min_visibility=0.3))

def get_val_transforms(img_size=640):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

def visualize_sample(image, bboxes, category_ids, class_names, output_path="sample.jpg"):
    """
    Saves a visualization of the augmented image with its bounding boxes drawn.
    Helps verify annotation mapping.
    """
    # Denormalize image for drawing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = image.permute(1, 2, 0).cpu().numpy()
    img = (img * std + mean) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    h, w, _ = img.shape
    
    # Draw boxes
    for bbox, cat_id in zip(bboxes, category_ids):
        # Convert from normalized YOLO (x_c, y_c, bw, bh) to absolute Pascal VOC (x1, y1, x2, y2)
        x_c, y_c, bw, bh = bbox
        x1 = int((x_c - bw / 2.0) * w)
        y1 = int((y_c - bh / 2.0) * h)
        x2 = int((x_c + bw / 2.0) * w)
        y2 = int((y_c + bh / 2.0) * h)
        
        color = (255, 0, 0)  # Red box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        label_text = f"{class_names[int(cat_id)]}"
        cv2.putText(img, label_text, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Visualization saved to: {output_path}")

if __name__ == "__main__":
    # Test script locally
    CLASS_NAMES = ["Open", "Short", "Mousebite", "Spur", "Spurious Copper", "Pin-hole"]
    
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    train_dataset = IndustrialPCBDataset(
        images_dir=os.path.join(ROOT_DIR, "dataset/images/train"),
        labels_dir=os.path.join(ROOT_DIR, "dataset/labels/train"),
        transform=get_train_transforms()
    )
    
    print(f"Total training images: {len(train_dataset)}")
    
    # Test loading and augment a sample image
    image, bboxes, cat_ids = train_dataset[0]
    print(f"Sample 0 - Image shape: {image.shape}, Number of bboxes: {len(bboxes)}")
    
    visualize_sample(image, bboxes, cat_ids, CLASS_NAMES, "augmented_sample.jpg")

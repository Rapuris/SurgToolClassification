from typing import Any, Dict, Optional, Tuple
import random
from pathlib import Path
import logging
import regex as re
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from rich.progress import track
import numpy as np
import json
from .. import utils
from typing import Any, Optional, List, Tuple
from ..augmentations import spatial_augmentation, spatial_augmentation_with_heatmap, intensity_augmentation
from PIL import Image
import cv2

log = logging.getLogger(__name__)

def crop_image_to_bbox(image: np.ndarray, label_path: Path) -> np.ndarray:
    """
    Crops the image to the region of the first bounding box in the YOLO label file.
    The label file should be in YOLO format: class x_center y_center width height (all normalized).
    Returns the cropped image. If no bbox is found, returns the original image.
    """
    #if not label_path.exists():
    #    return image
    try:
        with open(label_path, "r") as f:
            line = f.readline()
            if not line:
                return image
            parts = line.strip().split()
            if len(parts) < 5:
                return image
            # YOLO: class x_center y_center width height (all normalized)
            _, x_c, y_c, w, h = map(float, parts[:5])
            #print(f"x_c: {x_c}, y_c: {y_c}, w: {w}, h: {h}")
            H, W = image.shape[:2]
            x_c, y_c, w, h = x_c * W, y_c * H, w * W, h * H
            x1 = int(max(0, x_c - w / 2))
            y1 = int(max(0, y_c - h / 2))
            x2 = int(min(W, x_c + w / 2))
            y2 = int(min(H, y_c + h / 2))
            if x2 > x1 and y2 > y1:
                return image[y1:y2, x1:x2]
            else:
                return image
    except Exception as e:
        log.warning(f"Failed to crop image to bbox for {label_path}: {e}")
        #assert False, f"Failed to crop image to bbox for {label_path}: {e}"
        return image

def center_plus_jitter_heatmap(height, width, center=None, jitter_std=0.1):
    """
    Generate a Gaussian heatmap centered at the specified point.
    
    Args:
        height: Height of the heatmap
        width: Width of the heatmap  
        center: (x, y) center point. If None, uses image center
        jitter_std: Standard deviation for jittering the center
    """
    if center is None:
        center_x, center_y = width // 2, height // 2
    else:
        center_x, center_y = center
    
    # Add jitter to the center
    jitter_x = np.random.normal(0, jitter_std * width)
    jitter_y = np.random.normal(0, jitter_std * height)
    
    center_x += jitter_x
    center_y += jitter_y
    
    # Create coordinate grids
    y, x = np.ogrid[:height, :width]
    
    # Calculate distances from center
    sigma = min(height, width) * 0.1  # Adjust sigma based on image size
    heatmap = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
    
    return heatmap

class ClassificationDataset(Dataset):
    def __init__(self, items: List[Tuple[Path, int]], train=True, image_size=224):
        self.items = items
        self.train = train
        self.image_size = image_size
        # Use the imported augmentations
        self.spatial_transform = spatial_augmentation_with_heatmap()
        
        #spatial_augmentation(
        #    train=self.train,
        #    last_test=False,
        #    test=not self.train,
        #    annotations=False,
        #    image_size=(self.image_size, self.image_size, 3),
        #    normalize=False,
        #    do_neglog=False,
        #    three_channel=True
        #)
        self.intensity_transform = intensity_augmentation(
            train=self.train,
            last_test=True,
            test=not self.train,
            annotations=False,
            image_size=None,
            normalize=True,
            do_neglog=False,
            three_channel=False
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        image_path, class_idx = self.items[idx]

        # Load and preprocess image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label_path = Path(str(image_path).replace('/images/', '/labels/')).with_suffix('.txt')
        #print(f"Label path: {label_path}")
        image = crop_image_to_bbox(image, label_path)
        
        orig_image = image.copy()
        
        #cv2.imwrite("orig_image.png", image)

        # Get the original center point
        Hh, Hw, _ = image.shape
        original_center = (Hw // 2, Hh // 2)  # (x, y) format for keypoints

        # Apply spatial transformation with center point as keypoint
        transformed = self.spatial_transform(
            image=image.copy(), 
            bboxes=[], 
            category_ids=[], 
            masks=[], 
            keypoints=[original_center]  # Pass center as keypoint
        )
        image_spatial = transformed["image"]
        transformed_center = transformed["keypoints"][0]  # Get transformed center point
        assert transformed_center != original_center, f"Transformed center: {transformed_center}, Original center: {original_center}"

        # Get the dimensions of the transformed image
        transformed_h, transformed_w = image_spatial.shape[:2]

        # Generate heatmap at the new transformed center location with correct dimensions
        heatmap = center_plus_jitter_heatmap(transformed_h, transformed_w, center=transformed_center).astype(np.float32)

        # Apply intensity augmentation (only to RGB image, not heatmap)
        intensity_transformed = self.intensity_transform(
            image=image_spatial.copy(), 
            bboxes=[], 
            category_ids=[], 
            masks=[], 
            keypoints=[]
        )
        input_image = intensity_transformed["image"]

        # Resize heatmap to match the final image size after intensity transformation
        final_height, final_width = input_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (final_width, final_height))

        # Convert to channel-first
        input_image = input_image.transpose(2, 0, 1).astype(np.float32)  # (3, 224, 224)
        heatmap_resized = heatmap_resized[np.newaxis, :, :].astype(np.float32)  # Add channel dimension: (1, 224, 224)
        heatmap_input_image = np.concatenate([input_image, heatmap_resized], axis=0).astype(np.float32)  # (4, 224, 224)
       
        orig_image = cv2.resize(orig_image, (224, 224))  # Resize to target size
        orig_image = orig_image.transpose(2, 0, 1).astype(np.float32)  # Convert to (C, H, W)
        assert orig_image.shape == (3,224, 224), f"Orig image shape: {orig_image.shape}"
        assert heatmap_input_image.shape == (4, 224, 224)   
        return orig_image, heatmap_input_image, class_idx


        
def parse_yolo_classification(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, int]]:
    items = []
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            log.warning(f"Label file {label_path} not found for image {img_path}, skipping.")
            continue
        with open(label_path, "r") as f:
            class_idx = int(f.readline().split()[0])
        items.append((img_path, class_idx))
    return items

def split_items(items: List[Tuple[Path, int]], val_ratio=0.1, seed=42):
    random.seed(seed)
    random.shuffle(items)
    n = len(items)
    n_val = int(val_ratio * n)
    val_items = items[:n_val]
    train_items = items[n_val:]
    return train_items, val_items

class ToolDataModule(LightningDataModule):
    def __init__(
        self,
        data_dirs: list = [
            "/mnt/data2/Sampath/RoboFlowDatasets/Sampath_Annotations/YOLO_finetune"
        ],
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        image_size: int = 224,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_dirs = [Path(d).expanduser() for d in data_dirs]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.image_size = image_size
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        pass

    def _gather_split_items(self, split: str) -> list:
        """
        Gather and combine all (img, class_idx) items for a split from all data_dirs.
        """
        items = []
        for data_dir in self.data_dirs:
            images_dir = data_dir / split / "images"
            labels_dir = data_dir / split / "labels"
            if images_dir.exists() and labels_dir.exists():
                items.extend(parse_yolo_classification(images_dir, labels_dir))
        return items

    def setup(self, stage: Optional[str] = None):
        # Gather and combine all train/val/test items from all directories
        all_train_items = self._gather_split_items("train")
        train_items, real_val_items = split_items(all_train_items, val_ratio=0.1)
        val_items, test_items = split_items(real_val_items, val_ratio=0.25)
        
        #train_items = train_items[:10]
        #val_items = val_items[:5]
        #test_items = test_items[:5]
        log.error(f"Train items: {len(train_items)}, Val items: {len(real_val_items)}, Test items: {len(test_items)}")
    
        if stage is None or stage == "fit":
            self.data_train = ClassificationDataset(train_items, train=True, image_size=self.image_size)
            self.data_val = ClassificationDataset(real_val_items, train=False, image_size=self.image_size)
        if stage is None or stage == "test":
            self.data_test = ClassificationDataset(test_items, train=False, image_size=self.image_size)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        pass

if __name__ == "__main__":
    _ = ToolDataModule()

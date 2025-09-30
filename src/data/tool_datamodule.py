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
from ..augmentations import spatial_augmentation, intensity_augmentation
from PIL import Image
import cv2

log = logging.getLogger(__name__)

def crop_image_to_bbox(image: np.ndarray, label_path: Path) -> np.ndarray:
    """
    Crops the image to the region of the first bounding box in the YOLO label file.
    The label file should be in YOLO format: class x_center y_center width height (all normalized).
    Returns the cropped image. If no bbox is found, returns the original image.
    """
    if not label_path.exists():
        return image
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
        return image

def center_plus_jitter_heatmap(
    h, w,
    center_sigma_frac=0.12,      # width of main lobe ≈ 12% of min(h,w)
    center_jitter_frac=0.02,     # how far the center can wander (as frac of min(h,w))
    n_jitter=10,                 # number of tiny blobs across the image
    jitter_sigma_px_range=(1,4), # size of tiny blobs (in px)
    jitter_amp=0.10,             # amplitude of jitter blobs vs center (0..1)
    seed=None
):
    """
    Returns float32 heatmap in [0,1] shaped (h,w).
    Big central Gaussian (+ small random offset) + many weak tiny Gaussians anywhere.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    yy = yy.astype(np.float32); xx = xx.astype(np.float32)
    m = float(min(h, w))

    # ---- central lobe (with small offset) ----
    cx = w/2 + rng.normal(0, center_jitter_frac*m)
    cy = h/2 + rng.normal(0, center_jitter_frac*m)
    sigma_c = center_sigma_frac * m
    H = np.exp(-(((xx-cx)**2 + (yy-cy)**2) / (2*sigma_c**2))).astype(np.float32)  # peak ≈1

    # ---- global jitter blobs ----
    for _ in range(n_jitter):
        jx = rng.uniform(0, w)
        jy = rng.uniform(0, h)
        sj = rng.uniform(*jitter_sigma_px_range)
        blob = np.exp(-(((xx-jx)**2 + (yy-jy)**2) / (2*sj*sj)))
        H += jitter_amp * blob.astype(np.float32)

    # ---- normalize to [0,1] ----
    H -= H.min()
    mx = H.max()
    if mx > 0: H /= mx
    return H

class ClassificationDataset(Dataset):
    def __init__(self, items: List[Tuple[Path, int]], train=True, image_size=224):
        self.items = items
        self.train = train
        self.image_size = image_size
        # Use the imported augmentations
        self.spatial_transform = spatial_augmentation(
            train=self.train,
            last_test=False,
            test=not self.train,
            annotations=False,
            image_size=(self.image_size, self.image_size, 3),
            normalize=False,
            do_neglog=False,
            three_channel=True
        )
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

    def __getitem__(self, idx):
        img_path, class_idx = self.items[idx]
        label_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
        image = np.array(Image.open(img_path))#.convert("RGB"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #log.error(image.shape)
        # Crop to bbox if possible
        image = crop_image_to_bbox(image, label_path)
        image = cv2.resize(image, (self.image_size, self.image_size))
        orig_image = image.copy()
        
        cv2.imwrite("orig_image.png", orig_image)
        #log.error(orig_image.mean())
        #log.error(f"Before spatial transform: orig_image.shape={orig_image.shape}")
        #log.error(image.shape)
        # Apply spatial augmentation
        transformed = self.spatial_transform(image=image.copy(), bboxes=[], category_ids=[], masks=[], keypoints=[])
        image_spatial = transformed["image"]
        # Apply intensity augmentation
        intensity_transformed = self.intensity_transform(image=image_spatial.copy(), bboxes=[], category_ids=[], masks=[], keypoints=[])
        input_image = intensity_transformed["image"]
        # Convert to channel-first
        #print(input_image.shape)
        Hh, Hw, _ = input_image.shape
        heatmap = center_plus_jitter_heatmap(Hh, Hw).astype(np.float32)
        #print('heatmap.shape', heatmap.shape)
        #print('input_image.shape', input_image.shape)
        input_image = input_image.transpose(2, 0, 1).astype(np.float32)  # (3, 224, 224)
        heatmap = heatmap[np.newaxis, :, :].astype(np.float32)  # Add channel dimension: (1, 224, 224)
        heatmap_input_image = np.concatenate([input_image, heatmap], axis=0).astype(np.float32)  # (4, 224, 224)
        orig_image = orig_image.transpose(2, 0, 1).astype(np.float32)
        #log.error(f"Image mean: {orig_image.mean()}")
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

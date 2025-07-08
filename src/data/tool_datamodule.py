import time
from typing import Any, Dict, Optional, Tuple
import random
from pathlib import Path
import logging
import regex as re
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from rich.progress import track
import matplotlib.pyplot as plt
import numpy as np
from perphix import utils as perphix_utils
import json
from .. import utils
from .components import PelphixAEDataset
from typing import Any, Optional, List, Tuple
from ..augmentations import spatial_augmentation, intensity_augmentation, prephix_intensity_augmentation
from albumentations.augmentations.crops.functional import crop
import albumentations as A
import cv2
from PIL import Image
from skimage.exposure import match_histograms

log = logging.getLogger(__name__)


class ToolDataset(Dataset):
    """
    Dataset that loads a PNG image and its corresponding JSON file containing landmarks.
    The JSON is assumed to contain landmark entries at the top level (e.g., "r_sps": [749, 1304], etc.).
    We extract the landmarks using the sorted order of the keys, apply augmentations,
    and then generate a heatmap for each keypoint. If a keypoint is out of bounds, its heatmap is all zeros.
    """
    def __init__(
        self,
        items: List[Tuple[dict, Path]],
        train: bool = True,
        image_size: List[int] = [224, 224],
        fliph: bool = False,
        test: bool = False,
    ):
        """
        Args:
            items: List of tuples (json_data, image_path). The json_data should contain landmark entries.
            train: Whether the dataset is used for training (affecting augmentation).
            image_size: Desired output size [width, height] for spatial augmentation.
            fliph: Whether to horizontally flip the images.
        """
        self.test = test
        self.items = items
        self.train = train
        self.image_size = list(image_size)
        self.fliph = fliph


    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[dict, dict]:
        # Load the JSON and image and seg path
        annotation, image_path, seg_file = self.items[index]

        image = np.array(Image.open(str(image_path)))
        if image.shape != (224, 224):
            image = image[:, :, 0]
        if image is None:
            raise FileNotFoundError(f"Image {image_path} not found.")

        spatial_transform = spatial_augmentation(
            train=self.train,
            last_test = False,
            test = self.test,
            annotations=True,
            image_size= (224, 224, 3),
            normalize=False, 
            do_neglog = True,
            three_channel= True
        )
    
        intensity_transform = prephix_intensity_augmentation(
            train=self.train,
            last_test = True,
            test = self.test,
            annotations=True,
            image_size=None,
            normalize=True,
            do_neglog = False,
            three_channel = False
        )
        bboxes: List = []
        category_ids: List = []
        masks: List = []
        keypoints: List = []
        transformed = spatial_transform(
            image=image.copy(),
            bboxes=bboxes,
            category_ids=category_ids,
            masks=masks,
            keypoints=keypoints, 
        )
     
        image_spatial = transformed["image"]
    
        H, W = image_spatial.shape[:2]
        target_image = image_spatial

        # Apply intensity augmentation.
        intensity_transformed = intensity_transform(
            image=image_spatial.copy(),
            bboxes=bboxes,
            category_ids=category_ids,
            masks=masks,
            keypoints=keypoints,
        )
        input_image = intensity_transformed["image"]
        if self.fliph:
            input_image = input_image[:, ::-1]
            target_image = target_image[:, ::-1]

        # Convert images to channel-first format.
        input_image = input_image.transpose(2, 0, 1).astype(np.float32)
        assert np.isnan(input_image).sum() == 0, np.isnan(input_image).sum()
        assert np.isinf(input_image).sum() == 0, np.isinf(input_image).sum()
        target_image = target_image.transpose(2, 0, 1).astype(np.float32)
        # Build the outputs.
        inputs = {
            "image": input_image,
        }
        targets = {
            "image": target_image
        }
        return inputs, targets

class ToolDataModule(LightningDataModule):
    """Lighning DataModule for the Surgical Tool Classification dataset.

    This dataset only contains train and validation sets.

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dirs: List[str] = ["/mnt/data/Sampath/tool_dataset_1", "/mnt/data/Sampath/tool_dataset_2"], 
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        image_size: list[int] = [224, 224], #[512, 512], #[256, 256]
        fliph: bool = False,
        image_channels: int = 3,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.image_size = [224, 224]
        self.fliph = fliph
        self.data_dir = Path(data_dir).expanduser()
        log.info(f"Data directory: {self.data_dir}")
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        # Downloads to data/annotations/{name}.json and data/{name} image dir.
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Scans data_dir for JSON files and expects a matching PNG image (with the same basename)
        for each JSON file. Splits the items into train (80%), validation (10%), and test (10%).
        """
        def _load_items(split: str) -> List[Tuple[dict, Path]]:
            image_files = sorted(list(self.data_dir.rglob("*.json")))
            train_json_files = [f for f in json_files if self.exclude_pat not in f.parts]
            #test_json_files = list(set(json_files).difference(train_json_files))
            log.error('train_json_files: ' + str(len(train_json_files)))
            assert(len([f for f in json_files if self.exclude_pat in f.parts]) > 0)
            log.error(len([f for f in json_files if self.exclude_pat in f.parts]))
            test_json_files = sorted(list(self.rob_dir.rglob("*.json")))
            random.shuffle(train_json_files)
            n = len(train_json_files)
            if n == 0:
                raise FileNotFoundError(f"No JSON files found in {self.data_dir}")
            if split == "train":
                files = train_json_files#[: int(0.9 * n)]
            elif split == "val":
                files = train_json_files[int(0.9 * n):]
            elif split == "test":
                files = test_json_files #train_json_files[int(0.9 * n):] # change out later
            else:
                raise ValueError(f"Invalid split: {split}")

            items: List[Tuple[dict, Path]] = []
            for json_file in files:
                #log.info(json_file)
                with open(json_file, "r") as f:
                    json_data = json.load(f)
                    #log.error(json_data)
                #DO ONLY IF ROB:
                json_data = {real_name: json_data.get(num_name) for num_name, real_name in ROB_mapping.items()}
                #assert(len(json_data.values()) == 12), f"bad json data {json_data.values()}"
                #for all
                json_data = {i:v for i,v in json_data.items() if i not in keys_to_remove}
                json_data = {keypoint: json_data[keypoint] for keypoint in SET_ORDER}
                #log.error(json_data)
                name = json_file.stem.replace("_landmarks", "")
                num = name.replace("DRR_", "")
                img_file = json_file.parent / (name + json_file.suffix)

                image_file = img_file.with_suffix(".jpg")
               
                if not image_file.exists():
                    raise FileNotFoundError(f"Image file {image_file} not found for {json_file}")
                if not seg_file.exists():
                    raise FileNotFoundError(f"Seg file {seg_file} not found for {json_file}")
                items.append((json_data, image_file, seg_file))
            #log.info(items[0])
            log.info(f"Loaded {len(items)} items for {split} split.")
            return items

        if stage is None or stage == "fit":
            train_items = _load_items("train")
            val_items = _load_items("val")
            self.data_train = LandmarkDataset(
                train_items, train=True, test= False, image_size=self.image_size, fliph=self.fliph
            )
            self.data_val = LandmarkDataset(
                val_items, train=False, test= False, image_size=self.image_size, fliph=self.fliph
            )
        if stage is None or stage == "test":
            test_items = _load_items("test")
            self.data_test = LandmarkDataset(
                test_items, train=False, test= True, image_size=self.image_size, fliph=self.fliph
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = ToolDataModule()

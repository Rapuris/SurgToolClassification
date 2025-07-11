# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import os
import numpy as np
from PIL import Image
import cv2
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from ..utils import pylogger
from pathlib import Path


log = pylogger.get_pylogger(__name__)

def fix_plotting(image: np.ndarray) -> np.ndarray:
    # Handle 4D arrays (batch, height, width, channel)
    if len(image.shape) == 4:
        image = image.squeeze(0)  # Remove batch dimension
    
    # Handle 3D arrays (height, width, channel) - already in HWC format
    if len(image.shape) == 3:
        if image.shape[2] == 1:  # Single channel (H, W, 1)
            image = np.concatenate([image, image, image], axis=2)  # Convert to RGB
        elif image.shape[2] == 3:  # Already RGB (H, W, 3)
            pass  # No change needed
        else:
            # Take the first channel and duplicate
            image = np.concatenate([image[:, :, :1], image[:, :, :1], image[:, :, :1]], axis=2)
    
    return image 

class ImageLogger(Callback):
    def __init__(
        self,
        batch_frequency,
        epoch_frequency,
        max_images,
        clamp=True,
        increase_log_steps=True,
        log_train: bool = True,
        log_val: bool = True,
    ):
        super().__init__()
        self.batch_freq = batch_frequency
        self.epoch_freq = epoch_frequency
        self.max_images = max_images
        self.log_steps = [2**n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

        self.log_train = log_train
        self.log_val = log_val

    @rank_zero_only
    def log_local(
        self,
        save_dir,
        split,
        images: dict[str, np.ndarray],
        global_step,
        current_epoch,
        batch_idx,
    ):
        root = os.path.join(save_dir, "images", split)

        def convert(img: np.ndarray | torch.Tensor) -> np.ndarray:
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()

            img = img.transpose(1, 2, 0).copy()
            img = (img + 1.0) * 127.5  # std + mean
            img = np.clip(img, 0, 255)
            return img.astype(np.uint8)

        def to_numpy(img: torch.Tensor | np.ndarray) -> np.ndarray:
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            return img

        input_images = images["input_images"]
        for k in range(input_images.shape[0]):
            input_image = input_images[k] #convert(input_images[k])
            #log.error(f"input image mean: {input_image.mean()}")
            #input_image = fix_plotting(input_image)
            #log.error(f"input image mean after fix: {input_image.mean()}")
            if isinstance(input_image, torch.Tensor):
                image = input_image.detach().cpu().numpy().copy()
            image = image.transpose(1, 2, 0).astype(np.uint8)  # Convert from (C, H, W) to (H, W, C)
            #assert image.shape[2] == 3, f"Image shape: {image.shape}"
            #log.error(f"image shape: {image.shape}")
            if split == "train" and "target_images" in images:
                #log.error("in train")
                target_image = convert(images["target_images"][k])
                #log.error(f"target image shape: {target_image.shape}")
                #print("target image", target_image.min(), target_image.max())
                target_image = fix_plotting(target_image)
                #log.error(f"target image shape: {target_image.shape}")
                #log.error(f"image shape: {image.shape}")
                #if input_image.dtype != np.uint8 and target_image.dtype != np.uint8:
                #    input_image = input_image.astype(np.uint8)
                #    target_image = target_image.astype(np.uint8)
                # Normalize both images to 0-255 range for consistent stacking
                #image_norm = ((input_image - input_image.min()) / (input_image.max() - input_image.min())).astype(np.uint8)
                #target_norm = ((target_image - target_image.min()) / (target_image.max() - target_image.min()) * 255).astype(np.uint8)
                if isinstance(input_image, torch.Tensor):
                    input_image = input_image.detach().cpu().numpy().copy()
                input_image = input_image.transpose(1, 2, 0).astype(np.uint8)
                
                image = np.concatenate([input_image, target_image], axis=0)

            filename = "e-{:06}_b-{:06}_n-{:06}.png".format(current_epoch, batch_idx, k)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            # Ensure image is uint8 before saving
            #if image.dtype != np.uint8:
            #    image = image.astype(np.uint8)
            cv2.imwrite(path, image)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (
            self.check_frequency(batch_idx)
            and pl_module.current_epoch % self.epoch_freq == 0
            and hasattr(pl_module, "log_images")
            and callable(pl_module.log_images)
            and self.max_images > 0
        ):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()
            with torch.no_grad():
                images: dict = pl_module.log_images(batch, split=split)

            for k, v in images.items():
                if isinstance(v, torch.Tensor):
                    images[k] = v.detach().cpu()[: self.max_images]

            self.log_local(
                pl_module.logger.save_dir,
                split,
                images,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx,
            )

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0
    ):
        if self.log_train:
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0
    ):
        if self.log_val:
            self.log_img(pl_module, batch, batch_idx, split="val")

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0
    ):
        self.log_img(pl_module, batch, batch_idx, split="test")

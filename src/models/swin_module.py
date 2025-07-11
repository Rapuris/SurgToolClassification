from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from PIL import Image
import numpy as np
import timm
import torchmetrics

from ..utils import pylogger
from .losses import DiceLoss2D, HeatmapLoss2D

log = pylogger.get_pylogger(__name__)


class SwinTransformerModule(LightningModule):
    """LightningModule for Swin Transformer image classification with checkpoint loading support."""
    def __init__(
        self,
        num_classes: int,
        optimizer: list[torch.optim.Optimizer],
        scheduler: Optional[list[torch.optim.lr_scheduler.LRScheduler]] = None,
        model_name: str = "swin_tiny_patch4_window7_224",
        use_pretrained: bool = True,
        checkpoint_path: str = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model = timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=True, num_classes=14
        )
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.val_acc = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.test_acc = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes, average="macro")

        self.train_precision = torchmetrics.classification.MulticlassPrecision(num_classes=num_classes, average="macro")
        self.val_precision = torchmetrics.classification.MulticlassPrecision(num_classes=num_classes, average="macro")
        self.test_precision = torchmetrics.classification.MulticlassPrecision(num_classes=num_classes, average="macro")

        self.train_recall = torchmetrics.classification.MulticlassRecall(num_classes=num_classes, average="macro")
        self.val_recall = torchmetrics.classification.MulticlassRecall(num_classes=num_classes, average="macro")
        self.test_recall = torchmetrics.classification.MulticlassRecall(num_classes=num_classes, average="macro")

        self.train_auc = torchmetrics.classification.MulticlassAUROC(num_classes=num_classes, average="macro")
        self.val_auc = torchmetrics.classification.MulticlassAUROC(num_classes=num_classes, average="macro")
        self.test_auc = torchmetrics.classification.MulticlassAUROC(num_classes=num_classes, average="macro")

    def configure_optimizers(self):
        opt = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            sched = self.hparams.scheduler(optimizer=opt)
            return {"optimizer": opt, "lr_scheduler": sched, "monitor": "val/loss"}
        return opt

    def forward(self, x):
        return self.model(x)

    def model_step(self, batch, batch_idx, mode):
        _, images, labels = batch
        #log.error(images.shape)
        #log.error(labels.shape)
        logits = self(images)
        #log.error(logits.shape)
        loss = self.criterion(logits, labels)
        #log.error(loss)
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        #log.error(preds.shape)
        #log.error(probs.shape)
        # Metrics
        if mode == "train":
            self.train_acc.update(preds, labels)
            self.train_precision.update(preds, labels)
            self.train_recall.update(preds, labels)
            self.train_auc.update(probs, labels)
            self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/precision", self.train_precision, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/recall", self.train_recall, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/auc", self.train_auc, on_step=True, on_epoch=True, prog_bar=True)
        elif mode == "val":
            self.val_acc.update(preds, labels)
            self.val_precision.update(preds, labels)
            self.val_recall.update(preds, labels)
            self.val_auc.update(probs, labels)
            self.log("val/acc", self.val_acc, on_step=True, on_epoch=True, prog_bar=True)
            self.log("val/precision", self.val_precision, on_step=True, on_epoch=True, prog_bar=True)
            self.log("val/recall", self.val_recall, on_step=True, on_epoch=True, prog_bar=True)
            self.log("val/auc", self.val_auc, on_step=True, on_epoch=True, prog_bar=True)
        elif mode == "test":
            self.test_acc.update(preds, labels)
            self.test_precision.update(preds, labels)
            self.test_recall.update(preds, labels)
            self.test_auc.update(probs, labels)
            self.log("test/acc", self.test_acc, on_step=True, on_epoch=True, prog_bar=True)
            self.log("test/precision", self.test_precision, on_step=True, on_epoch=True, prog_bar=True)
            self.log("test/recall", self.test_recall, on_step=True, on_epoch=True, prog_bar=True)
            self.log("test/auc", self.test_auc, on_step=True, on_epoch=True, prog_bar=True)

        self.log(f"{mode}/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.model_step(batch, batch_idx, "train")
        opt = self.optimizers()
        current_lr = opt.param_groups[0]['lr']
        # Log the learning rate: log on every step and show it in the progress bar.
        self.log("lr", current_lr, on_step=True, on_epoch=False, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        loss = self.model_step(batch, batch_idx, "val")
        return loss


    def test_step(self, batch, batch_idx):
        loss = self.model_step(batch, batch_idx, "test")
        return loss

    
    def log_images(self, batch, **kwargs):
        orig_image, transformed_img, class_idx = batch
        #log.error(f"orig image mean: {orig_image.mean()}")
        #log.error(f"transformed image mean: {transformed_img.mean()}")
        input_images = orig_image.detach().cpu()
        target_images = transformed_img.detach().cpu()
        
        return dict(
            input_images=input_images,
            target_images=target_images
        )


from typing import Any, Optional

import torch
import torch.nn.functional as F
from lightning import LightningModule
from PIL import Image
import numpy as np
import timm

from ..utils import pylogger
from .losses import DiceLoss2D, HeatmapLoss2D
from .components import SwinTransformerUnet

log = pylogger.get_pylogger(__name__)


class SwinTransformerModule(LightningModule):
    """LighningModule for training a SWIN module.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        num_classes: int,
        num_keypoints: int,
        swin_unet: SwinTransformerUnet,
        optimizer: list[torch.optim.Optimizer],
        scheduler: Optional[list[torch.optim.lr_scheduler.LRScheduler]] = None,
        use_pretrained: bool = True,
        checkpoint_path: str = None,
    ):
        """
        Args:
            optimizers: The VQ optimizer and the discriminator optimizer, partially initialized
            schedulers: list of schedulers for each optimizer, or None.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        if use_pretrained:
            checkpoint_path = '/nfs/centipede/sampath/swin_tiny_patch4_window7_224_22k.pth' # using swin small now
            #checkpoint_path = '/data1/sampath/swin_small_patch4_window7_224_22k.pth'
            #checkpoint_path = '/projects/synfbct_trial_unberathlab/sampath/swin_small_patch4_window7_224_22k.pth'
            #try upgrading to swin small
            pretrained_dict = torch.load(checkpoint_path, map_location='cpu')
            self.swin_unet = swin_unet
            # Load the checkpoint into your model; strict=False allows missing or mismatched keys.
            self.swin_unet.load_state_dict(pretrained_dict, strict=False)
            log.info("pretrained finetuning")
        else:
            self.swin_unet = swin_unet
        
        self.dice_loss = DiceLoss2D(skip_bg=False)
        self.heatmap_loss = HeatmapLoss2D()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        opt = self.hparams.optimizer
        opt = opt(params=self.parameters())

        if self.hparams.scheduler is not None:
            sched = self.hparams.scheduler
            sched = sched(optimizer=opt)
            return {"optimizer": opt, "lr_scheduler": sched, "interval": "step", "frequency": 1,}

        return opt

    def forward(self, x: torch.Tensor):
        # TODO: implement forward pass
        outputs = self.swin_unet(x)
        return dict(
            segs=outputs[:, : self.num_classes], heatmaps=outputs[:, self.num_classes :]
        )
    
    

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        pass

    def model_step(
        self,
        batch: Any,
        batch_idx: int,
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            inputs: dict[str, torch.Tensor]
            targets: dict[str, torch.Tensor]
            inputs, targets = batch

            imgs = inputs["image"]  # intensity augmentations applied
            target_imgs = targets["image"]  # no intensity augmentations

            assert np.isnan(imgs.cpu().detach().numpy()).sum() == 0, 'input to model have no nans'
            assert np.isinf(imgs.cpu().detach().numpy()).sum() == 0, 'input to model have no infs'
            log
            outputs = self(imgs)
            decoded_segs = outputs["segs"]
            assert np.isnan(decoded_segs.cpu().detach().numpy()).sum() == 0, f'segs have nans in them, { np.isnan(decoded_segs.cpu().detach().numpy()).sum()}'
            decoded_heatmaps = outputs["heatmaps"]
            assert np.isnan(decoded_heatmaps.cpu().detach().numpy()).sum() == 0, 'heatmaps have nans in them'
            #log.info(decoded_heatmaps.shape)
            #log.info(decoded_segs.shape)
            #log.info(targets['segs'].shape)
            
            target_seg = targets["segs"]
            #target_seg = targets["segs"].cpu().numpy()
            #target_seg = torch.as_tensor(target_seg)
            target_seg = (target_seg > 0).int()

            decoded_seg_clone = decoded_segs.clone()  # Ensure we don't modify the original tensor in-place
            #train on DRR eval on ROb
            #decoded_segs[:, 3] = decoded_seg_clone[:, 4]
            #decoded_segs[:, 4] = decoded_seg_clone[:, 2]
            #decoded_segs[:, 2] = decoded_seg_clone[:, 5]
            #decoded_segs[:, 5] = decoded_seg_clone[:, 3]

            #train on Rob eval on DRR
            #decoded_segs[:, 4] = decoded_seg_clone[:, 3]
            #decoded_segs[:, 2] = decoded_seg_clone[:, 4]
            #decoded_segs[:, 5] = decoded_seg_clone[:, 2]
            #decoded_segs[:, 3] = decoded_seg_clone[:, 5]

            #train on DRR eval on Rob
            #mask = torch.tensor([0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13], dtype=torch.long)
            #mask = torch.tensor([0, 1, 2, 3, 5, 6, 8, 9, 10, 11 ,13, 14])
            #decoded_heatmaps = decoded_heatmaps[:, mask, :, :]
            #order_idx = torch.tensor([0, 1, 3, 2, 4, 5, 6, 9, 10, 11, 12, 13], dtype=torch.long)
            #order_idx = torch.tensor([12, 8, 6, 4, 10, 0, 13, 9, 7, 5, 11, 1])
            target_heatmaps = targets["heatmaps"]
            #target_heatmap = target_heatmaps[:, order_idx, :, :]

            #train on Rob eval on DRR
            #mask = torch.tensor([0, 1, 3, 2, 4, 5, 6, 9, 10, 11, 12, 13], dtype=torch.long)
            #decoded_heatmaps = decoded_heatmaps[:, mask, :, :]
            #order_idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13], dtype=torch.long)
            #target_heatmap = targets["heatmaps"]
            #target_heatmap = target_heatmap[:, order_idx, :, :]
            
            # Reorder target heatmaps (if they are also in DRR order)
            #target_heatmap = targets["heatmaps"][:, order_idx, :, :]
            
            #
            # seg_loss = self.dice_loss(decoded_segs, target_seg)
            # #log.info(seg_loss)
            #self.log(f"{mode}/seg_loss", seg_loss, on_step=True)
            # #log.info(targets['heatmaps'].shape)
            # #decoded_heatmaps = decoded_heatmaps[:, [0, 8, 1, 9, 2, 10, 3, 11, 6, 14, 7, 15], :, :]
            # #log.info('decoded rearranged')
            # #log.info(decoded_heatmaps.shape)
            # target_heatmap = targets["heatmaps"]
            # #log.info('test')
            # target_heatmap = torch.cat((target_heatmap[:, :2], target_heatmap[:, 4:]), dim=1)
            # #log.info('target rearranged')
            # #log.info(target_heatmap)
            # #log.info(target_heatmap.shape)
            # heatmap_loss = self.heatmap_loss(decoded_heatmaps, target_heatmap) # 
            # self.log(f"{mode}/heatmap_loss", heatmap_loss, on_step=True)
                #target_seg_clone = target_seg.clone()
                #target_seg[:, 3] = target_seg_clone[:, 4]
                #target_seg[:, 4] = target_seg_clone[:, 2]
                #target_seg[:, 2] = target_seg_clone[:, 5]
                #target_seg[:, 5] = target_seg_clone[:, 3]
    #
            seg_loss = self.dice_loss(decoded_segs, target_seg)
            self.log(f"{mode}/seg_loss", seg_loss, on_step=True)
    #
                ## For heatmaps, now apply the concat/slice operation to decoded_heatmaps
                ## instead of target_heatmap
                #decoded_heatmaps = torch.cat((decoded_heatmaps[:, :2], decoded_heatmaps[:, 4:]), dim=1)
                #order_idx = torch.tensor([0, 11, 7, 15, 8, 5, 13, 1, 9, 2, 10, 3], dtype=torch.long)
    #
                ## Assume target_heatmap is of shape (batch, channels, height, width)
                #target_heatmap = targets["heatmaps"][:, order_idx, :, :]
                ##target_heatmap = targets["heatmaps"]
    #
            heatmap_loss = self.heatmap_loss(decoded_heatmaps, target_heatmaps)
            self.log(f"{mode}/heatmap_loss", heatmap_loss, on_step=True)

            loss = heatmap_loss + seg_loss 

            return loss, decoded_segs, decoded_heatmaps, target_imgs # 

    def training_step(self, batch: Any, batch_idx: int):
        loss, _, _, _, = self.model_step(batch, batch_idx, "train")
        opt = self.optimizers()
        current_lr = opt.param_groups[0]['lr']
        
        # Log the learning rate: log on every step and show it in the progress bar.
        self.log("lr", current_lr, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss,  decoded_segs, decoded_heatmaps, target_imgs = self.model_step( # 
            batch, batch_idx, "val"
        )
        # TODO: compute the keypoint error
        return loss

    def test_step(self, batch: Any, batch_idx: int):
        loss, decoded_segs, decoded_heatmaps, target_imgs = self.model_step( #  
            batch, batch_idx, "test"
        )
        return loss

    def log_images(self, batch, **kwargs):
        inputs, targets = batch
        imgs = inputs["image"]
        target_imgs = targets["image"]
        outputs = self(imgs)
        decoded_segs = outputs["segs"]
        decoded_seg_clone = decoded_segs.clone()  # Ensure we don't modify the original tensor in-place

        #train on DRR eval on Rob
        #decoded_segs[:, 3] = decoded_seg_clone[:, 4]
        #decoded_segs[:, 4] = decoded_seg_clone[:, 2]
        #decoded_segs[:, 2] = decoded_seg_clone[:, 5]
        #decoded_segs[:, 5] = decoded_seg_clone[:, 3]


        #train on Rob eval on DRR
        #decoded_segs[:, 4] = decoded_seg_clone[:, 3]
        #decoded_segs[:, 2] = decoded_seg_clone[:, 4]
        #decoded_segs[:, 5] = decoded_seg_clone[:, 2]
        #decoded_segs[:, 3] = decoded_seg_clone[:, 5]

        decoded_heatmaps = outputs["heatmaps"]

        #train on DRR eval on Rob
        #mask = torch.tensor([0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13], dtype=torch.long)
        #decoded_heatmaps = decoded_heatmaps[:, mask, :, :]
        #order_idx = torch.tensor([0, 1, 3, 2, 4, 5, 6, 9, 10, 11, 12, 13], dtype=torch.long)
        target_heatmaps = targets["heatmaps"]
        #target_heatmap = target_heatmaps[:, order_idx, :, :]

        #train on Rob eval on DRR
        #mask = torch.tensor([0, 1, 3, 2, 4, 5, 6, 9, 10, 11, 12, 13], dtype=torch.long)
        #decoded_heatmaps = decoded_heatmaps[:, mask, :, :]
        #order_idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13], dtype=torch.long)
        #target_heatmap = targets["heatmaps"]
        #target_heatmap = target_heatmap[:, order_idx, :, :]
            
        

        #decoded_heatmaps = outputs["heatmaps"]
        #decoded_heatmaps = decoded_heatmaps[:, [0, 8, 1, 9, 2, 10, 3, 11, 6, 14, 7, 15], :, :]
            #log.info('decoded rearranged')
            #log.info(decoded_heatmaps.shape)
        #target_heatmap = targets["heatmaps"]
            #log.info('test')
        #target_heatmap = torch.cat((target_heatmap[:, :2], target_heatmap[:, 4:]), dim=1)
          ##target_seg = targets["segs"]
          ##target_seg_clone = target_seg.clone()
          ##target_seg[:, 3] = target_seg_clone[:, 4]
          ##target_seg[:, 4] = target_seg_clone[:, 2]
          ##target_seg[:, 2] = target_seg_clone[:, 5]
          ##target_seg[:, 5] = target_seg_clone[:, 3]
  ##
          ###seg_loss = self.dice_loss(decoded_segs, target_seg)
  ##
          ### For heatmaps, now apply the concat/slice operation to decoded_heatmaps
          ### instead of target_heatmap
          ##decoded_heatmaps = outputs["heatmaps"]
          ##decoded_heatmaps = torch.cat((decoded_heatmaps[:, :2], decoded_heatmaps[:, 4:]), dim=1)
          ##order_idx = torch.tensor([0, 11, 7, 15, 8, 5, 13, 1, 9, 2, 10, 3], dtype=torch.long)
  ##
          ##    # Assume target_heatmap is of shape (batch, channels, height, width)
          ##target_heatmap = targets["heatmaps"][:, order_idx, :, :]
    
    
        input_images = imgs.detach().cpu()
        target_images = target_imgs.detach().cpu()
        target_segs = targets["segs"].detach().cpu()
        target_heatmaps = target_heatmaps.detach().cpu() #targets["heatmaps"].detach().cpu()
        target_keypoints = targets["keypoints"].detach().cpu()

        

        return dict(
            input_images=input_images,
            target_images=target_images,
            target_segs=target_segs,
            target_heatmaps=target_heatmaps,
            target_keypoints=target_keypoints,
            decoded_heatmaps=decoded_heatmaps,
            decoded_segs=decoded_segs,
        )

#
# 
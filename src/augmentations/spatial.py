import numpy as np
import albumentations as A
import cv2
from .crop_Rob import SymmetricHorizontalFlip
from .utils import augmentation_builder
import logging


log = logging.getLogger(__name__)
KEYPOINT_NAMES = [
    "l_asis", "l_gsn", "l_iof", "l_ips", "l_mof", "l_sps",
    "r_asis", "r_gsn", "r_iof", "r_ips", "r_mof", "r_sps"
]

LR_PAIRS = [
    (0, 6),  # l_asis ↔ r_asis
    (1, 7),  # l_gsn  ↔ r_gsn
    (2, 8),  # l_iof  ↔ r_iof
    (3, 9),  # l_ips  ↔ r_ips
    (4, 10), # l_mof  ↔ r_mof
    (5, 11)  # l_sps  ↔ r_sps
]

def assert_float32(img, **kwargs):
    assert img.dtype == np.float32, f"Expected float32 but got {img.dtype}"
    #assert(1==2)
    return img

@augmentation_builder
def spatial_augmentation() -> A.Sequential:
    """Build an augmentation only involving spatial transformations.

    Note: keypoints might not allow some of these.

    Args:
        train: Whether to build an augmentation for training or testing. If True, the wrapped
            function is used to get the training augmentations.

        annotations: Whether the dataset contains annotations.
        image_size: The size to resize images to. If None, no resizing is done.
        normalize: Whether to normalize the image to [-1, 1].

    """

    #add more spatial transforms 
    #just rotate 
    #-45 to 45 
    #use the flip function from Jeremy 
    #log.info("spatial augmentation happening")
    #assert(1==2)
    return A.Compose(
        [
            #A.RandomCrop(height=150, width=150, p = 0.5),#FromBorders(),
            A.RandomResizedCrop(size = (224,224), scale = (0, 0.3), p = 0.3),
            #A.ElasticTransform(keypoint_remapping_method = 'direct'),
            #don't do crop from border on simulated 

            #random crop, elastic
            #A.CropAndPad( # only keep for Rob data 
            #    px=-20,  # Negative for cropping should be -20 
            #    keep_size=False,  # Don't resize within this transform
            #    p=1.0
            #),
            A.SomeOf([
                 A.OneOf([
                    #A.Rotate(limit=(-45, 45), crop_border=True, p=0.5),
                   A.Affine( #  this is the one that messes up the keypoints
                   scale=(0.6, 1.2),                     
                   translate_percent={"x":(-0.1,0.1), "y":(-0.1,0.1)},
                   rotate=(-30, 30),
                   shear={"x":(-5,5), "y":(-5,5)},
                   balanced_scale=True,
                   p=0.3
                   ),
                    A.RandomRotate90(p=0.7),
                 ], p=0.7),
                A.OneOf([
                   #A.ElasticTransform(),
                   A.GridDistortion(),
                ], p=0.5),
                A.OpticalDistortion(distort_limit=(-0.05, 0.05), p = 0.5),
                A.OneOf([
                    A.Perspective(p = 0.5),
                    #A.ThinPlateSpline(p = 0.5),
                ], p=0.5),
            ], n=np.random.randint(1, 4)),
            
            SymmetricHorizontalFlip(lr_pairs=LR_PAIRS, p=0.5),
            #A.Resize(224, 224),
            #A.Lambda(name="assert_beforefromfloat", image=assert_float32),
        ],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
    )


import numpy as np
import albumentations as A
import cv2
from .crop_Rob import SymmetricHorizontalFlip
from .utils import augmentation_builder
import logging


log = logging.getLogger(__name__)

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

    return A.Compose(
        [
            A.RandomCrop(height=170, width=170, p = 0.4),#FromBorders(),
            #A.RandomResizedCrop(size = (224,224), scale = (0, 0.1), p = 0.1),
            A.SomeOf([
                 A.OneOf([
                   A.Affine(
                   scale=(0.6, 1.2),                     
                   translate_percent={"x":(-0.1,0.1), "y":(-0.1,0.1)},
                   rotate=(-30, 30),
                   shear={"x":(-5,5), "y":(-5,5)},
                   balanced_scale=True,
                   p=0.3
                   ),
                    A.RandomRotate90(p=0.7),
                 ], p=0.7),
                A.OpticalDistortion(distort_limit=(-0.05, 0.05), p = 0.3),
                A.Perspective(p = 0.3),
                A.OneOf([
                A.ElasticTransform(alpha=0.5, sigma=25, p=0.3),
                A.GridDistortion(num_steps=3, distort_limit=0.05, p=0.2),
            ], p=0.2),
            ], n=np.random.randint(1, 3)),
        ],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
    )


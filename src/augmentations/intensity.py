import numpy as np
import albumentations as A
from .dropout import Dropout, CoarseDropout
from .gaussian_contrast import gaussian_contrast
from .utils import augmentation_builder
from .window import window, adjustable_window, mixture_window


def assert_uint8(img, **kwargs):
    assert img.dtype == np.uint8, f"Expected uint8 but got {img.dtype}"
    #assert(1==2)
    return img
def assert_float32(img, **kwargs):
    assert img.dtype == np.float32, f"Expected float32 but got {img.dtype}"
    #assert(1==2)
    return img



@augmentation_builder
def intensity_augmentation() -> A.SomeOf:
    """Build an augmentation only involving intensity transformations.

    That is, these augmentations do not "move pixels around," only adjust their values. These are
    augmentations that a model should be invariant to.

    Args:
        train: Whether to build an augmentation for training or testing. If True, the wrapped
            function is used to get the training augmentations.

        annotations: Whether the dataset contains annotations.
        image_size: The size to resize images to. If None, no resizing is done.
        normalize: Whether to normalize the image to [-1, 1].

#tune up the brightness tone curve 
#bring lr down by 2 factors of 10 after it "converges"
    """
    return A.Compose(
        [
            A.InvertImg(p=0.4),  # Increased probability
            A.SomeOf(
                [
                    A.OneOf(
                        [
                            A.GaussianBlur((3, 7)),  # Increased blur range
                            A.MotionBlur(blur_limit=(3, 7)),  # Increased blur range
                            A.MedianBlur(blur_limit=(3, 7)),  # Increased blur range
                            A.Blur(blur_limit=(3, 7)),  # Added general blur
                        ],
                    ),
                    A.OneOf(
                        [
                            A.Sharpen(alpha=(0.3, 0.7)),  # Increased sharpening
                            A.Emboss(alpha=(0.3, 0.7)),  # Increased embossing
                            A.CLAHE(clip_limit=(2, 6)),  # Increased CLAHE range
                            A.UnsharpMask(blur_limit=(3, 7), alpha=(0.2, 0.5)),  # Added unsharp mask
                        ],
                    ),
                    A.OneOf(
                        [
                            A.MultiplicativeNoise(multiplier=(0.8, 1.2)),  # Increased range
                            A.HueSaturationValue(
                                hue_shift_limit=25,  # Increased hue shift
                                sat_shift_limit=35,  # Increased saturation shift
                                val_shift_limit=25,  # Increased value shift
                            ),
                            A.RandomBrightnessContrast(
                                brightness_limit=(-0.3, 0.3),   # Increased brightness range
                                contrast_limit=(-0.3, 0.3),  # Increased contrast range
                                p=0.7  # Increased probability
                            ),
                            A.RandomGamma(p=0.5),  # Increased probability
                            gaussian_contrast(
                                alpha=(0.5, 1.5), sigma=(0.1, 0.8), max_value=1  # Increased range
                            ),
                        ],
                    ),
                    A.RandomToneCurve(scale=0.2),  # Increased scale
                    A.OneOf(
                        [
                            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=3),
                            A.RandomFog(
                                fog_coef_range = (0.1, 0.4), alpha_coef=0.1  # Increased fog
                            ),
                        ],
                    ),
                    # A.Posterize(),
                    A.OneOf(
                        [
                            Dropout(dropout_prob=0.1),  # Increased dropout
                            CoarseDropout(
                                num_holes_range = (6, 16),  # Increased holes
                                hole_height_range = (6, 32),  # Increased hole size
                                hole_width_range = (6, 32)  # Increased hole size
                            ),
                        ],
                        p=0.5,  # Increased probability
                    ),
                    A.SaltAndPepper(p=0.7),  # Increased probability
                    A.MultiplicativeNoise(multiplier=(0.7, 1.3), p = 0.5),  # Increased range and probability
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),  # Added ISO noise
                    A.GaussNoise(var_limit=(10.0, 50.0)),  # Added Gaussian noise
                    A.OneOf([
                        A.RandomGamma(gamma_limit=(80, 120)),  # Added gamma correction
                        A.RandomToneCurve(scale=0.2),  # Increased tone curve scale
                    ]),
                    A.OneOf([
                        A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=35, val_shift_limit=25),
                        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25),  # Added RGB shift
                    ]),
                ],
                n=np.random.randint(1, 3),  # Increased number of augmentations
                replace=False,
            ),
        ],
       # keypoint_params=A.KeypointParams(format='xy', remove_invisible=True)
    )

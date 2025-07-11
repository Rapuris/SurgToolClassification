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
def prephix_intensity_augmentation() -> A.SomeOf: 
    """
    revised pipeline given that ive not been doing this right for the past few months

    """
    #clahe = A.Sequential(
    #    [
    #        A.FromFloat(max_value=255, dtype="uint8", p = 1),
    #        A.CLAHE(clip_limit=(4, 6), tile_grid_size=(8, 12), p = 1),
    #        A.ToFloat(max_value=255, p = 1),
    #    ],
    #    p=1,
    #)

    clahe = A.Sequential(
        [
            #A.Lambda(name="assert_beforefromfloat", image=assert_float32),
            A.FromFloat(max_value=255, dtype="uint8", p = 1),
            #A.Lambda(name="assert_after_fromfloat", image=assert_uint8),
            A.CLAHE(clip_limit=(4, 6), tile_grid_size=(8, 12), p = 1),
            A.ToFloat(max_value=255, p = 1),
        ],
        p=1,
    )

    intensity_transforms = A.SomeOf(
        [
            A.OneOf(
                [
                    A.GaussianBlur((3, 5)),
                    A.MotionBlur(blur_limit=(3, 5)),
                    A.MedianBlur(blur_limit=5),
                ],
            ),
            A.OneOf(
                [
                    A.Sharpen(alpha=(0.2, 0.5)),
                    A.Emboss(alpha=(0.2, 0.5)),
                ],
            ),
            A.PlanckianJitter(mode="blackbody"),  # Fine
            A.RandomBrightnessContrast(brightness_limit=(0, 0), contrast_limit=(-2, 2)),  # Fine
            gaussian_contrast(alpha=(0.6, 1.4), sigma=(0.1, 0.5), max_value=1),
            A.OneOf(
                [
                    A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.08),
                    A.RandomSnow(snow_point_range=(0.1,0.3)),
                    A.RandomRain(rain_type="drizzle", drop_width=1, blur_value=1),
                ],
            ),
            A.OneOf(
                [
                    Dropout(dropout_prob=0.05),
                    CoarseDropout(
                        num_holes_range = (4, 12),
                        hole_height_range = (4, 24),
                        hole_width_range = (4, 24)
                    ),
                ],
                p=3,
            ),
            A.OneOf([
                A.SaltAndPepper(p=0.5),
                A.MultiplicativeNoise(multiplier=(0.8, 1.2), p = 0.5)
            ]),
            A.ChromaticAberration(),
        ],
        n=np.random.randint(2, 6),
        replace=False,
    )
    return A.Compose(
        [
            #neglog() the images are already neglogd
            A.InvertImg(p = 0.1),
            A.OneOf(
                [
                    window(0, 1.0, convert=False),
                    adjustable_window(-4, 4),
                    mixture_window(keep_original=True, model="kmeans"),
                ],
                p=1.0,
            ),
            A.InvertImg(p=0.5),
            A.OneOf(
                [clahe, intensity_transforms],
            ),
        ],
    )

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
            A.InvertImg(p=0.3),
            A.SomeOf(
                [
                    A.OneOf(
                        [
                            A.GaussianBlur((3, 5)),
                            A.MotionBlur(blur_limit=(3, 5)),
                            A.MedianBlur(blur_limit=5),
                        ],
                    ),
                    A.OneOf(
                        [
                            A.Sharpen(alpha=(0.2, 0.5)), # i don't think IAASharpen exists anymore
                            A.Emboss(alpha=(0.2, 0.5)), # i don't think IAAEmboss exists anymore
                            A.CLAHE(clip_limit=(1, 4)),
                        ],
                    ),
                    A.OneOf(
                        [
                            A.MultiplicativeNoise(multiplier=(0.9, 1.1)),
                            A.HueSaturationValue(
                                hue_shift_limit=20,
                                sat_shift_limit=30,
                                val_shift_limit=20,
                            ),
                            A.RandomBrightnessContrast(
                                brightness_limit=(-0.2, 0.0),   # only darker
                                contrast_limit=(-0.2, 0.2),
                                p=0.5
                            ),
                            A.RandomGamma(p=0.4),
                            gaussian_contrast(
                                alpha=(0.6, 1.4), sigma=(0.1, 0.5), max_value=1 #255
                            ),
                        ],
                    ),
                    A.RandomToneCurve(scale=0.1),
                    A.OneOf(
                        [
                            A.RandomShadow(),
                            A.RandomFog(
                                fog_coef_range = (0.1, 0.3), alpha_coef=0.08
                            ),
                        ],
                    ),
                    # A.Posterize(),
                    A.OneOf(
                        [
                            Dropout(dropout_prob=0.05),
                            CoarseDropout(
                                num_holes_range = (4, 12),
                                hole_height_range = (4, 24),
                                hole_width_range = (4, 24)
                            ),
                        ],
                        p=3,
                    ),
                    A.SaltAndPepper(),
                    A.MultiplicativeNoise(multiplier=(0.8, 1.2), p = 0.3)
                ],
                n=np.random.randint(1, 3), # up to 5 before
                replace=False,
            ),
        ],
       # keypoint_params=A.KeypointParams(format='xy', remove_invisible=True)
    )

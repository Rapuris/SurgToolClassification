"""Version of the perphix dataset with custom augmentation pipeline, etc."""

from typing import Optional, Callable
import albumentations as A
import numpy as np
from decohints import decohints
import logging
import sys
#sys.path.insert(0, '/nfs/centipede/sampath/prephixSynth/src')
#from prephix.alb.gaussian_contrast import gaussian_contrast
from prephix.alb.window import window, adjustable_window, mixture_window
from prephix.alb.neglog import neglog


log = logging.getLogger(__name__)

def instance_normalize() -> A.Lambda:
    """Normalize an image instance to the range [-1, 1] based on its own min and max values.
    Now it just [0, 1] as i removed the *2 -1 part

    Args:
        epsilon (float): A small constant added to the denominator to prevent division by zero
                         in case the image has little to no variation.

    Returns:
        A.Lambda: An Albumentations lambda transform that normalizes images instance-wise.
    """
    def f_image(image: np.ndarray, **kwargs) -> np.ndarray:
        # Compute the minimum and maximum pixel values of the image.
        imin, imax = image.min(), image.max()
        # Check if the range is too small to avoid division by zero
        #assert imin != imax, f'max and min of image are equal {imin} == {imax}'
        if imax == imin:
           # If the image is nearly constant, return an array of zeros.
            assert(1==2), 'this should never happen dude u messed up somewhere'
            return np.zeros_like(image)
        # Normalize image values to the range [0, 1]
        img_norm = (image - imin) / (imax - imin) # min max normalization
        # Map the normalized image to the range [-1, 1]
        assert img_norm.max() <= 1.0, f"Image max is {img_norm.max()}, expected <= 1.0"
        assert img_norm.min() >= 0.0, f"Image min is {img_norm.min()}, expected >= 0.0"
        return img_norm #(img_norm * 2) - 1 

    def f_identity(x, **kwargs):
        # For masks, keypoints, and bboxes, just pass the values through without changes.
        return x

    return A.Lambda(
        image=f_image,
        mask=f_identity,
        keypoints=f_identity,
        bboxes=f_identity,
        name="instance_normalize"
    )

crop_border = 5

border_crop = A.Lambda(
    image=lambda img, **kwargs: img[crop_border : img.shape[0] - crop_border,
                                   crop_border : img.shape[1] - crop_border],
    mask=lambda mask, **kwargs: (
        mask[crop_border : mask.shape[0] - crop_border,
             crop_border : mask.shape[1] - crop_border]
        if mask is not None else None
    )
)

def fill_border(img: np.ndarray, border: int = crop_border) -> np.ndarray:
    """
    Replace a border of width `border` with the mean of the interior pixels.
    Works for H×W (grayscale) or H×W×C images.
    """
    # compute mean over the interior region (exclude the border)
    if img.ndim == 2:      # grayscale
        mean_val = img[border:-border, border:-border].mean()
        img[:border, :]        = mean_val   # top
        img[-border:, :]       = mean_val   # bottom
        img[:, :border]        = mean_val   # left
        img[:, -border:]       = mean_val   # right
    else:                  # colour / multi‑channel
        mean_val = img[border:-border, border:-border, :].mean(axis=(0, 1))
        #mean_val = 0
        img[:border, :, :]     = mean_val
        img[-border:, :, :]    = mean_val
        img[:, :border, :]     = mean_val
        img[:, -border:, :]    = mean_val
    return img

border_fill = A.Lambda(
    image=lambda img, **kw: fill_border(img, border=crop_border),
    mask=lambda m, **kw: m  # leave masks untouched; change if you want
)

# Define named functions for all transformations
def to_3channel(img: np.ndarray, **kwargs) -> np.ndarray:
    """Convert single channel to 3-channel by replication"""
    #sanity check
    three_image = np.stack((img,)*3, axis=-1) if img.ndim == 2 else img
    #they're in 0 to 1 range 
    #three_image = three_image.astype(np.uint8)
    assert three_image.shape[-1] == 3, f"Image shape is {three_image.shape}, expected 3 channels"
    assert np.isnan(three_image).sum() == 0, f"Image contains NaN values: {np.isnan(three_image).sum()} NaNs"
    #log.error(three_image.shape)
    return three_image

def to_1channel(img: np.ndarray, **kwargs) -> np.ndarray:
    """Convert three channel to 1-channel by slicing"""
    return img[:, :, 0]

@decohints
def augmentation_builder(builder: Callable) -> Callable:
    """Decorator for augmentation builders.

    Args:
        builder: The builder to decorate. This should take no arguments and return a base-level augmentation (not a Compose.)
            It should assume uint8 inputs.

    Returns:
        build_augmentation: New function that takes in
    """

    def build_augmentation(
        train: bool = True,
        annotations: bool = True,
        image_size: Optional[tuple[int, int]] = None,
        normalize: bool = True,
        do_neglog: bool = True,
        three_channel: bool = True,
        test: bool = False,
        last_test: bool = False,
    ) -> A.Compose:
        docstring = f"""Build an augmentation pipeline.

        {builder.__doc__}

        Args:
            train: Whether to build an augmentation for training or testing. If True, the wrapped
                function is used to get the training augmentations.

            annotations: Whether the dataset contains annotations.
            image_size: The size to resize images to. If None, no resizing is done.
            normalize: Whether to normalize the image to [-1, 1].

        Returns:
            A.Compose: The augmentation pipeline.
        """

        builder.__doc__ = docstring

        resize = A.Resize(*image_size) if image_size is not None else A.NoOp()
        #normalizer = (
            #A.Normalize(
            #    mean=[0.5, 0.5, 0.5],
            #    std=[0.5, 0.5, 0.5],
            #    max_pixel_value=1,
            #    p = 1,
            #)
        #    instance_normalize(),
        #    if normalize:
        #    else A.NoOp()
        #)
        normalizer =  A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=1) #instance_normalize() if normalize else A.NoOp()
        # could also do A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255) but it looks like shit
        if three_channel:
            #log.error("you're three channeling")
            channeler = A.Lambda(image=to_3channel, name='to_3channel')
        else:
            #assert(1==2), 'this should never happen, something is wrong with your dataset'
            channeler = A.NoOp()

        if do_neglog:
            #neglog(),  bruh i was doing this on already neglog'd images 
            #images are already normalized from 0 to 1
            #log.error(channeler)
        #mixture_window(model = 'kmeans')
            neglog_op = A.Compose([channeler])
            #neglog_op = neglog()
            #windowing = window(0.00, 0.70, convert=True)# 0.9 qualitatively is p good 
            #the contrast was messed up with my windowing 
            #windowing = mixture_window(model = "kmeans")
        else:
            neglog_op = A.NoOp()
            #windowing = A.NoOp()
        
        neglog_op = A.Compose([channeler])
        one_channeler = A.Lambda(image=to_1channel, name='to_1channel')


        # TODO: handle empty keypoints/stuff

        def transform_func(
            *,
            image: np.ndarray,
            bboxes: list,
            category_ids: list,
            masks: list,
            keypoints: list = [],
        ) -> dict:
            """Apply the transform to an image and annotations.

            Filter args, so if any are empty, get rid of the annotations.
            """
            kwargs = dict()
            transform_kwargs = dict(image=image)
            if annotations and len(bboxes) > 0:
                kwargs["bbox_params"] = A.BboxParams(
                    format="coco",
                    label_fields=["category_ids"],
                    min_visibility=0.1,
                    min_area=10,
                )
                transform_kwargs["bboxes"] = bboxes
                transform_kwargs["category_ids"] = category_ids
            assert((len(keypoints) == 12)), f'something got screwed up {len(keypoints)}'
            if len(keypoints) > 0:
                #log.error("don't remove the invisible keypoints")
                kwargs["keypoint_params"] = A.KeypointParams(
                    format="xy", remove_invisible=False
                )
                transform_kwargs["keypoints"] = keypoints
            else:
                assert(1==2)

            if annotations and len(masks) > 0:
                transform_kwargs["masks"] = masks

            if train and last_test:
                #
                transform = A.Compose([A.ToFloat(p = 1), neglog_op, builder(), resize, normalizer], **kwargs) #normalizer, channeler, windowing, 
            elif train and not last_test:
                #A.ToFloat(p = 1)
                transform = A.Compose([A.ToFloat(p = 1), neglog_op, builder(), resize], **kwargs)
                #log.error(transform)
            elif last_test and test:
                #transform = A.Compose(
                #[
                    #A.InvertImg(always_apply=True),
                    #mixture_window(keep_original=True, model="kmeans"),
                    #A.InvertImg(always_apply=True),
                #    resize,
                #    normalizer
                #])
                #log.info(transform)
                #log.error("you're in the test time augmentation")
                transform =  A.Compose(
                    [   
                        #A.CropAndPad(px = (-5, -5), p = 1),
                        #border_crop,
                        #A.Resize(height = 224, width=224, p = 1),
                        #one_channeler,
                        A.ToFloat(p = 1),
                        neglog(),
                        border_crop,
                        A.Resize(height = 224, width=224, p = 1),
                        A.InvertImg(p =1),
                        #channeler, 
                        mixture_window(keep_original=True, model="kmeans"),
                        A.InvertImg(p = 1),
                        normalizer
                    ],
                    **kwargs
                )
                #transform = A.Compose([neglog_op, resize, normalizer], **kwargs) # neglog_op, windowing,
            elif last_test and not test: # for validation dataset
                 transform =  A.Compose(
                    [
                        A.InvertImg(p =1),
                        mixture_window(keep_original=True, model="kmeans"),
                        A.InvertImg(p = 1),
                        normalizer
                    ],
                    **kwargs
                )
            else:
                 transform =  A.NoOp()
            if len(transform_kwargs['keypoints']) < 2:
                #log.error(transform_kwargs)
                return transform_func
            
            out = transform(**transform_kwargs)
            #log.error(out['image'].max())
            #log.error(out['image'].min())
            #log.error(out['image'].mean())
            out.setdefault("bboxes", [])
            out.setdefault("category_ids", [])
            out.setdefault("masks", [])
            out.setdefault("keypoints", [])
            return out

        return transform_func
    #log.info('building augmentation')

    return build_augmentation

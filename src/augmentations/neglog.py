import numpy as np
import albumentations as A


def neglog_fn(image: np.ndarray, epsilon: float = 0.001) -> np.ndarray:
    """Take the negative log transform of an intensity image.
    Args:
        image (np.ndarray): a single 2D image, or N such images.
        epsilon (float, optional): positive offset from 0 before taking the logarithm.
    Returns:
        np.ndarray: the image or images after a negative log transform, scaled to [0, 1]
    """
    image = np.array(image)
    shape = image.shape
    #assert shape == (224, 224)
    if len(shape) == 2:
        image = image[np.newaxis, :, :]
        
    # shift image to avoid invalid values
    image += image.min(axis=(1, 2), keepdims=True) + epsilon
    #assert(1==2)
    # negative log transform
    image = -np.log(image)
      # linear interpolate to range [0, 1]
    image_min = image.min(axis=(0, 1), keepdims=True)
    image_max = image.max(axis=(0, 1), keepdims=True)
    if np.any(image_max == image_min):
        print(
            f"mapping constant image to 0. This probably indicates the projector is pointed away from the volume."
        )
        print(f'bad and about to assert')
        assert(1==2), 'this should never happen dude u messed up somewhere'
        # TODO(killeen): for multiple images, only fill the bad ones
        image[:] = 0
        if image.shape[0] > 1:
            print("TODO: zeroed all images, even though only one might be bad.")
    else:
        image = (image - image_min) / (image_max - image_min)
    if np.any(np.isnan(image)):
        assert 1==2, f"got NaN values from negative log transform."
    #if image_max == image_min:
    #
    #assert(1==2), 'this should never happen of img min = max dude u messed up somewhere'
    if len(shape) == 2:
        return image[0]
    else:
        return image
    

def neglog(epsilon: float = 0.001) -> A.Lambda:
    """Take the negative log transform of an intensity image.

    Args:
    """

    def f_image(image: np.ndarray, **kwargs) -> np.ndarray:
        return neglog_fn(image, epsilon)

    def f_id(x, **kwargs):
        return x

    return A.Lambda(
        image=f_image,
        mask=f_id,
        keypoints=f_id,
        bboxes=f_id,
        name="neglog",
    )

#  # linear interpolate to range [0, 1]
# image_min = image.min(axis=(1, 2), keepdims=True)
# image_max = image.max(axis=(1, 2), keepdims=True)
# if np.any(image_max == image_min):
#     print(
#         f"mapping constant image to 0. This probably indicates the projector is pointed away from the volume."
#     )
#     # TODO(killeen): for multiple images, only fill the bad ones
#     image[:] = 0
#     if image.shape[0] > 1:
#         print("TODO: zeroed all images, even though only one might be bad.")
# else:
#     image = (image - image_min) / (image_max - image_min)
# if np.any(np.isnan(image)):
#     assert 1==2, f"got NaN values from negative log transform."
# if len(shape) == 2:
#     return image[0]
# else:
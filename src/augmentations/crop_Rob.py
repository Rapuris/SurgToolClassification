import numpy as np
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations.geometric.functional import hflip


def _make_lr_pairs(names):
    name2idx = {n: i for i, n in enumerate(names)}
    pairs = []
    for n, i in name2idx.items():
        if n.startswith("l_"):
            r_name = "r_" + n[2:]            # l_asis -> r_asis, etc.
            if r_name in name2idx:
                pairs.append((i, name2idx[r_name]))
    return pairs

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
# ----------------------------------------------------------------------
#  Horizontal‑flip transform that also swaps semantic L/R keypoints
# ----------------------------------------------------------------------
class SymmetricHorizontalFlip(DualTransform):
    """
    Horizontal flip with symmetric keypoint reordering for anatomical structures.
    
    Args:
        lr_pairs (list): List of tuples representing pairs of keypoints to swap (left-right pairs)
        p (float): Probability of applying the transform. Default: 0.5.
    """
    
    def __init__(self, lr_pairs, always_apply=False, p=0.5):
        super(SymmetricHorizontalFlip, self).__init__(always_apply, p)
        self.lr_pairs = lr_pairs
        
    def apply(self, img, **params):
        """Apply horizontal flip to the image."""
        # Store image dimensions for keypoint transformation
        if img is not None:
            self.height, self.width = img.shape[:2]
        return hflip(img)
    
    def apply_to_mask(self, mask, **params):
        """Apply horizontal flip to the mask."""
        return hflip(mask)
    
    def apply_to_keypoints(self, keypoints, **params):
        """Apply horizontal flip to keypoints with left-right swapping."""
        # Get width from params or use stored width
        width = getattr(self, 'width', params.get('width', 0))
        
        # First, create a copy of the keypoints
        new_keypoints = keypoints.copy()
        
        # Flip the x-coordinate for all keypoints
        for i in range(len(new_keypoints)):
            x, y = new_keypoints[i][:2]  # Unpack the first two elements (x, y)
            # Flip x-coordinate
            new_x = width - x
            
            # Create new keypoint with flipped x
            if len(new_keypoints[i]) > 2:
                # If there are additional values like angle, scale, etc.
                extra_data = new_keypoints[i][2:]
                new_keypoints[i] = (new_x, y) + extra_data
            else:
                new_keypoints[i] = (new_x, y)
        
        # Now swap the left-right pairs
        for left_idx, right_idx in self.lr_pairs:
            if left_idx < len(new_keypoints) and right_idx < len(new_keypoints):
                new_keypoints[left_idx], new_keypoints[right_idx] = new_keypoints[right_idx], new_keypoints[left_idx]
        
        return new_keypoints
    
    def get_transform_init_args_names(self):
        """Returns the parameter names for __init__ method."""
        return ("lr_pairs",)


# Example usage
def create_transforms(p=0.5):
    LR_PAIRS = [
        (0, 6),   # l_asis ↔ r_asis
        (1, 7),   # l_gsn  ↔ r_gsn
        (2, 8),   # l_iof  ↔ r_iof
        (3, 9),   # l_ips  ↔ r_ips
        (4, 10),  # l_mof  ↔ r_mof
        (5, 11)   # l_sps  ↔ r_sps
    ]
    
    return SymmetricHorizontalFlip(lr_pairs=LR_PAIRS, p=p)
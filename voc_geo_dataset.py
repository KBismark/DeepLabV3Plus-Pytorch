
import numpy as np
import cv2
import torch
from scipy.ndimage import distance_transform_edt

from datasets import VOCSegmentation


def _compute_sdt(label_np, num_classes, ignore_label=255):
    h, w = label_np.shape
    sdt_stack = np.zeros((num_classes, h, w), dtype=np.float32)
    valid = label_np != ignore_label

    for c in range(num_classes):
        binary = ((label_np == c) & valid).astype(np.uint8)
        if binary.sum() == 0 or binary.sum() == valid.sum():
            sdt_stack[c] = -1.0 if binary.sum() == 0 else 1.0
            continue
        dist_in = distance_transform_edt(binary)
        dist_out = distance_transform_edt(1 - binary)
        sdf = dist_in - dist_out
        max_val = max(abs(sdf.max()), abs(sdf.min())) + 1e-6
        sdt_stack[c] = (sdf / max_val).astype(np.float32)

    return sdt_stack


def _compute_boundary(label_np, ignore_label=255, dilate_size=2):
    lm = label_np.astype(np.int32).copy()
    lm[lm == ignore_label] = -1

    boundary = np.zeros_like(lm, dtype=np.uint8)
    boundary[:-1, :] |= (lm[:-1, :] != lm[1:, :])
    boundary[1:, :] |= (lm[:-1, :] != lm[1:, :])
    boundary[:, :-1] |= (lm[:, :-1] != lm[:, 1:])
    boundary[:, 1:] |= (lm[:, :-1] != lm[:, 1:])
    boundary[lm == -1] = 0

    if dilate_size > 1:
        kernel = np.ones((dilate_size, dilate_size), np.uint8)
        boundary = cv2.dilate(boundary, kernel, iterations=1)

    return boundary.astype(np.float32)


class VOCSegmentationGeo(VOCSegmentation):
    """
    Extended VOCSegmentation to also return distance + boundary maps.
    """
    def __init__(self, *args, num_classes=21, ignore_label=255, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.ignore_label = ignore_label

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)   # label is already a transformed tensor (H,W), long

        label_np = label.numpy() if torch.is_tensor(label) else np.array(label)

        distance = _compute_sdt(label_np, self.num_classes, self.ignore_label)
        boundary = _compute_boundary(label_np, self.ignore_label)

        distance = torch.from_numpy(distance).float()
        boundary = torch.from_numpy(boundary).float().unsqueeze(0)

        return image, label, distance, boundary


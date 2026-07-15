import numpy as np
import cv2

# Fixed camera parameters (OpenCV model)
K_DEFAULT = np.array([
    [802.319, 0.0, 668.286],
    [0.0, 801.885, 547.733],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

D_DEFAULT = np.array([-0.42234, 0.10654, 0.0, 0.0], dtype=np.float32)


def build_undistort_map(width, height, alpha=0.0, new_size=None, use_roi=False, K=K_DEFAULT, D=D_DEFAULT):
    """
    Build undistortion maps using OpenCV.

    Args:
        width, height: input image size
        alpha: 0 (crop black) or 1 (keep all pixels)
        new_size: (w, h) for output size, default input size
        use_roi: if True, output will be cropped to ROI
        K, D: camera parameters

    Returns:
        map1, map2, new_K, roi, out_size
    """
    if new_size is None:
        new_size = (width, height)

    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (width, height), alpha, new_size)
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, new_size, cv2.CV_32FC1)

    if use_roi:
        x, y, w, h = roi
        out_size = (w, h)
    else:
        out_size = new_size
    return map1, map2, new_K, roi, out_size


def remap_depth_with_mask(depth_u16, map1, map2, interpolation="linear", mask_thresh=0.5, use_roi=False, roi=None):
    """
    Remap depth with joint mask to avoid invalid propagation.

    Args:
        depth_u16: uint16 depth map
        map1, map2: undistortion maps
        interpolation: "linear" or "nearest"
        mask_thresh: threshold on remapped mask
        use_roi: crop to roi
        roi: (x, y, w, h)

    Returns:
        depth_remap_u16
    """
    if interpolation == "nearest":
        interp = cv2.INTER_NEAREST
    else:
        interp = cv2.INTER_LINEAR

    mask = (depth_u16 > 0).astype(np.float32)
    depth_f = depth_u16.astype(np.float32)

    depth_remap = cv2.remap(depth_f, map1, map2, interpolation=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_remap = cv2.remap(mask, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    valid = mask_remap > mask_thresh
    depth_remap[~valid] = 0

    if use_roi and roi is not None:
        x, y, w, h = roi
        depth_remap = depth_remap[y:y + h, x:x + w]

    depth_remap = np.clip(depth_remap, 0, 65535)
    return depth_remap.astype(np.uint16)

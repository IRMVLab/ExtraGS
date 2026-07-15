from pathlib import Path
import numpy as np
import imageio.v3 as iio


def read_tiff_u16(path):
    """Read TIFF as uint16."""
    arr = iio.imread(path)
    if arr.dtype != np.uint16:
        arr = arr.astype(np.uint16)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def write_png_u16(path, arr_u16):
    """Write uint16 PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, arr_u16.astype(np.uint16))


def ensure_uint16(arr):
    if arr.dtype != np.uint16:
        arr = np.clip(arr, 0, 65535).astype(np.uint16)
    return arr


def list_tiff_files(input_dir, pattern="*_depth.tiff"):
    return sorted(Path(input_dir).glob(pattern))

#!/usr/bin/env python3
import argparse
import numpy as np
from io_utils import read_tiff_u16, write_png_u16
from camera import build_undistort_map, remap_depth_with_mask, K_DEFAULT, D_DEFAULT


def encode_invdepth_u16(depth_mm_u16, z_min_mm=0.5, z_max_mm=100.0, invalid_to_zero=True):
    depth_mm = depth_mm_u16.astype(np.float32) / 65535.0 * 100.0
    valid = depth_mm > 0

    inv_min = 1.0 / z_max_mm
    inv_max = 1.0 / z_min_mm

    inv = np.zeros_like(depth_mm, dtype=np.float32)
    inv[valid] = 1.0 / depth_mm[valid]
    inv = np.clip(inv, inv_min, inv_max)

    inv_u16 = np.zeros_like(depth_mm_u16, dtype=np.uint16)
    inv_u16[valid] = np.round((inv[valid] - inv_min) / (inv_max - inv_min) * 65535.0).astype(np.uint16)

    if not invalid_to_zero:
        inv_u16[~valid] = 0
    return inv_u16


def main():
    parser = argparse.ArgumentParser(description="Undistort depth TIFF and export linear + inverse depth PNGs.")
    parser.add_argument("--input", required=True, help="Input TIFF depth path")
    parser.add_argument("--output-linear", required=True, help="Output undistorted linear depth PNG (uint16)")
    parser.add_argument("--output-inv", required=True, help="Output undistorted inverse depth PNG (uint16)")
    parser.add_argument("--alpha", type=float, default=0.0, help="Alpha for new camera matrix")
    parser.add_argument("--interp", choices=["linear", "nearest"], default="linear", help="Interpolation method")
    parser.add_argument("--use-roi", action="store_true", help="Crop output to ROI")
    parser.add_argument("--mask-thresh", type=float, default=0.5, help="Mask threshold after remap")
    parser.add_argument("--z-min", type=float, default=0.5, help="Min depth (mm) for inverse depth encoding")
    parser.add_argument("--z-max", type=float, default=100.0, help="Max depth (mm) for inverse depth encoding")
    args = parser.parse_args()

    depth_u16 = read_tiff_u16(args.input)
    h, w = depth_u16.shape[:2]

    map1, map2, new_K, roi, out_size = build_undistort_map(
        w, h, alpha=args.alpha, new_size=(w, h), use_roi=args.use_roi, K=K_DEFAULT, D=D_DEFAULT
    )

    depth_ud = remap_depth_with_mask(
        depth_u16, map1, map2, interpolation=args.interp, mask_thresh=args.mask_thresh, use_roi=args.use_roi, roi=roi
    )

    inv_u16 = encode_invdepth_u16(depth_ud, z_min_mm=args.z_min, z_max_mm=args.z_max)

    write_png_u16(args.output_linear, depth_ud)
    write_png_u16(args.output_inv, inv_u16)

    print(f"Saved linear depth: {args.output_linear}")
    print(f"Saved inverse depth: {args.output_inv}")


if __name__ == "__main__":
    main()

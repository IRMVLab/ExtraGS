#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
from io_utils import list_tiff_files, read_tiff_u16, write_png_u16
from depth_encoding import encode_invdepth_u16
from camera import build_undistort_map, remap_depth_with_mask, K_DEFAULT, D_DEFAULT


def collect_images(images_dir):
    images_dir = Path(images_dir)
    if not images_dir.exists():
        return []
    return sorted([p.name for p in images_dir.glob("*.png")])


def build_name_mapping(images_dir, split, sample_interval=10):
    names = collect_images(images_dir)
    if not names:
        return None, None
    if split == "train":
        train_names = [n for n in names if "frame_1_" not in n]
        return train_names, None
    if split == "test":
        test_names = [n for n in names if "frame_1_" in n]
        return None, test_names
    train_names = [n for n in names if "frame_1_" not in n]
    test_names = [n for n in names if "frame_1_" in n]
    return train_names, test_names


def main():
    parser = argparse.ArgumentParser(description="Batch convert C3VD depth TIFF to undistorted PNG + inverse depth.")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument("--train-depth-dir", required=True)
    parser.add_argument("--test-depth-dir", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-linear-dir", required=True)
    parser.add_argument("--output-inv-dir", required=True)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--interp", choices=["linear", "nearest"], default="linear")
    parser.add_argument("--use-roi", action="store_true")
    parser.add_argument("--mask-thresh", type=float, default=0.5)
    parser.add_argument("--z-min", type=float, default=0.5)
    parser.add_argument("--z-max", type=float, default=100.0)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train_names, test_names = build_name_mapping(args.images_dir, args.split, sample_interval=args.sample_interval)

    def process_split(split_name, depth_dir, output_linear_dir, output_inv_dir, names_list):
        tiffs = list_tiff_files(depth_dir)
        if split_name == "test":
            tiffs = [tiffs[i] for i in range(0, len(tiffs), args.sample_interval)]
        if names_list is None:
            if split_name == "train":
                names_list = [f"{i:04d}.png" for i in range(len(tiffs))]
            else:
                names_list = [f"frame_1_{i:04d}.png" for i in range(len(tiffs))]
        if len(names_list) != len(tiffs):
            raise RuntimeError(f"Name count mismatch: {len(names_list)} vs {len(tiffs)} for {split_name}")

        for idx, tiff_path in enumerate(tiffs):
            out_name = names_list[idx]
            out_linear = Path(output_linear_dir) / out_name
            out_inv = Path(output_inv_dir) / out_name

            if args.dry_run:
                print(f"[{split_name}] {tiff_path.name} -> {out_name}")
                continue

            depth_u16 = read_tiff_u16(str(tiff_path))
            h, w = depth_u16.shape[:2]
            map1, map2, new_K, roi, out_size = build_undistort_map(
                w, h, alpha=args.alpha, new_size=(w, h), use_roi=args.use_roi, K=K_DEFAULT, D=D_DEFAULT
            )
            depth_ud = remap_depth_with_mask(
                depth_u16, map1, map2, interpolation=args.interp, mask_thresh=args.mask_thresh, use_roi=args.use_roi, roi=roi
            )
            inv_u16 = encode_invdepth_u16(depth_ud, z_min_mm=args.z_min, z_max_mm=args.z_max)

            write_png_u16(out_linear, depth_ud)
            write_png_u16(out_inv, inv_u16)

    if args.split in ["train", "both"]:
        process_split("train", args.train_depth_dir, args.output_linear_dir, args.output_inv_dir, train_names)
    if args.split in ["test", "both"]:
        process_split("test", args.test_depth_dir, args.output_linear_dir, args.output_inv_dir, test_names)


if __name__ == "__main__":
    main()

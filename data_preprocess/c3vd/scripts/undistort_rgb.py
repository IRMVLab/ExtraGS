#!/usr/bin/env python3
"""
Step 1: Undistort RGB Images
将畸变的 RGB 图像进行去畸变处理
"""

import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import argparse

def undistort_images(input_dir, output_dir, K, D, target_size=(1350, 1080)):
    """
    去畸变处理
    
    Args:
        input_dir: 输入图像目录
        output_dir: 输出图像目录
        K: 相机内参矩阵 (3x3)
        D: 畸变系数 (k1, k2, p1, p2)
        target_size: 目标图像尺寸 (width, height)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = sorted(Path(input_dir).glob("*.png"))
    
    print(f"找到 {len(image_files)} 张图像")
    print(f"相机内参:\n{K}")
    print(f"畸变系数: {D}")
    print(f"目标分辨率: {target_size}")
    
    # 预计算去畸变映射
    # 使用原始相机矩阵作为新相机矩阵（保持相同的内参）
    new_K, roi = cv2.getOptimalNewCameraMatrix(
        K, D, target_size, alpha=0, newImgSize=target_size
    )
    
    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, None, new_K, target_size, cv2.CV_16SC2
    )
    
    print(f"\n新相机内参:\n{new_K}")
    print(f"ROI: {roi}")
    
    # 处理每张图像
    for img_file in tqdm(image_files, desc="去畸变处理"):
        # 读取图像
        img = cv2.imread(str(img_file))
        
        if img is None:
            print(f"警告: 无法读取 {img_file}")
            continue
        
        # 去畸变
        undistorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
        
        # 保存
        output_path = os.path.join(output_dir, img_file.name)
        cv2.imwrite(output_path, undistorted)
    
    # 保存新的相机内参
    camera_params_file = os.path.join(output_dir, "undistorted_camera_params.txt")
    with open(camera_params_file, 'w') as f:
        f.write("# Undistorted Camera Parameters\n")
        f.write(f"# Original K:\n# {K.tolist()}\n")
        f.write(f"# Original D: {D.tolist()}\n\n")
        f.write(f"# New K (after undistortion):\n")
        f.write(f"fx = {new_K[0, 0]}\n")
        f.write(f"fy = {new_K[1, 1]}\n")
        f.write(f"cx = {new_K[0, 2]}\n")
        f.write(f"cy = {new_K[1, 2]}\n")
        f.write(f"width = {target_size[0]}\n")
        f.write(f"height = {target_size[1]}\n")
        f.write(f"# ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}\n")
    
    print(f"\n✅ 完成！去畸变图像保存至: {output_dir}")
    print(f"✅ 相机参数保存至: {camera_params_file}")
    
    return new_K, roi


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="去畸变 RGB 图像")
    parser.add_argument("--input_v1", type=str, required=True, help="v1 RGB 目录")
    parser.add_argument("--input_v2", type=str, required=True, help="v2 RGB 目录")
    parser.add_argument("--output_v1", type=str, required=True, help="v1 输出目录")
    parser.add_argument("--output_v2", type=str, required=True, help="v2 输出目录")
    parser.add_argument("--camera_params_output", type=str, required=True, 
                       help="新相机参数输出文件")
    args = parser.parse_args()
    
    # 相机参数
    K = np.array([[802.319, 0, 668.286],
                  [0, 801.885, 547.733],
                  [0, 0, 1]], dtype=np.float32)
    D = np.array([-0.42234, 0.10654, 0, 0], dtype=np.float32)
    target_size = (1350, 1080)  # width x height
    
    print("=" * 60)
    print("步骤 1: RGB 图像去畸变")
    print("=" * 60)
    
    # 处理 v1
    print("\n处理 v1 (训练集)...")
    new_K_v1, roi_v1 = undistort_images(args.input_v1, args.output_v1, K, D, target_size)
    
    # 处理 v2
    print("\n处理 v2 (测试集)...")
    new_K_v2, roi_v2 = undistort_images(args.input_v2, args.output_v2, K, D, target_size)
    
    # 保存新相机参数到文件
    print(f"\n保存相机参数到: {args.camera_params_output}")
    with open(args.camera_params_output, 'w') as f:
        f.write(f"fx_new = {new_K_v1[0, 0]}\n")
        f.write(f"fy_new = {new_K_v1[1, 1]}\n")
        f.write(f"cx_new = {new_K_v1[0, 2]}\n")
        f.write(f"cy_new = {new_K_v1[1, 2]}\n")
    
    print("\n" + "=" * 60)
    print("✅ 步骤 1 完成")
    print("=" * 60)

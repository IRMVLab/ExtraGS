#!/usr/bin/env python3
"""
Step 5: Generate Multi-Resolution Images
生成多分辨率图像 (images_2, images_4, images_8)
"""

import cv2
import os
from pathlib import Path
from tqdm import tqdm
import argparse

def generate_downsampled_images(input_dir, output_dirs, scales=[2, 4, 8]):
    """
    生成下采样图像
    
    Args:
        input_dir: 原始图像目录 (images/)
        output_dirs: 输出目录列表 [images_2, images_4, images_8]
        scales: 下采样比例 [2, 4, 8]
    """
    # 获取所有图像
    image_files = sorted(Path(input_dir).glob("*.png"))
    print(f"找到 {len(image_files)} 张图像")
    
    for scale, output_dir in zip(scales, output_dirs):
        print(f"\n生成 1/{scale} 分辨率图像...")
        os.makedirs(output_dir, exist_ok=True)
        
        for img_file in tqdm(image_files, desc=f"Scale 1/{scale}"):
            # 读取原始图像
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"警告: 无法读取 {img_file}")
                continue
            
            # 计算新尺寸
            h, w = img.shape[:2]
            new_h, new_w = h // scale, w // scale
            
            # 下采样 (使用 INTER_AREA 获得最佳质量)
            img_downsampled = cv2.resize(img, (new_w, new_h), 
                                        interpolation=cv2.INTER_AREA)
            
            # 保存
            output_path = os.path.join(output_dir, img_file.name)
            cv2.imwrite(output_path, img_downsampled)
        
        print(f"  ✅ 完成 {output_dir} ({new_w}x{new_h})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成多分辨率图像")
    parser.add_argument("--input_dir", type=str, required=True, 
                       help="原始图像目录")
    parser.add_argument("--output_2", type=str, required=True, 
                       help="1/2 分辨率输出目录")
    parser.add_argument("--output_4", type=str, required=True, 
                       help="1/4 分辨率输出目录")
    parser.add_argument("--output_8", type=str, required=True, 
                       help="1/8 分辨率输出目录")
    args = parser.parse_args()
    
    print("=" * 60)
    print("步骤 5: 生成多分辨率图像")
    print("=" * 60)
    
    generate_downsampled_images(
        input_dir=args.input_dir,
        output_dirs=[args.output_2, args.output_4, args.output_8],
        scales=[2, 4, 8]
    )
    
    print("\n" + "=" * 60)
    print("✅ 步骤 5 完成")
    print("=" * 60)

#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
ExtraGS download helper

Large assets are not stored in this repository.

Please download and place files as follows:

1. Diffusion enhancer checkpoint
   diffusion/ckpt/enhancer.ckpt

2. Depth-Anything-V2 checkpoint
   Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth

3. Datasets
   dataset/nerfbusters-dataset/<scene>/
   dataset/wild-explore/<scene>/

Update this script with the official release URLs after the project data and
checkpoints are public.
EOF

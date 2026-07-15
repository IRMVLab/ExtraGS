#!/bin/bash
# 检查并安装依赖

echo "检查 Python 依赖..."

# 检查 OpenCV
python -c "import cv2" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ OpenCV 未安装"
    echo "   安装中: pip install opencv-python"
    pip install opencv-python
else
    echo "✅ OpenCV 已安装"
fi

# 检查 NumPy
python -c "import numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ NumPy 未安装"
    echo "   安装中: pip install numpy"
    pip install numpy
else
    echo "✅ NumPy 已安装"
fi

# 检查 tqdm
python -c "import tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ tqdm 未安装"
    echo "   安装中: pip install tqdm"
    pip install tqdm
else
    echo "✅ tqdm 已安装"
fi

echo ""
echo "✅ 所有依赖已就绪"

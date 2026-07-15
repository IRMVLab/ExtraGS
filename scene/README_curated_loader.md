# curated 数据集读取流程（ExploreGS）

> 本文记录 ExploreGS 如何读取 Nerfbusters curated 格式数据集（含深度）。

## 入口函数
- 入口是 `readNerfbustersInfo_with_depth()`，用于读取带深度的 curated 数据集。
  - 代码位置：[scene/dataset_readers.py](dataset_readers.py#L436-L520)

## 相机与位姿读取（COLMAP）
1. 优先读取二进制：
   - `colmap/sparse/0/images.bin`
   - `colmap/sparse/0/cameras.bin`
2. 失败则回退到文本：
   - `colmap/sparse/0/images.txt`
   - `colmap/sparse/0/cameras.txt`

对应实现：[scene/dataset_readers.py](dataset_readers.py#L444-L456)

## 图像分辨率选择
- 默认读取 `images_2`（`downsample_factor=2`），只有 scene 名为 `century` 时读取 `images`。

对应实现：[scene/dataset_readers.py](dataset_readers.py#L468-L474)

## 深度参数加载
- 深度参数来自：`model_train/sparse/depth_params.json`
- 若 `depths` 参数非空，会加载并计算所有尺度的中位数 `med_scale`，写回每张图。

对应实现：[scene/dataset_readers.py](dataset_readers.py#L459-L467)

## 图像与深度路径绑定
- 对每个相机条目，会拼接：
  - 图像：`images_2/<image_name>`
  - 深度：`<depths>/<image_name>.png`
- 并从 `depth_params.json` 中按图像名索引对应 `scale/offset`。

对应实现：[scene/dataset_readers.py](dataset_readers.py#L124-L189)

## 训练/测试划分逻辑
- 以文件名是否包含 `frame_1_` 为测试集：
  - 包含 `frame_1_` → 测试集
  - 其余 → 训练集
- `mode=swap` 时会反转划分。

对应实现：[scene/dataset_readers.py](dataset_readers.py#L481-L503)

## 深度读取与归一化
- 深度图使用 `cv2.imread(..., -1)` 读取，随后除以 `2**16` 转为 `[0, 1]` 浮点。
- 后续在 `Camera` 中会乘上 `depth_params.scale` 并加 `offset`（用于还原尺度）。

读取实现：[utils/camera_utils.py](../utils/camera_utils.py#L21-L36)

## 深度的进一步处理（camera_utils + Camera）
1. **读取阶段（camera_utils）**
  - `cv2.imread(..., -1)` 读取原始深度 PNG。
  - 转成 `float32` 后除以 `2**16`，得到 $[0, 1)$ 的归一化值。

  对应实现：[utils/camera_utils.py](../utils/camera_utils.py#L21-L36)

2. **缩放与偏移（Camera）**
  - 若 `depth_params` 存在：
    - 先用中位数尺度 `med_scale` 做可靠性判断；
    - 再执行：`invdepth = invdepth * scale + offset`。

  对应实现：[scene/cameras.py](cameras.py#L78-L91)

3. **文件名索引规则**
  - `depth_params.json` 的 key 必须与图像名（不含扩展名）一致。
  - 深度文件名：`<image_name>.png`。

  对应实现：[scene/dataset_readers.py](dataset_readers.py#L170-L184)

## depth scale 的计算（make_depth_scale.py）
1. **对齐 COLMAP 真实深度与单目逆深度**
   - 取 COLMAP 点云与相机位姿，得到相机坐标系下的深度 $z$，并转为逆深度：
     $$d_{colmap}^{-1} = 1 / z$$
   - 读取深度图（被当作“逆深度”）并归一化到 $[0,1)$：
     $$d_{mono}^{-1} = \text{imread} / 2^{16}$$

2. **在图像平面采样对应像素**
   - 将 COLMAP 的像素坐标映射到深度图分辨率后采样。

3. **用鲁棒统计估计 scale/offset**
   - 以 **中位数** 作为中心，以 **平均绝对偏差** 作为尺度：
     $$t = \text{median}(d^{-1}), \quad s = \text{mean}(|d^{-1}-t|)$$
   - 计算：
     $$\text{scale} = s_{colmap} / s_{mono}$$
     $$\text{offset} = t_{colmap} - t_{mono} \cdot \text{scale}$$

对应实现：[utils/make_depth_scale.py](../utils/make_depth_scale.py#L16-L83)

## 点云读取
- 读取 `model_train/sparse/points3D.bin`（或 `model_test/sparse/points3D.bin`）。
- 读取后会生成 `points3D.ply` 并载入为点云。

对应实现：[scene/dataset_readers.py](dataset_readers.py#L505-L517)

## curated 数据集关键文件清单
- 相机与位姿：
  - `colmap/sparse/0/{cameras.bin, images.bin}` 或 `.txt`
- 点云：
  - `model_train/sparse/points3D.bin`
- 图像：
  - `images_2/*.png`（默认）
- 深度：
  - `<depths>/*.png`（由训练参数 `--depths` 指定目录）
- 深度参数：
  - `model_train/sparse/depth_params.json`

## 常见问题提示
- 如果训练集为 0，检查：
  - `images_2/` 中是否有非 `frame_1_` 的图像
  - `images.txt` 中的文件名是否与图像一致
- 如果 `depths_params not found`，检查：
  - `depth_params.json` 的 key 是否和图像名（不含扩展名）一致
  - `depths` 目录是否包含对应 `.png`

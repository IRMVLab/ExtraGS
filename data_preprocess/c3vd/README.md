# C3VD Preprocessing

Our experiments use the public [C3VDv2](https://durrlab.github.io/C3VDv2/) colonoscopy dataset. Download the raw sequences from the [Johns Hopkins Research Data Repository](https://archive.data.jhu.edu/dataset.xhtml?persistentId=doi%3A10.7281%2FT1%2FJC64MK), then convert them with the pipeline below.

Our C3VD experiments do **not** use the WildExplore-style loader. C3VD sequences are converted into the Nerfbusters curated-style layout and are trained with:

```text
--load Nerfbusters
```

The conversion uses one C3VD sequence as the training trajectory and another sequence as the held-out test trajectory.

To reproduce our released C3VD results, train with the swapped Nerfbusters split (`swap=1` in the helper scripts). With `swap=1`, frames named `frame_1_*` are treated as training views, the remaining frames are used for evaluation, and the loader initializes from `model_test/sparse`.

## Raw C3VD Input

Each raw C3VD sequence is expected to contain:

```text
<C3VD_SEQUENCE>/
├── rgb/
├── depth/
└── pose.txt
```

For example:

```text
dataset/c3vd_dataset/c1_a_t1_v1/
dataset/c3vd_dataset/c1_a_t1_v2/
```

## Output Layout

The preprocessing script writes a curated-style scene:

```text
dataset/nerfbusters-dataset/<scene>/
├── images/                         # train + sampled test frames
├── images_train/                   # training frames only
├── images_test/                    # sampled test frames, named frame_1_XXXX.png
├── images_2/
├── images_4/
├── images_8/
├── depths/                         # inverse-depth uint16 PNGs
├── depths_linear/                  # linear depth PNGs
├── colmap/sparse/0/
├── model_train/sparse/
│   └── depth_params.json
├── model_test/sparse/
└── batch_captions_result_<scene>.jsonl
```

This is the layout consumed by `scene/dataset_readers.py::readNerfbustersInfo_with_depth()`.

## Tool Structure

```text
data_preprocess/c3vd/
├── prepare_c3vd_curated.sh          # one-command entry point
├── caption_scene.py                 # writes batch_captions_result_<scene>.jsonl
├── compute_depth_scale.py           # writes model_train/sparse/depth_params.json
├── scripts/
│   ├── prepare_rgb_images.sh        # undistort, split, and create multires images
│   ├── undistort_rgb.py
│   └── create_multires_images.py
├── depth/
│   ├── prepare_depths.sh            # convert raw C3VD TIFF depth maps
│   ├── convert_c3vd_depth.py
│   ├── depth_encoding.py
│   ├── camera.py
│   └── io_utils.py
└── colmap/
    └── build_and_split_colmap.sh    # build COLMAP and split train/test models
```

Most users should call only `prepare_c3vd_curated.sh`.

## One-command Pipeline

From the repository root:

First edit the OpenAI/proxy settings near the top of `prepare_c3vd_curated.sh` if you want the pipeline to generate the diffusion caption automatically:

```bash
OPENAI_API_KEY="sk-..."
OPENAI_BASE_URL="https://your-relay.example.com/v1"  # optional
OPENAI_MODEL="gpt-4o"
```

Then run:

```bash
bash data_preprocess/c3vd/prepare_c3vd_curated.sh \
  --train-root dataset/c3vd_dataset/c1_a_t1_v1 \
  --test-root dataset/c3vd_dataset/c1_a_t1_v2 \
  --out-root dataset/nerfbusters-dataset/curated_c1_a_t1 \
  --sample-interval 10 \
  --clean-temp 1
```

If you want to prepare the dataset first and generate captions later:

```bash
bash data_preprocess/c3vd/prepare_c3vd_curated.sh \
  --train-root dataset/c3vd_dataset/c1_a_t1_v1 \
  --test-root dataset/c3vd_dataset/c1_a_t1_v2 \
  --out-root dataset/nerfbusters-dataset/curated_c1_a_t1 \
  --sample-interval 10 \
  --clean-temp 1 \
  --skip-gpt 1
```

Then generate the caption:

```bash
python data_preprocess/c3vd/caption_scene.py \
  --source_path dataset/nerfbusters-dataset/curated_c1_a_t1
```

## What The Pipeline Does

1. Undistorts C3VD RGB frames.
2. Builds `images/`, `images_train/`, `images_test/`, and multi-resolution image folders.
3. Converts C3VD TIFF depth maps into inverse-depth PNGs under `depths/`.
4. Runs COLMAP over the combined image folder.
5. Splits the COLMAP model into `model_train` and `model_test` using the `frame_1_` test-frame prefix.
6. Computes `model_train/sparse/depth_params.json` for inverse-depth regularization.
7. Generates `batch_captions_result_<scene>.jsonl` for the diffusion enhancer unless `--skip-gpt 1` is used.

## Training After Preprocessing

Run Stage 1 with the Nerfbusters loader:

```bash
bash scripts/run_stage1.sh curated_c1_a_t1 0 depthreg local 1
```

Run Stage 2 with the ExtraGS settings:

```bash
bash scripts/run_stage2.sh \
  curated_c1_a_t1 0 \
  enhancer default full visibility_mask ours \
  depthreg 30000 local extrags 1
```

For C3VD, use the standard `scripts/run_stage1.sh`, `scripts/run_stage2.sh`, `render.py`, and `metrics.py` entry points.

## Notes

- `--sample-interval` controls how densely the test sequence is sampled.
- `COLMAP` must be available through `--colmap-bin` or on `PATH`.
- The GPT caption step reads `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL` from the configuration block at the top of `prepare_c3vd_curated.sh`. `OPENAI_BASE_URL` is useful for OpenAI-compatible relay/proxy endpoints.
- The output scene name is `basename(--out-root)`, and Stage 2 expects `batch_captions_result_<scene>.jsonl` in that same directory.

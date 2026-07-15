# Data Preprocessing Utilities

This directory contains helper scripts for preparing text prompts and captions used by the diffusion enhancer.

## C3VD Conversion

Raw C3VD sequences are converted into Nerfbusters curated-style scenes under:

```text
dataset/nerfbusters-dataset/<scene>/
```

Use:

Edit `data_preprocess/c3vd/prepare_c3vd_curated.sh` first if you need an OpenAI API key, relay endpoint, or custom model for automatic caption generation.

```bash
bash data_preprocess/c3vd/prepare_c3vd_curated.sh \
  --train-root dataset/c3vd_dataset/c1_a_t1_v1 \
  --test-root dataset/c3vd_dataset/c1_a_t1_v2 \
  --out-root dataset/nerfbusters-dataset/curated_c1_a_t1
```

Then train with `scripts/run_stage1.sh` / `scripts/run_stage2.sh` and `--load Nerfbusters`. See `data_preprocess/c3vd/README.md` for the complete C3VD workflow.

## Scene Caption Generation

`gpt_batch_nerfbusters_scene_caption.py` generates one scene-level caption from a Nerfbusters-style image folder. The caption is used as the text condition for `3DGS_Enhancer_Extra` during Stage 2.

### Why This File Is Needed

During Stage 2, `diffusion/enhance_utils.py` constructs `EnhanceDiffusionPriorPipeline` and reads:

```text
<source_path>/batch_captions_result_<scene>.jsonl
```

where `<scene>` is `os.path.basename(args.source_path)`. If this file is missing, diffusion enhancement will fail before virtual view generation.

The JSON object must contain the standard OpenAI Batch chat completion response structure, and the caption is read from:

```text
response.body.choices[0].message.content
```

### Expected Input

Default layout:

```text
dataset/nerfbusters-dataset/<scene>/
├── images_train/
│   ├── frame_00001.png
│   ├── frame_00002.png
│   └── ...
└── images_test/
```

The script also works with other image folders if you pass `--images_subdir`, for example `images`, `images_2`, or `rgb`.

Supported image extensions:

```text
.png, .jpg, .jpeg
```

### Dependencies

```bash
pip install openai opencv-python
```

You also need an OpenAI API key:

```bash
export OPENAI_API_KEY=<your_api_key>
```

For an OpenAI-compatible proxy or custom endpoint:

```bash
export OPENAI_BASE_URL=<your_base_url>
```

### Basic Usage

From the repository root:

```bash
python data_preprocess/gpt_batch_nerfbusters_scene_caption.py \
  --scene_dir dataset/nerfbusters-dataset/<scene> \
  --scene_name <scene>
```

This uses the defaults:

```text
--images_subdir images_train
--num_frames 15
--scale 0.5
--model gpt-4o
--max_tokens 70
--poll_interval 10.0
```

### Common Variants

If your scene stores images in `images_2`:

```bash
python data_preprocess/gpt_batch_nerfbusters_scene_caption.py \
  --scene_dir dataset/nerfbusters-dataset/<scene> \
  --scene_name <scene> \
  --images_subdir images_2
```

If your images are high resolution and the request payload is too large, reduce the scale or number of frames:

```bash
python data_preprocess/gpt_batch_nerfbusters_scene_caption.py \
  --scene_dir dataset/nerfbusters-dataset/<scene> \
  --scene_name <scene> \
  --images_subdir images_train \
  --num_frames 8 \
  --scale 0.25
```

### Outputs

The script writes two files into `--scene_dir`:

```text
batchinput_<scene>.jsonl
batch_captions_result_<scene>.jsonl
```

`batchinput_<scene>.jsonl` is the request uploaded to OpenAI Batch.

`batch_captions_result_<scene>.jsonl` is the file consumed by Stage 2. Keep it inside the scene directory.

### What The Script Does

1. Finds images under `<scene_dir>/<images_subdir>`.
2. Uniformly samples `--num_frames` frames.
3. Resizes each sampled frame by `--scale`.
4. Encodes frames as base64 PNG data URLs.
5. Creates one `/v1/chat/completions` batch request.
6. Uploads the request with `purpose="batch"`.
7. Creates an OpenAI Batch job with a `24h` completion window.
8. Polls until the batch is `completed`, `failed`, or `expired`.
9. Downloads the completed JSONL result.

### Troubleshooting

- `OPENAI_API_KEY is not set`: export `OPENAI_API_KEY` before running the script.
- `images dir not found`: check `--scene_dir` and `--images_subdir`.
- `No images found`: only `.png`, `.jpg`, and `.jpeg` are collected.
- Batch ended with `failed` or `expired`: inspect the batch error file in your OpenAI dashboard or API logs.
- Stage 2 cannot find the caption: make sure the result file name is exactly `batch_captions_result_<scene>.jsonl`, where `<scene>` matches the basename of `--source_path`.

### Notes

The default prompt asks the model to keep the caption under 70 tokens because the caption is used as input to a CLIP text encoder. If you change `--max_tokens`, keep the text concise and scene-level rather than frame-specific.

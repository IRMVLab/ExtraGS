"""
No-Batch version (API does NOT support file upload)
- Sample 15 images from a specified folder (uniformly)
- Downsample images (scale=0.5)
- Call chat.completions synchronously
- Save caption result to JSONL-like single JSON object that is compatible with
  diffusion/enhance_utils.py:
  ['response']['body']['choices'][0]['message']['content']
"""

import os
import glob
import cv2
import base64
import json
import time
import argparse
from openai import OpenAI


SYSTEM_MESSAGE_DICT = {
    "role": "system",
    "content": (
        "You are a helpful assistant that can caption videos. "
        "This caption will be used as input for CLIP text encoder. "
        "Please describe the video frames in detail while keeping the max tokens under 70."
    ),
}


def create_prompt_message(frames):
    content = [{"type": "text", "text": "Please caption the following video frames:"}]
    for frame in frames:
        ok, buf = cv2.imencode(".png", frame)
        if not ok:
            raise RuntimeError("cv2.imencode('.png', frame) failed")
        b64 = base64.b64encode(buf).decode("utf-8")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )
    return [SYSTEM_MESSAGE_DICT, {"role": "user", "content": content}]


def sample_uniform(image_paths, num_samples=15):
    if len(image_paths) <= num_samples:
        return image_paths

    indices = [
        round(i * (len(image_paths) - 1) / (num_samples - 1))
        for i in range(num_samples)
    ]
    indices = [min(max(int(i), 0), len(image_paths) - 1) for i in indices]
    return [image_paths[i] for i in indices]


def collect_images(train_image_dir):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(train_image_dir, ext)))
    return sorted(image_paths)


def load_and_resize(paths, scale=0.5):
    frames = []
    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        if scale != 1.0:
            frame = cv2.resize(frame, dsize=(0, 0), fx=scale, fy=scale)
        frames.append(frame)
    return frames


def convert_to_enhancer_compatible_json(scene, caption):
    # Keep fields expected by diffusion/enhance_utils.py
    return {
        "custom_id": f"nobatch_{scene}",
        "response": {
            "status_code": 200,
            "request_id": "",
            "body": {
                "id": "chatcmpl_nobatch",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": caption},
                        "finish_reason": "stop",
                    }
                ],
            },
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_path",
        default="",
        help="Scene directory that contains images_train/ (e.g., ./dataset/nerfbusters-dataset/curated)",
    )
    parser.add_argument("--dataset_root", default="./dataset/nerfbusters-dataset")
    parser.add_argument("--scene", default="curated")
    parser.add_argument("--num_samples", type=int, default=15)
    parser.add_argument("--resize_scale", type=float, default=0.5)
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "chatgpt-4o-latest"),
        help="Chat Completions model name",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output file path (default: <source_path>/batch_captions_result_<scene>.jsonl)",
    )
    args = parser.parse_args()

    # Client settings
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is empty. Please export it before running.")

    client = OpenAI(base_url=base_url, api_key=api_key) if base_url else OpenAI(api_key=api_key)

    if args.source_path:
        source_paths = [args.source_path]
    else:
        source_paths = [os.path.join(args.dataset_root, args.scene)]

    for source_path in source_paths:
        scene_name = os.path.basename(os.path.abspath(source_path))
        train_image_dir = os.path.join(source_path, "images_train")

        image_paths = collect_images(train_image_dir)
        if len(image_paths) == 0:
            raise FileNotFoundError(f"No images found in: {train_image_dir}")

        sampled_paths = sample_uniform(image_paths, num_samples=args.num_samples)
        frames = load_and_resize(sampled_paths, scale=args.resize_scale)

        prompt_messages = create_prompt_message(frames)

        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=prompt_messages,
                max_tokens=70,
            )
            caption = resp.choices[0].message.content
            print(f"[INFO] scene={scene_name} caption:\n{caption}")

            # IMPORTANT: output filename and nested keys are designed for enhance_utils.py
            result_path = args.output or os.path.join(source_path, f"batch_captions_result_{scene_name}.jsonl")
            result_obj = convert_to_enhancer_compatible_json(scene_name, caption)

            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result_obj, f, ensure_ascii=False)

            print(f"[INFO] Saved enhancer-compatible result: {result_path}")

        except Exception as e:
            print(f"[ERROR] scene={scene_name} request failed: {repr(e)}")

        time.sleep(0.5)

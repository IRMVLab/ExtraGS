"""Batch-generate a single scene caption from Nerfbusters-style image folders.

This script is designed for folders like:
  <scene_dir>/images_train/frame_00001.png
  <scene_dir>/images_test/frame_1_00001.png

It samples a small set of frames (default: 15) from images_train and submits one
OpenAI Batch request. The downloaded result is the standard batch JSONL output
(e.g. batch_captions_result_aloe.jsonl).

Environment variables:
  OPENAI_API_KEY      Required
  OPENAI_BASE_URL     Optional (for proxies), e.g. https://az.gptplus5.com/v1

Example:
  OPENAI_API_KEY=... \
  python ExploreGS/data_preprocess/gpt_batch_nerfbusters_scene_caption.py \
    --scene_dir ExploreGS/dataset/nerfbusters-dataset/aloe \
    --scene_name aloe
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path

import cv2
from openai import OpenAI


SYSTEM_MESSAGE_DICT = {
    "role": "system",
    "content": (
        "You are a helpful assistant that can caption videos. "
        "This caption will be used as input for a CLIP text encoder. "
        "Please describe the video frames in detail while keeping the max tokens under 70."
    ),
}


def _sample_evenly(items: list[Path], k: int) -> list[Path]:
    if k <= 0:
        return []
    if len(items) <= k:
        return items
    if k == 1:
        return [items[len(items) // 2]]

    # Evenly sample k indices in [0, n-1] inclusive.
    n = len(items)
    indices: list[int] = []
    for i in range(k):
        idx = round(i * (n - 1) / (k - 1))
        indices.append(int(idx))

    # De-dup if rounding caused collisions.
    out: list[Path] = []
    seen: set[int] = set()
    for idx in indices:
        if idx in seen:
            continue
        seen.add(idx)
        out.append(items[idx])
    return out


def _encode_frame_as_data_url_b64_png(frame_bgr) -> str:
    ok, buffer = cv2.imencode(".png", frame_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode('.png', frame) failed")
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def create_prompt_message(frames_bgr) -> list[dict]:
    return [
        SYSTEM_MESSAGE_DICT,
        {
            "role": "user",
            "content": [
                "Please caption the following video frames:",
                *(
                    {
                        "type": "image_url",
                        "image_url": {"url": _encode_frame_as_data_url_b64_png(frame)},
                    }
                    for frame in frames_bgr
                ),
            ],
        },
    ]


def _make_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=Path, required=True)
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--images_subdir", type=str, default="images_train")
    parser.add_argument("--num_frames", type=int, default=15)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--max_tokens", type=int, default=70)
    parser.add_argument("--poll_interval", type=float, default=10.0)
    args = parser.parse_args()

    scene_dir: Path = args.scene_dir
    images_dir = scene_dir / args.images_subdir
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images dir not found: {images_dir}")

    image_paths = sorted(
        [
            *images_dir.glob("*.png"),
            *images_dir.glob("*.jpg"),
            *images_dir.glob("*.jpeg"),
        ]
    )
    if not image_paths:
        raise RuntimeError(f"No images found under {images_dir}")

    sampled_paths = _sample_evenly(image_paths, args.num_frames)

    frames_bgr = []
    for p in sampled_paths:
        frame = cv2.imread(str(p))
        if frame is None:
            raise RuntimeError(f"cv2.imread failed: {p}")
        if args.scale != 1.0:
            frame = cv2.resize(frame, dsize=(0, 0), fx=args.scale, fy=args.scale)
        frames_bgr.append(frame)

    prompt_messages = create_prompt_message(frames_bgr)

    request_body = {
        "model": args.model,
        "messages": prompt_messages,
        "max_tokens": args.max_tokens,
    }

    request_data = {
        "custom_id": f"_{args.scene_name}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": request_body,
    }

    input_filename = scene_dir / f"batchinput_{args.scene_name}.jsonl"
    with input_filename.open("w", encoding="utf-8") as f:
        f.write(json.dumps(request_data, ensure_ascii=False) + "\n")

    client = _make_openai_client()

    batch_input_file = client.files.create(file=input_filename.open("rb"), purpose="batch")
    batch_input_file_id = batch_input_file.id
    print(f"[INFO] Created batch input file: {batch_input_file_id}")

    created_batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"nerfbusters scene caption: {args.scene_name}"},
    )
    print(f"[INFO] Batch created: {created_batch.id} status={created_batch.status}")

    while True:
        batch_status = client.batches.retrieve(created_batch.id)
        print(f"[INFO] Batch status: {batch_status.status}")
        if batch_status.status in ["completed", "failed", "expired"]:
            break
        time.sleep(args.poll_interval)

    if batch_status.status != "completed":
        raise RuntimeError(
            f"Batch ended with status={batch_status.status}. "
            "Check batch_status.error_file_id if present."
        )

    output_file_id = batch_status.output_file_id
    if not output_file_id:
        raise RuntimeError("Batch completed but output_file_id is empty")

    result_bytes = client.files.content(output_file_id).content
    result_file = scene_dir / f"batch_captions_result_{args.scene_name}.jsonl"
    with result_file.open("wb") as f:
        f.write(result_bytes)

    print(f"[INFO] Wrote: {result_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

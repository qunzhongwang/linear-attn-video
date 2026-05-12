"""Drop-in replacement for `torchvision.io.write_video`, which was removed in torchvision 0.26.

The trainers expect:
    write_video(filename, video_array, fps, video_codec='libx264')
where video_array is a uint8 (T, H, W, C) tensor or numpy array.

We delegate to imageio's libx264 writer (already in the env via `imageio[ffmpeg]`).
"""
from __future__ import annotations

import os
from typing import Optional

import imageio
import numpy as np
import torch


def write_video(
    filename: str,
    video_array,                        # uint8 (T, H, W, C) tensor or ndarray
    fps: float,
    video_codec: str = "libx264",
    options: Optional[dict] = None,
    **_unused,
) -> None:
    if isinstance(video_array, torch.Tensor):
        video_array = video_array.detach().cpu().numpy()
    if video_array.dtype != np.uint8:
        video_array = np.clip(video_array, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    writer = imageio.get_writer(filename, fps=fps, codec=video_codec, quality=8, macro_block_size=1)
    for frame in video_array:
        writer.append_data(frame)
    writer.close()

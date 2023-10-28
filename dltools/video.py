import pathlib
import time

import cv2
import decord
import moviepy.editor as mpy
import numpy as np
import torch

from . import image as image_utils

decord.bridge.set_bridge("torch")


class DecordVideo(object):
    def __init__(self, path, num_threads=1):
        super().__init__()
        if isinstance(path, str):
            path = pathlib.Path(path)
        self.path = path
        self.num_threads = num_threads
        self.capture = decord.VideoReader(
            self.path.as_posix(), ctx=decord.cpu(0), num_threads=num_threads
        )

    def reset(self):
        self.capture.__exit__()
        self.capture = decord.VideoReader(
            self.path.as_posix(), ctx=decord.cpu(0), num_threads=self.num_threads
        )

    def set_capture(self, frame_idx):
        self.capture.seek(frame_idx)

    def __exit__(self):
        self.capture.__exit__()

    def get_frame(self, frame_idx):
        return self.capture[frame_idx].permute(2, 0, 1)

    def get_frames(self, frame_idxs):
        frames = []
        for frame_idx in frame_idxs:
            frame = self.capture[frame_idx].permute(2, 0, 1)
            frames.append(frame)
        frames = torch.stack(frames, dim=0)
        return frames

    def get_all_frames(self):
        return self.get_chunk(0, self.get_num_frames())

    def get_next_chunk(self, chunk_size):
        frames = []
        for i in range(chunk_size):
            frame = self.capture.next().permute(2, 0, 1)
            if frame is None:
                break
            frames.append(frame)
        frames = torch.stack(frames, dim=0)
        return frames

    def get_chunk(self, start, end):
        frames = self.capture[start:end]
        frames = frames.permute(0, 3, 1, 2)
        return frames

    def get_num_frames(self):
        return len(self.capture)

    def get_frame_size(self):
        return tuple(self.capture[0].shape[:2])  # height, width

    def get_fps(self):
        return self.capture.get_avg_fps()

    def get_audio(self):
        self.audio = mpy.AudioFileClip(self.path.as_posix())
        return self.audio

    def set_audio(self, audio):
        self.audio = audio


def video_from_frames(
    frames, filepath, fps, codec="mp4v", audio_file=None, audio_subclip=None
):
    if isinstance(frames, list):
        if torch.is_tensor(frames[0]):
            if frames[0].ndim == 4:
                frames = torch.cat(frames, dim=0)
            else:
                frames = torch.stack(frames, dim=0)
        else:
            if frames[0].ndim == 4:
                frames = np.concatenate(frames, axis=0)
            else:
                frames = np.stack(frames, axis=0)

    if torch.is_tensor(frames):
        frames = image_utils.torch2cv(frames)

    if frames.dtype in (float, np.float32):
        frames = (frames * 255.0).astype(np.uint8)

    res = frames.shape[1:3]

    filepath.parent.mkdir(parents=True, exist_ok=True)

    if audio_file is not None:
        home = pathlib.Path.home()
        temp_file = home / f"dump/{int(time.time() * 1e6)}.mp4"
        temp_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        temp_file = filepath

    video = cv2.VideoWriter(
        temp_file.as_posix(), cv2.VideoWriter_fourcc(*codec), fps, (res[1], res[0])
    )
    for frame in frames:
        video.write(frame)
    video.release()

    if audio_file is not None:
        add_audio(temp_file, audio_file, filepath, audio_subclip)
        temp_file.unlink()


def add_audio(video_path, audio_path, target_path, audio_subclip=None):
    video_clip = mpy.VideoFileClip(video_path.as_posix())
    audio_clip = mpy.AudioFileClip(audio_path.as_posix())

    if audio_subclip is not None:
        audio_clip = audio_clip.subclip(*audio_subclip)
    video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile(target_path.as_posix(), logger=None)

    video_clip.close()
    audio_clip.close()

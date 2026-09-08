"""Extract a normalized pose-landmark sequence from a video using MediaPipe's
Pose Landmarker (the Tasks API -- MediaPipe 0.10.30+ dropped the older
`mp.solutions.pose` convenience API entirely, so this uses the model-based
Tasks API instead, per the current MediaPipe docs).

Usage (as a library):
    from src.pose_estimation import extract_pose_sequence
    seq = extract_pose_sequence("clip.mp4", sample_fps=5)
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe import Image, ImageFormat

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_lite.task"

# Landmark indices we use for comparison -- limb/torso joints that matter for
# posture and movement, not e.g. the individual face-mesh-like points
# MediaPipe's pose model also emits. These match BlazePose's fixed 33-point
# layout (see MediaPipe's pose landmark docs) -- not derived from any
# particular reference implementation's numbering.
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
COMPARISON_LANDMARKS = [
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
]


@dataclass
class PoseFrame:
    """One sampled frame's pose, normalized to be invariant to the subject's
    distance from the camera and position in frame."""

    timestamp_s: float
    points: np.ndarray  # shape (len(COMPARISON_LANDMARKS), 2), centered + scaled
    visibility: np.ndarray  # shape (len(COMPARISON_LANDMARKS),), 0-1 per point
    detected: bool
    raw_landmarks: object = field(default=None, repr=False)  # full landmark list, for drawing


def _ensure_model() -> str:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        print(f"Downloading pose landmarker model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return str(MODEL_PATH)


def _normalize(landmarks, frame_w: int, frame_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-space landmarks, centered on mid-hip and scaled by torso length,
    so two videos shot at different distances/resolutions/aspect ratios are
    still comparable on the same footing."""
    all_xy = np.array([[lm.x * frame_w, lm.y * frame_h] for lm in landmarks])

    hip_mid = (all_xy[LEFT_HIP] + all_xy[RIGHT_HIP]) / 2.0
    shoulder_mid = (all_xy[LEFT_SHOULDER] + all_xy[RIGHT_SHOULDER]) / 2.0
    torso_length = float(np.linalg.norm(shoulder_mid - hip_mid)) or 1.0

    points = np.array([all_xy[i] for i in COMPARISON_LANDMARKS])
    points = (points - hip_mid) / torso_length

    visibility = np.array([landmarks[i].visibility for i in COMPARISON_LANDMARKS])
    return points, visibility


def extract_pose_sequence(video_path: str, sample_fps: float = 5.0) -> list[PoseFrame]:
    """Sample a video at ~sample_fps and run MediaPipe's Pose Landmarker on
    each sampled frame."""
    model_path = _ensure_model()
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, round(native_fps / sample_fps))

    sequence: list[PoseFrame] = []
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(frame_idx / native_fps * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                timestamp = frame_idx / native_fps
                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    points, visibility = _normalize(landmarks, frame_w, frame_h)
                    sequence.append(PoseFrame(timestamp, points, visibility, True, landmarks))
                else:
                    sequence.append(PoseFrame(timestamp, np.zeros((len(COMPARISON_LANDMARKS), 2)),
                                               np.zeros(len(COMPARISON_LANDMARKS)), False, None))
            frame_idx += 1
    cap.release()
    return sequence

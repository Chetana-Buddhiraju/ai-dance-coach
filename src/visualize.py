"""Render pose skeletons and build a side-by-side comparison GIF from an
aligned pair sequence. Draws only the extracted stick-figure skeleton, not
the source video frames -- keeps the output privacy-preserving by default
(no camera footage leaves the pipeline) and doubles as a clear illustration
of exactly what signal the comparison is actually based on.
"""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from src.pose_estimation import (
    LEFT_ANKLE, LEFT_ELBOW, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER, LEFT_WRIST,
    RIGHT_ANKLE, RIGHT_ELBOW, RIGHT_HIP, RIGHT_KNEE, RIGHT_SHOULDER, RIGHT_WRIST,
    COMPARISON_LANDMARKS,
)
from src.pose_compare import AlignedPair

# Index-into-COMPARISON_LANDMARKS bone connections (not raw MediaPipe indices)
_IDX = {lm: i for i, lm in enumerate(COMPARISON_LANDMARKS)}
BONES = [
    (LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP), (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
]
BONES_IDX = [(_IDX[a], _IDX[b]) for a, b in BONES]


def draw_skeleton(points, visibility, size=(360, 480), color=(86, 211, 100), bg=(13, 17, 23)) -> Image.Image:
    """points: normalized (12,2) array centered on mid-hip, scaled by torso length."""
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    cx, cy = size[0] // 2, size[1] // 2
    scale = size[1] / 6.0  # normalized units -> pixels; torso ~= 1 unit, full body ~= 4-5 units

    def to_px(pt):
        return (cx + pt[0] * scale, cy + pt[1] * scale)

    for a, b in BONES_IDX:
        if visibility[a] > 0.3 and visibility[b] > 0.3:
            d.line([to_px(points[a]), to_px(points[b])], fill=color, width=4)
    for i, pt in enumerate(points):
        if visibility[i] > 0.3:
            x, y = to_px(pt)
            d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=color)
    return img


def build_comparison_gif(reference_seq, candidate_seq, aligned_pairs: list[AlignedPair],
                          out_path: str, subsample: int = 3, frame_ms: int = 140) -> None:
    font = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 18)
    font_small = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 15)

    panel_w, panel_h = 340, 460
    header_h = 50
    W, H = panel_w * 2 + 20, panel_h + header_h

    # Skip aligned pairs where either side has too few confidently-detected
    # joints to draw a recognizable figure (e.g. the dancer stepping into
    # frame at the start of a clip) -- this only affects which frames make it
    # into the GIF, not the underlying score, which already weights every
    # frame (including sparse ones) by per-joint visibility.
    well_tracked = [p for p in aligned_pairs
                    if (reference_seq[p.ref_idx].visibility > 0.3).sum() >= 6
                    and (candidate_seq[p.cand_idx].visibility > 0.3).sum() >= 6]
    pairs_for_gif = well_tracked or aligned_pairs

    frames = []
    shown = pairs_for_gif[::subsample] or pairs_for_gif
    for k, pair in enumerate(shown):
        ref = reference_seq[pair.ref_idx]
        cand = candidate_seq[pair.cand_idx]
        ref_img = draw_skeleton(ref.points, ref.visibility, size=(panel_w, panel_h), color=(110, 168, 254))
        cand_img = draw_skeleton(cand.points, cand.visibility, size=(panel_w, panel_h), color=(255, 166, 87))

        canvas = Image.new("RGB", (W, H), (13, 17, 23))
        d = ImageDraw.Draw(canvas)
        dist_txt = "n/a" if pair.distance != pair.distance else f"{pair.distance:.3f}"
        d.text((10, 8), "Reference", font=font, fill=(110, 168, 254))
        d.text((panel_w + 30, 8), "Candidate", font=font, fill=(255, 166, 87))
        d.text((W - 190, 10), f"frame {k+1}/{len(shown)}  \u00b7  dist={dist_txt}", font=font_small, fill=(139, 148, 158))
        canvas.paste(ref_img, (0, header_h))
        canvas.paste(cand_img, (panel_w + 20, header_h))
        d.line([(panel_w + 10, header_h), (panel_w + 10, H)], fill=(48, 54, 61), width=2)
        frames.append(canvas)

    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=frame_ms, loop=0, optimize=True)

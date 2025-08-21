# AI Dance Coach
ai-dance-coach — A computer-vision–based “Dance Coach” that uses pose estimation to track movement, compare with reference choreography, and give feedback on posture, rhythm and timing.

## About / Motivation

I started this passion project while practicing alone in Singapore while my troupe was rehearsing in India. I wanted a way to self-correct without a partner — so I combined two things I love: dance and AI.
The idea: use pose estimation to track a dancer’s movement, compare it to reference choreography, and give near-real-time feedback on posture, timing and alignment — like a coach by your side.

I paused development while juggling my NUS AI coursework, performances and an internship at SGH, but I’m returning to this because it’s both personal and educational.

### Current Plan

.Dataset gathered: dance videos (Hip Hop & Ballet) and extracted frames.
.Basic data-prep code (frame extraction + resizing) in a Colab notebook and simple scripts.
.Pose-estimation pipeline (planned: Mediapipe / MMPose / OpenPose)
.Posture comparison & feedback model
.Simple UI (Streamlit/Gradio) for an interactive coach

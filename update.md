# ComfyUI-Open-H3-Regenerate-2K Update Log

## V1.0.0 (2026/10/28)
### Compatibility & Model Architecture
- **Initial Release (`AILab_OpenH3_Regenerate_2K`)**:
  - Engineered for NVIDIA's Open-H3-Regenerate-2K pipeline on MiniMax H3.
  - Unified 2K refinement into a single node outputting patched `MODEL`, audio-locked `LATENT`, and calibrated `SIGMAS`.
  - Built-in automatic model download system from `1038lab/Open-H3-Regenerate-2K` to `ComfyUI/models/loras/minmax/`.
  - Full recursive search support across all `models/loras/` subdirectories.
  - Native SwiGLU-fixed BF16 support: Full-Rank (`8.80 GB`) and Rank-128 (`1.95 GB`).
  - Integrated 50 LMS Sharp AdaLN curve branches (`sharp_curve`) and audio protection (`lock_audio`).

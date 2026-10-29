# ComfyUI-Open-H3-Regenerate-2K

A dedicated ComfyUI custom node engineered for NVIDIA's **Open-H3-Regenerate-2K** video super-resolution pipeline on MiniMax H3. It integrates 2K latent refinement into a single unified node with automatic weight downloading, SwiGLU-fixed native BF16 support, 50 LMS Sharp AdaLN curve branch injection, calibrated Euler 2/3-step sigmas, and audio-locked latent masking.

<img width="1373" height="686" alt="image" src="https://github.com/user-attachments/assets/8f794f1a-6c45-429e-bcc4-86741276fec5" />


## News & Updates

- **2026/10/28**: Release ComfyUI-Open-H3-Regenerate-2K **v1.0.0** ( [update.md](update.md#v100-20261028) )
  - Initial release of `Open-H3 Regenerate 2K` (`AILab_OpenH3_Regenerate_2K`) node.
  - Built-in automatic model download system fetching weights directly to `ComfyUI/models/loras/minmax/`.
  - Added SwiGLU-fixed native BF16 (`8.80 GB`) and lightweight Rank-128 BF16 (`1.95 GB`) models.
  - Built-in 50 LMS Sharp AdaLN curve branches (`sharp_curve`) and audio-locking (`lock_audio`).

## Features

- **All-in-One 2K Refinement**: Directly outputs patched `MODEL`, audio-locked `LATENT`, and official 2/3-step `SIGMAS` in a single pass.
- **Automatic Model Download**: Missing weights automatically download to `ComfyUI/models/loras/minmax/` on first run.
- **Flexible Directory Lookup**: Recursively searches `models/loras/` so users can organize models in any subfolder without re-downloading.
- **SwiGLU-Fixed BF16 Native Weights**: Pre-fused QKV and aligned SwiGLU channels for ComfyUI H3 base architecture.
- **50 LMS Sharp AdaLN Curve Branches**: Built-in support for Sharp curve conditioning to enhance 2K video textures.
- **Audio Stream Protection**: Built-in `lock_audio` masks audio latents at sigma 0 so soundtrack fidelity is fully preserved.

## Installation

### Method 1: Install via ComfyUI Manager (Recommended)
Search for `ComfyUI-Open-H3-Regenerate-2K` in ComfyUI Manager and click **Install**.

### Method 2: Manual Clone
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/1038lab/ComfyUI-Open-H3-Regenerate-2K.git
pip install -r ComfyUI-Open-H3-Regenerate-2K/requirements.txt
```

## Model Downloads

> [!TIP]
> **Automatic Download Enabled**: You do **not** need to download Open-H3 LoRA weights manually. The node downloads them on-demand to `ComfyUI/models/loras/minmax/`. Direct download links are provided below for offline environments.

### 1. Open-H3 2K LoRA Weights (Target: `ComfyUI/models/loras/minmax/`)
- **Open-H3-Regenerate-2K Full-Rank BF16 (8.80 GB, Recommended)**:
  Download [`open_h3_regenerate_2k_bf16.safetensors`](https://huggingface.co/1038lab/Open-H3-Regenerate-2K/resolve/main/lora/open_h3_regenerate_2k_bf16.safetensors)
- **Open-H3-Regenerate-2K r128 BF16 (1.95 GB, Fast / Low VRAM)**:
  Download [`minimax_h3_open_regenerate_2k_r128_bf16.safetensors`](https://huggingface.co/1038lab/Open-H3-Regenerate-2K/resolve/main/lora/minimax_h3_open_regenerate_2k_r128_bf16.safetensors)
- **Sharp Curve Branches (1.24 GB, 50 AdaLN branches)**:
  Download [`sharp_curve_branches.safetensors`](https://huggingface.co/1038lab/Open-H3-Regenerate-2K/resolve/main/lora/sharp_curve_branches.safetensors)

### 2. MiniMax H3 Base Models (Required for Workflow)
Place these standard MiniMax H3 models in their respective ComfyUI directories:
- **Diffusion Model** (`ComfyUI/models/diffusion_models/`):
  [`minimax_h3_ref2va_pruned_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors)
- **Text Encoder** (`ComfyUI/models/text_encoders/`):
  [`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors)
- **Video VAE** (`ComfyUI/models/vae/`):
  [`minimax_h3_video_vae_fp16.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors)
- **Audio VAE** (`ComfyUI/models/vae/`):
  [`minimax_h3_audio_vae_fp32.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors)
- **3D Latent Upscaler** (`ComfyUI/models/latent_upscale_models/`):
  [`minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors`](https://huggingface.co/1038lab/Open-H3-Regenerate-2K/resolve/main/latent_upscale_models/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors)

## Usage

### Node: Open-H3 Regenerate 2K (`AILab_OpenH3_Regenerate_2K`)
Category: `🧪AILab/Open-H3`

1. Connect the MiniMax H3 base model to `model`.
2. Connect the 3D-upscaled joint audio-video latent to `latent`.
3. Choose your preset in `model_name`:
   - `Open-H3-Regenerate-2K (BF16)` for highest 2K quality.
   - `Open-H3-Regenerate-2K-r128 (BF16)` for faster generation and lower memory.
4. Outputs:
   - `MODEL`: Patched model ready for sampling.
   - `LATENT`: Latent with audio stream protected at sigma 0.
   - `SIGMAS`: Official calibrated Euler sigmas (3-step or 2-step) to connect to `SamplerCustomAdvanced`.

### Recommended Settings
- **Sampler**: `euler`
- **CFG**: `1.0`
- **Steps**: `"3"` (official schedule: `0.909375 -> 0.857143 -> 0.666667 -> 0.0`)
- **Strength**: `1.0`
- **Sharp Curve**: `True`
- **Lock Audio**: `True`

## Workflows

## License & Acknowledgements

- **Model Hosting & Conversion**: [1038lab/Open-H3-Regenerate-2K](https://huggingface.co/1038lab/Open-H3-Regenerate-2K)
- **Upstream Research**: [NVIDIA / Efficient-Large-Model — Open-H3-Regenerate-2K](https://nvlabs.github.io/Sana/Open-H3-Regenerate-2K/), [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3), [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler).
- **License**: Code released under [GPL-3.0 License](LICENSE).

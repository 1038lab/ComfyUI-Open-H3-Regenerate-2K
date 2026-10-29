import os
import gc
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

import folder_paths
import comfy.model_management as mm
from comfy.nested_tensor import NestedTensor
from comfy.weight_adapter.lora import LoRAAdapter

BASE_LORA_DIR = folder_paths.get_folder_paths("loras")[0] if folder_paths.get_folder_paths("loras") else os.path.join(folder_paths.models_dir, "loras")
MINMAX_LORA_DIR = os.path.join(BASE_LORA_DIR, "minmax")

AVAILABLE_MODELS = {
    "Open-H3-Regenerate-2K (BF16)": {
        "type": "comfy_down_up",
        "repo_id": "1038lab/Open-H3-Regenerate-2K",
        "files": {
            "lora/open_h3_regenerate_2k_bf16.safetensors": "open_h3_regenerate_2k_bf16.safetensors",
        },
        "cache_dir": MINMAX_LORA_DIR,
    },
    "Open-H3-Regenerate-2K-r128 (BF16)": {
        "type": "comfy_lora_ab",
        "repo_id": "1038lab/Open-H3-Regenerate-2K",
        "files": {
            "lora/minimax_h3_open_regenerate_2k_r128_bf16.safetensors": "minimax_h3_open_regenerate_2k_r128_bf16.safetensors",
        },
        "cache_dir": MINMAX_LORA_DIR,
    },
    "Sharp-Curve-Branches": {
        "type": "sharp_curve",
        "repo_id": "1038lab/Open-H3-Regenerate-2K",
        "files": {
            "lora/sharp_curve_branches.safetensors": "sharp_curve_branches.safetensors",
        },
        "cache_dir": MINMAX_LORA_DIR,
    },
}

SIGMA_SCHEDULES = {
    "3": (0.909375, 0.8571428571428571, 0.6666666666666666, 0.0),
    "2": (0.7, 0.49411764705882355, 0.0),
}


def handle_model_error(e, msg="Failed to load Open-H3 model"):
    print(f"[Open-H3 ERROR] {msg}: {str(e)}")
    raise RuntimeError(f"{msg}: {str(e)}\nPlease ensure the model exists in ComfyUI/models/loras/minmax or can be downloaded from 1038lab/Open-H3-Regenerate-2K.")


def clean_vram():
    gc.collect()
    mm.soft_empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def unload_model_safely(model_patcher):
    if hasattr(mm, "unload_model_and_clones"):
        try:
            mm.unload_model_and_clones(model_patcher)
        except Exception:
            pass
    if hasattr(model_patcher, "model") and hasattr(model_patcher.model, "to"):
        try:
            model_patcher.model.to("cpu")
        except Exception:
            pass
    clean_vram()


class BaseModelLoader:
    def __init__(self):
        self.model = None
        self.current_model_version = None
        self.base_cache_dir = MINMAX_LORA_DIR

    def get_cache_dir(self, model_name):
        return AVAILABLE_MODELS[model_name]["cache_dir"]

    def resolve_file_path(self, model_name, remote_name, local_name):
        cache_dir = self.get_cache_dir(model_name)
        direct_candidates = [
            os.path.join(cache_dir, local_name),
            os.path.join(cache_dir, remote_name),
            folder_paths.get_full_path("loras", os.path.join("minmax", local_name)),
            folder_paths.get_full_path("loras", local_name),
        ]
        for path in direct_candidates:
            if path and os.path.exists(path):
                return path

        for rel_path in folder_paths.get_filename_list("loras"):
            if os.path.basename(rel_path).lower() == local_name.lower():
                full_path = folder_paths.get_full_path("loras", rel_path)
                if full_path and os.path.exists(full_path):
                    return full_path

        return os.path.join(cache_dir, remote_name)

    def check_model_cache(self, model_name):
        model_info = AVAILABLE_MODELS[model_name]
        missing_files = []
        for remote_name, local_name in model_info["files"].items():
            if not os.path.exists(self.resolve_file_path(model_name, remote_name, local_name)):
                missing_files.append(local_name)

        if missing_files:
            return False, f"Missing model files: {', '.join(missing_files)}"
        return True, "Model cache verified"

    def download_model(self, model_name):
        model_info = AVAILABLE_MODELS[model_name]
        cache_dir = self.get_cache_dir(model_name)
        try:
            os.makedirs(cache_dir, exist_ok=True)
            for remote_name, local_name in model_info["files"].items():
                print(f"Downloading {local_name} for {model_name} to {cache_dir}...")
                downloaded = hf_hub_download(
                    repo_id=model_info["repo_id"],
                    filename=remote_name,
                    local_dir=cache_dir,
                )
                target_path = os.path.join(cache_dir, local_name)
                if os.path.abspath(downloaded) != os.path.abspath(target_path) and os.path.exists(downloaded):
                    os.replace(downloaded, target_path)
                    subdir = os.path.dirname(os.path.join(cache_dir, remote_name))
                    if os.path.isdir(subdir) and os.path.abspath(subdir) != os.path.abspath(cache_dir) and not os.listdir(subdir):
                        try:
                            os.rmdir(subdir)
                        except OSError:
                            pass
            return True, "Model files downloaded"
        except Exception as e:
            return False, f"Error downloading model files: {str(e)}"

    def ensure_model_file(self, model_name):
        cache_status, message = self.check_model_cache(model_name)
        if not cache_status:
            print(f"Cache check: {message}")
            print(f"Downloading required model files for {model_name}...")
            download_status, download_message = self.download_model(model_name)
            if not download_status:
                handle_model_error(download_message)
        remote_name, local_name = next(iter(AVAILABLE_MODELS[model_name]["files"].items()))
        return self.resolve_file_path(model_name, remote_name, local_name)

    def clear_model(self):
        if self.model is not None:
            del self.model
            self.model = None
        self.current_model_version = None
        clean_vram()


class OpenH3ModelLoader(BaseModelLoader):
    def load_main_patches(self, model_name: str, state_dict: dict) -> dict:
        ckpt_path = self.ensure_model_file(model_name)
        fmt = AVAILABLE_MODELS[model_name]["type"]
        patches = {}

        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            keys = set(f.keys())

            if fmt == "comfy_down_up":
                roots = sorted(k[:-len(".lora_down.weight")] for k in keys if k.endswith(".lora_down.weight"))
                for root in roots:
                    down_key, up_key = f"{root}.lora_down.weight", f"{root}.lora_up.weight"
                    down, up = f.get_tensor(down_key), f.get_tensor(up_key)
                    target = f"diffusion_model.{root}.weight"
                    if target not in state_dict or state_dict[target].shape[0] != up.shape[0]:
                        raise ValueError(f"Mismatched target in base model: {target}")
                    patches[target] = LoRAAdapter({down_key, up_key}, (up, down, None, None, None, None))

            elif fmt == "comfy_lora_ab":
                roots = sorted(k[:-len(".lora_A.weight")] for k in keys if k.endswith(".lora_A.weight"))
                for root in roots:
                    a_key, b_key = f"{root}.lora_A.weight", f"{root}.lora_B.weight"
                    a, b = f.get_tensor(a_key), f.get_tensor(b_key)
                    target = f"{root}.weight" if root.startswith("diffusion_model.") else f"diffusion_model.{root}.weight"
                    if target not in state_dict or state_dict[target].shape[0] != b.shape[0]:
                        raise ValueError(f"Mismatched target in base model: {target}")
                    patches[target] = LoRAAdapter({a_key, b_key}, (b, a, None, None, None, None))

        return patches

    def load_curve_patches(self, diffusion_model, state_dict: dict) -> dict:
        table = getattr(diffusion_model, "adaln_t_table", None)
        if not torch.is_tensor(table) or tuple(table.shape) != (1025, 8):
            raise ValueError(
                "Open-H3 Sharp curve branches require the MiniMax H3 1025x8 curve-basis model "
                "(minimax_h3_ref2va_pruned_int8_convrot.safetensors)."
            )

        curve_path = self.ensure_model_file("Sharp-Curve-Branches")
        patches = {}
        with safe_open(curve_path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            roots = sorted(k[:-len(".curve_A.weight")] for k in keys if k.endswith(".curve_A.weight"))
            for root in roots:
                a_key, b_key = f"{root}.curve_A.weight", f"{root}.curve_B.weight"
                a, b = f.get_tensor(a_key), f.get_tensor(b_key)
                block_idx = root[len("transformer.transformer_blocks."):].split(".", 1)[0]
                target = f"diffusion_model.blocks.{block_idx}.adaln_proj.linear.weight"
                if target not in state_dict or state_dict[target].shape[0] != b.shape[0]:
                    raise ValueError(f"Mismatched Sharp curve target in base model: {target}")
                patches[target] = LoRAAdapter({a_key, b_key}, (b, a, None, None, None, None))
        return patches


def apply_audio_lock(latent: dict) -> dict:
    samples = latent.get("samples")
    if not getattr(samples, "is_nested", False):
        raise ValueError("Expected a MiniMax H3 joint video/audio NestedTensor latent.")
    streams = samples.unbind()
    if len(streams) != 2 or streams[0].ndim != 5 or streams[0].shape[1] != 24:
        raise ValueError("Expected H3 video stream (24ch) followed by audio stream.")
    out = latent.copy()
    out["noise_mask"] = NestedTensor((torch.ones_like(streams[0]), torch.zeros_like(streams[1])))
    return out


class AILab_OpenH3_Regenerate_2K:
    def __init__(self):
        self.loader = OpenH3ModelLoader()

    @classmethod
    def INPUT_TYPES(cls):
        model_names = [k for k, v in AVAILABLE_MODELS.items() if v["type"] != "sharp_curve"]
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "MiniMax H3 base model (minimax_h3_ref2va_pruned_int8_convrot.safetensors)."}),
                "model_name": (model_names, {"default": model_names[0], "tooltip": "Select Open-H3 Regenerate 2K model preset. Auto-downloads from 1038lab/Open-H3-Regenerate-2K to ComfyUI/models/loras/minmax/ if missing."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.5, "step": 0.01, "tooltip": "Patch strength applied to the Open-H3 2K model and Sharp AdaLN curve branches."}),
                "sharp_curve": ("BOOLEAN", {"default": True, "tooltip": "Load the 50 LMS Sharp AdaLN curve branches (sharp_curve_branches.safetensors) for enhanced 2K sharpness."}),
                "steps": (["3", "2"], {"default": "3", "tooltip": "Official calibrated Euler refinement schedule: 3 steps for best stability, 2 steps for speed."}),
                "lock_audio": ("BOOLEAN", {"default": True, "tooltip": "Lock the audio stream in the joint H3 latent at sigma 0 so only video frames are regenerated."}),
                "unload_model": ("BOOLEAN", {"default": False, "tooltip": "Offload cached models and flush GPU VRAM immediately after patching."}),
            },
            "optional": {
                "latent": ("LATENT", {"tooltip": "Joint MiniMax H3 video/audio latent (av_latent) to refine at 2K resolution."}),
            },
        }

    RETURN_TYPES = ("MODEL", "LATENT", "SIGMAS")
    RETURN_NAMES = ("MODEL", "LATENT", "SIGMAS")
    FUNCTION = "process"
    CATEGORY = "🧪AILab/Open-H3"
    SEARCH_ALIASES = ["OpenH3", "Open-H3", "Regenerate2K", "Regenerate 2K", "MiniMax H3 2K"]

    def process(self, model, model_name, strength, sharp_curve=True, steps="3", lock_audio=True, unload_model=False, latent=None):
        try:
            patched = model.clone()
            state_dict = patched.model_state_dict()

            main_patches = self.loader.load_main_patches(model_name, state_dict)
            patched.add_patches(main_patches, strength_patch=strength)

            if sharp_curve:
                diffusion_model = model.get_model_object("diffusion_model")
                curve_patches = self.loader.load_curve_patches(diffusion_model, state_dict)
                patched.add_patches(curve_patches, strength_patch=strength)

            out_latent = latent
            if latent is not None and lock_audio:
                out_latent = apply_audio_lock(latent)

            sigmas = torch.tensor(SIGMA_SCHEDULES[str(steps)], dtype=torch.float32)
            return (patched, out_latent, sigmas)
        finally:
            if unload_model:
                self.loader.clear_model()
                unload_model_safely(model)
            else:
                clean_vram()


NODE_CLASS_MAPPINGS = {
    "AILab_OpenH3_Regenerate_2K": AILab_OpenH3_Regenerate_2K,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_OpenH3_Regenerate_2K": "Open-H3 Regenerate 2K",
}

"""ComfyUI nodes for ChatterBox Turbo TTS."""

import json
import os
import tempfile
import threading
import time

import numpy as np
import torch
import folder_paths

# Register model folder with ComfyUI
_MODEL_FOLDER = "TTS/chatterbox-turbo"
_model_dir = os.path.join(folder_paths.models_dir, _MODEL_FOLDER)
os.makedirs(_model_dir, exist_ok=True)
folder_paths.add_model_folder_path("chatterbox-turbo", _model_dir)

# Global model singleton
_model = None
_model_lock = threading.Lock()

_REPO_ID = "ResembleAI/chatterbox-turbo"
_ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"]


def _get_model():
    """Load or return cached ChatterBox Turbo model."""
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model

        from huggingface_hub import snapshot_download
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        # Download to ComfyUI models dir (not HF cache)
        print(f"[ChatterBox Turbo] Model dir: {_model_dir}")
        local_path = snapshot_download(
            repo_id=_REPO_ID,
            token=False,
            allow_patterns=_ALLOW_PATTERNS,
            local_dir=_model_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        _model = ChatterboxTurboTTS.from_local(local_path, device="cuda")
        print(f"[ChatterBox Turbo] Model loaded (SR={_model.sr})")
        return _model


def _audio_to_comfy(wav_np, sample_rate):
    """Convert numpy waveform to ComfyUI AUDIO dict."""
    t = torch.from_numpy(wav_np).float()
    if t.dim() == 1:
        t = t.unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
    elif t.dim() == 2:
        t = t.unsqueeze(0)
    return {"waveform": t, "sample_rate": sample_rate}


def _comfy_audio_to_wav_path(audio_dict):
    """Save ComfyUI AUDIO dict to a temp WAV file, return path."""
    import soundfile as sf

    waveform = audio_dict["waveform"]
    sr = audio_dict["sample_rate"]
    wav_np = waveform.squeeze().cpu().numpy()

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, wav_np, sr)
    return path


class ChatterboxTurboGenerate:
    """Single text-to-speech generation with ChatterBox Turbo."""

    CATEGORY = "audio/tts"
    FUNCTION = "generate"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.05, "max": 5.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            },
            "optional": {
                "audio_prompt_path": ("STRING", {"default": ""}),
                "reference_audio": ("AUDIO",),
            },
        }

    def generate(self, text, temperature, top_p, repetition_penalty, seed,
                 audio_prompt_path="", reference_audio=None):
        if seed > 0:
            torch.manual_seed(seed)

        model = _get_model()
        sr = model.sr

        # Determine voice reference
        voice_path = None
        temp_path = None

        if reference_audio is not None:
            temp_path = _comfy_audio_to_wav_path(reference_audio)
            voice_path = temp_path
        elif audio_prompt_path and audio_prompt_path.strip():
            voice_path = audio_prompt_path.strip()

        try:
            if voice_path:
                model.prepare_conditionals(voice_path)

            wav = model.generate(
                text,
                audio_prompt_path=None,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

        wav_np = wav.squeeze().cpu().numpy().astype(np.float32)
        return (_audio_to_comfy(wav_np, sr),)


class ChatterboxTurboDialogue:
    """Multi-speaker dialogue generation with ChatterBox Turbo."""

    CATEGORY = "audio/tts"
    FUNCTION = "generate"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script": ("STRING", {"multiline": True}),
                "voice_map": ("STRING", {"default": "{}"}),
                "pause_s": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.05}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.05, "max": 5.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            },
        }

    def generate(self, script, voice_map, pause_s, temperature, top_p,
                 repetition_penalty, seed):
        if seed > 0:
            torch.manual_seed(seed)

        model = _get_model()
        sr = model.sr

        # Parse voice map
        try:
            voices = json.loads(voice_map)
        except json.JSONDecodeError:
            voices = {}

        default_voice = next(iter(voices.values()), None) if voices else None

        # Parse script lines: "Speaker: [emotion] text"
        lines = []
        for raw_line in script.strip().split("\n"):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            colon_idx = raw_line.find(":")
            if colon_idx > 0:
                speaker = raw_line[:colon_idx].strip()
                text = raw_line[colon_idx + 1:].strip()
                if text:
                    lines.append({"speaker": speaker, "text": text})

        if not lines:
            empty = np.zeros(sr, dtype=np.float32)  # 1s silence
            return (_audio_to_comfy(empty, sr), "No lines parsed from script")

        # Generate each line
        pause_samples = np.zeros(int(sr * pause_s), dtype=np.float32)
        all_wavs = []
        current_voice = None
        info_lines = []
        t0 = time.time()

        for i, line in enumerate(lines):
            speaker = line["speaker"]
            text = line["text"]
            voice_path = voices.get(speaker, default_voice)

            # Switch voice if speaker changed
            if voice_path and voice_path != current_voice:
                model.prepare_conditionals(voice_path)
                current_voice = voice_path

            wav = model.generate(
                text,
                audio_prompt_path=None,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            wav_np = wav.squeeze().cpu().numpy().astype(np.float32)
            all_wavs.append(wav_np)

            if i < len(lines) - 1:
                all_wavs.append(pause_samples)

            dur = len(wav_np) / sr
            info_lines.append(f"[{i+1}/{len(lines)}] {speaker}: {text[:60]}... ({dur:.1f}s)")
            print(f"  [ChatterBox Turbo] {info_lines[-1]}")

        combined = np.concatenate(all_wavs)
        total_dur = len(combined) / sr
        gen_time = time.time() - t0

        info = "\n".join(info_lines)
        info += f"\nTotal: {total_dur:.1f}s, generation: {gen_time:.1f}s"
        print(f"  [ChatterBox Turbo] Total: {total_dur:.1f}s, gen: {gen_time:.1f}s")

        return (_audio_to_comfy(combined, sr), info)

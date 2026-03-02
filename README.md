# ComfyUI-ChatterBox-Turbo

ComfyUI custom nodes for [ChatterBox Turbo](https://github.com/resemble-ai/chatterbox) — a fast 350M-parameter text-to-speech model with paralinguistic emotion tag support.

## Features

- **Fast generation** — 1-step mel decoder, significantly faster than the original ChatterBox
- **Emotion tags** — `[dramatic]`, `[sarcastic]`, `[happy]`, `[surprised]`, `[whispering]`, and more
- **Voice cloning** — pass a short WAV reference to clone any voice
- **Multi-speaker dialogue** — generate full conversations with automatic voice switching
- **Auto model download** — model weights download automatically from HuggingFace on first use

## Nodes

### ChatterBox Turbo TTS (`ChatterboxTurboGenerate`)

Single text-to-speech generation. Good for experimentation and manual workflows.

| Input | Type | Default | Notes |
|-------|------|---------|-------|
| `text` | STRING (multiline) | *required* | Supports emotion tags: `[dramatic] Breaking news!` |
| `audio_prompt_path` | STRING | `""` | Path to voice reference WAV inside the container |
| `reference_audio` | AUDIO (optional) | — | Alternative: connect audio from another node |
| `temperature` | FLOAT | 0.8 | 0.05–5.0 |
| `top_p` | FLOAT | 0.95 | 0.0–1.0 |
| `repetition_penalty` | FLOAT | 1.2 | 1.0–2.0 |
| `seed` | INT | 0 | 0 = random |

**Output:** `AUDIO` (24 kHz)

### ChatterBox Turbo Dialogue (`ChatterboxTurboDialogue`)

Multi-speaker dialogue generation. Designed for pipeline integration and batch scripts.

| Input | Type | Default | Notes |
|-------|------|---------|-------|
| `script` | STRING (multiline) | *required* | One `Speaker: text` per line |
| `voice_map` | STRING | `"{}"` | JSON mapping speaker names to WAV paths |
| `pause_s` | FLOAT | 0.35 | Pause between lines (seconds) |
| `temperature` | FLOAT | 0.8 | |
| `top_p` | FLOAT | 0.95 | |
| `repetition_penalty` | FLOAT | 1.2 | |
| `seed` | INT | 0 | |

**Outputs:** `AUDIO` (concatenated dialogue), `STRING` (generation info)

**Script format:**
```
Mac: [dramatic] Breaking news everyone!
Luna: [surprised] That's right Mac, incredible!
Mac: [sarcastic] Can you believe it?
```

**Voice map example:**
```json
{
  "Mac": "/root/ComfyUI/custom_nodes/TTS-Audio-Suite/voices_examples/Mac.wav",
  "Luna": "/root/ComfyUI/custom_nodes/TTS-Audio-Suite/voices_examples/Luna.wav"
}
```

## Supported Emotion Tags

`[angry]` `[fear]` `[happy]` `[surprised]` `[crying]` `[sarcastic]` `[dramatic]` `[whispering]` `[narration]` `[advertisement]` `[laugh]` `[chuckle]` `[gasp]` `[sigh]` `[groan]` `[sniff]` `[cough]` `[clear throat]` `[shush]`

## Installation

### ComfyUI Manager

Search for "ChatterBox Turbo" in ComfyUI Manager and install.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/wobba/ComfyUI-ChatterBox-Turbo.git
pip install -r ComfyUI-ChatterBox-Turbo/requirements.txt
```

Restart ComfyUI after installation.

## Model Download

The model (~700 MB) downloads automatically from [`ResembleAI/chatterbox-turbo`](https://huggingface.co/ResembleAI/chatterbox-turbo) on HuggingFace the first time either node runs. It is cached in the default HuggingFace hub cache directory:

```
~/.cache/huggingface/hub/models--ResembleAI--chatterbox-turbo/
```

No manual download or placement in a ComfyUI `models/` subfolder is needed.

## Runtime Patches

This node includes three compatibility patches applied at import time:

1. **Perth watermarker** — The `perth` audio watermarking library uses `pkg_resources`, which was removed in Python 3.13. If the real Perth module loads successfully, watermarking works normally. On Python 3.13+ where Perth fails to import, a no-op mock is installed so generation still works — but output audio will **not** be watermarked. This is logged at startup.

2. **s3tokenizer float32** — On numpy 2.x, STFT operations return float64 instead of float32. The patch casts magnitudes back to float32 after STFT.

3. **prepare_conditionals float32** — On numpy 2.x, `librosa.load()` returns float64 arrays. The patch ensures audio arrays are cast to float32 before being passed to the model.

## Requirements

- ComfyUI with CUDA GPU
- Python 3.11+ (tested on 3.13)
- Dependencies: `chatterbox-tts`, `soundfile`, `librosa`, `huggingface_hub`
- PyTorch, torchaudio, and numpy are expected from the ComfyUI environment

## License

MIT

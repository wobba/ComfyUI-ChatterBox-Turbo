# ComfyUI-ChatterBox-Turbo

ComfyUI custom nodes for [ChatterBox Turbo](https://github.com/resemble-ai/chatterbox?tab=readme-ov-file#chatterbox-turbo) — a fast 350M-parameter text-to-speech model with paralinguistic emotion tag support by [Resemble AI](https://github.com/resemble-ai/chatterbox).

## Features

- **Fast generation** — 1-step mel decoder, significantly faster than the original ChatterBox
- **19 emotion/paralinguistic tags** — `[dramatic]`, `[sarcastic]`, `[happy]`, `[whispering]`, and more
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
  "Mac": "/path/to/Mac.wav",
  "Luna": "/path/to/Luna.wav"
}
```

## Emotion & Paralinguistic Tags

ChatterBox Turbo supports 19 special tokens trained into the model as `added_tokens` on top of the GPT-2 vocabulary (token IDs 50257–50275). The full list is defined in [`added_tokens.json`](https://huggingface.co/ResembleAI/chatterbox-turbo/blob/main/added_tokens.json) in the model repo. The upstream [ChatterBox README](https://github.com/resemble-ai/chatterbox?tab=readme-ov-file#chatterbox-turbo) mentions `[cough]`, `[laugh]`, `[chuckle]` as examples — the complete set of 19 is listed below.

Place a tag at the start of a line to set the delivery style for that line.

### Emotion Tags (affect the entire delivery)

| Tag | Best for | Example |
|-----|----------|---------|
| `[dramatic]` | Breaking news, reveals, cliffhangers | `[dramatic] This changes everything!` |
| `[sarcastic]` | Irony, dry humor, deadpan delivery | `[sarcastic] Oh sure, that went perfectly.` |
| `[happy]` | Upbeat, celebratory, good news | `[happy] We just hit a million subscribers!` |
| `[surprised]` | Reactions, plot twists, disbelief | `[surprised] Wait, are you serious right now?` |
| `[angry]` | Outrage, frustration, intensity | `[angry] This is completely unacceptable!` |
| `[fear]` | Tension, warnings, alarm | `[fear] Something is very wrong here.` |
| `[crying]` | Sadness, emotion, sympathy | `[crying] I cannot believe they are gone.` |
| `[whispering]` | Secrets, asides, intimacy | `[whispering] Nobody is supposed to know this.` |
| `[narration]` | Neutral storytelling, documentary tone | `[narration] The year was nineteen sixty nine.` |
| `[advertisement]` | Sales pitch, promo, upbeat sell | `[advertisement] Try it free for thirty days!` |

### Vocal Sound Effects (inject a specific sound)

| Tag | Effect |
|-----|--------|
| `[laugh]` | Full laugh |
| `[chuckle]` | Soft/brief laugh |
| `[gasp]` | Sharp intake of breath |
| `[sigh]` | Exhalation, weariness |
| `[groan]` | Displeasure, frustration |
| `[sniff]` | Nasal sound |
| `[cough]` | Cough |
| `[clear throat]` | Throat clearing |
| `[shush]` | Shushing sound |

### Tips for Script Writing

- **One tag per line.** Place it at the start: `[dramatic] The verdict is in.`
- **Emotion tags work best at sentence start.** Sound effects can go mid-sentence: `Well [chuckle] that was unexpected.`
- **Don't overuse.** Not every line needs a tag — untagged lines get natural delivery from the voice reference.
- **Match tag to content.** `[surprised]` on mundane text sounds odd. Use it for genuine reveals.
- **Best tags for news/TikTok scripts:**
  - `[dramatic]` — opening lines, breaking news, reveals
  - `[surprised]` — reactions to the other speaker
  - `[sarcastic]` — commentary, humor, disbelief
  - `[happy]` — positive stories, celebrations
  - `[narration]` — factual context, transitions
  - `[laugh]` / `[chuckle]` — between speakers for natural feel

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

The model (~700 MB) downloads automatically from [`ResembleAI/chatterbox-turbo`](https://huggingface.co/ResembleAI/chatterbox-turbo) on HuggingFace the first time either node runs. Files are stored in the standard ComfyUI models directory:

```
ComfyUI/models/TTS/chatterbox-turbo/
```

The folder is registered with ComfyUI's `folder_paths` system and respects `extra_model_paths.yaml` overrides. No manual download needed.

## Runtime Patches

This node includes three compatibility patches applied at import time:

1. **Perth watermarker** — The `perth` audio watermarking library uses `pkg_resources`, which was removed in Python 3.13. If the real Perth module loads successfully, watermarking works normally. On Python 3.13+ where Perth fails to import, a no-op mock is installed so generation still works — but output audio will **not** be watermarked. This is logged at startup.

2. **s3tokenizer float32** — On numpy 2.x, STFT operations return float64 instead of float32. The patch casts magnitudes back to float32 after STFT.

3. **prepare_conditionals float32** — On numpy 2.x, `librosa.load()` returns float64 arrays. The patch ensures audio arrays are cast to float32 before being passed to the model.

## Requirements

- ComfyUI with CUDA GPU
- Python 3.11+ (tested on 3.13)
- Dependencies: `chatterbox-tts`, `soundfile`, `librosa`, `huggingface_hub`, `perth`
- PyTorch, torchaudio, and numpy are expected from the ComfyUI environment

## Credits

- [Resemble AI](https://www.resemble.ai/) for the ChatterBox model
- [ChatterBox GitHub](https://github.com/resemble-ai/chatterbox) — upstream model repository

## License

MIT

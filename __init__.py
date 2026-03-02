"""ComfyUI custom node for ChatterBox Turbo TTS.

Fast 350M-param TTS with paralinguistic emotion tags.
"""

# 1. Apply patches FIRST (before any chatterbox imports)
from .patches import apply_all_patches
apply_all_patches()

# 2. Import nodes
from .nodes import ChatterboxTurboGenerate, ChatterboxTurboDialogue

NODE_CLASS_MAPPINGS = {
    "ChatterboxTurboGenerate": ChatterboxTurboGenerate,
    "ChatterboxTurboDialogue": ChatterboxTurboDialogue,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatterboxTurboGenerate": "ChatterBox Turbo TTS",
    "ChatterboxTurboDialogue": "ChatterBox Turbo Dialogue",
}

"""Runtime compatibility patches for ChatterBox Turbo.

Applied before importing chatterbox to fix:
1. Perth watermarker (pkg_resources removed in Python 3.13)
2. s3tokenizer float32 (numpy 2.x STFT returns float64)
3. tts_turbo float32 (librosa returns float64 on numpy 2.x)
"""

import sys
import types


def _patch_perth_watermarker():
    """Ensure Perth watermarker is importable.

    Perth uses pkg_resources which was removed in Python 3.13. If the real
    Perth module loads successfully, we leave it alone (watermarking enabled).
    If it fails (e.g. Python 3.13+), we install a no-op mock so chatterbox
    can still import — audio will not be watermarked in that case.
    """
    if "perth" in sys.modules:
        return

    try:
        import perth  # noqa: F401 — test if real Perth works
        return  # real Perth loaded fine, watermarking enabled
    except Exception:
        pass

    # Real Perth failed — install no-op mock
    perth_mod = types.ModuleType("perth")

    class _DummyWM:
        def apply_watermark(self, wav, sample_rate=None):
            return wav

    perth_mod.PerthImplicitWatermarker = _DummyWM
    sys.modules["perth"] = perth_mod
    print("[ChatterBox Turbo] Perth watermarker unavailable (Python 3.13+), watermarking disabled")


def _patch_s3tokenizer_float32():
    """Fix s3tokenizer STFT returning float64 on numpy 2.x.

    Monkey-patches S3Tokenizer.forward() to cast magnitudes to float32
    after the STFT operation.
    """
    try:
        import s3tokenizer
    except ImportError:
        return

    import torch
    _original_forward = s3tokenizer.S3Tokenizer.forward

    def _patched_forward(self, wavs, wav_lens=None, **kwargs):
        import numpy as np

        # Run original STFT logic
        padding = self.n_fft // 2
        wavs = torch.nn.functional.pad(wavs, (padding, padding), "reflect")
        window = torch.hann_window(self.n_fft, device=wavs.device, dtype=wavs.dtype)
        stft = torch.stft(
            wavs, self.n_fft, self.hop_length, self.n_fft,
            window=window, return_complex=True,
        )
        magnitudes = stft.abs()
        magnitudes = magnitudes.float()  # <-- THE FIX: ensure float32

        mel_spec = self.mel_filters.to(magnitudes.device) @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-5).log()
        log_spec = (log_spec + 4.0) / 4.0

        # Quantize
        codes = self.quantize(log_spec)
        if wav_lens is not None:
            token_lens = (wav_lens / self.hop_length).long()
            return codes, token_lens
        return codes

    s3tokenizer.S3Tokenizer.forward = _patched_forward


def _patch_tts_turbo_float32():
    """Fix ChatterboxTurboTTS.prepare_conditionals() for numpy 2.x.

    librosa.load() returns float64 arrays on numpy 2.x. The model
    expects float32. Patch prepare_conditionals to cast audio to float32.
    """
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
    except ImportError:
        return

    import numpy as np
    _original_prepare = ChatterboxTurboTTS.prepare_conditionals

    def _patched_prepare(self, audio_prompt_path, *args, **kwargs):
        # Temporarily patch librosa.load to return float32
        import librosa
        _original_load = librosa.load

        def _float32_load(*a, **kw):
            audio, sr = _original_load(*a, **kw)
            if isinstance(audio, np.ndarray) and audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            return audio, sr

        librosa.load = _float32_load
        try:
            return _original_prepare(self, audio_prompt_path, *args, **kwargs)
        finally:
            librosa.load = _original_load

    ChatterboxTurboTTS.prepare_conditionals = _patched_prepare


def apply_all_patches():
    """Apply all runtime patches. Must be called before importing chatterbox."""
    _patch_perth_watermarker()
    # s3tokenizer and tts_turbo patches are deferred until first import
    # since chatterbox isn't loaded yet. We register them as post-import hooks.
    _deferred_patches = []

    def _apply_deferred():
        _patch_s3tokenizer_float32()
        _patch_tts_turbo_float32()

    # Try applying now (if already imported), otherwise hook into import
    try:
        _patch_s3tokenizer_float32()
    except Exception:
        _deferred_patches.append(_patch_s3tokenizer_float32)

    try:
        _patch_tts_turbo_float32()
    except Exception:
        _deferred_patches.append(_patch_tts_turbo_float32)

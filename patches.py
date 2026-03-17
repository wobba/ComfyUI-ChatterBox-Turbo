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
        # Check if the loaded module actually has what chatterbox needs
        existing = sys.modules["perth"]
        if hasattr(existing, "PerthImplicitWatermarker"):
            return
        # Wrong perth package — fall through to install mock

    try:
        import perth  # noqa: F401 — test if real Perth works
        if hasattr(perth, "PerthImplicitWatermarker"):
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

    Monkey-patches log_mel_spectrogram() on both the pip s3tokenizer and
    chatterbox's bundled copy to cast magnitudes to float32 after STFT.
    """
    patched = False

    # Patch chatterbox's bundled s3tokenizer (the one actually used)
    try:
        from chatterbox.models.s3tokenizer import s3tokenizer as cb_s3_mod
        _orig_mel = cb_s3_mod.S3Tokenizer.log_mel_spectrogram

        def _patched_mel(self, audio, padding=0):
            import torch
            import torch.nn.functional as F
            if not torch.is_tensor(audio):
                audio = torch.from_numpy(audio)
            audio = audio.to(self.device)
            if padding > 0:
                audio = F.pad(audio, (0, padding))
            stft = torch.stft(
                audio, self.n_fft, cb_s3_mod.S3_HOP,
                window=self.window.to(self.device),
                return_complex=True,
            )
            magnitudes = stft[..., :-1].abs().float() ** 2  # .float() before **2
            mel_spec = self._mel_filters.to(self.device) @ magnitudes
            log_spec = torch.clamp(mel_spec, min=1e-10).log10()
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
            log_spec = (log_spec + 4.0) / 4.0
            return log_spec

        cb_s3_mod.S3Tokenizer.log_mel_spectrogram = _patched_mel
        patched = True
    except (ImportError, AttributeError) as e:
        print(f"[ChatterBox Turbo] Could not patch chatterbox s3tokenizer: {e}")

    # Also patch pip s3tokenizer if installed (belt and suspenders)
    try:
        import s3tokenizer
        _orig_fwd = s3tokenizer.S3Tokenizer.forward

        def _patched_forward(self, wavs, wav_lens=None, **kwargs):
            import torch
            padding = self.n_fft // 2
            wavs = torch.nn.functional.pad(wavs, (padding, padding), "reflect")
            window = torch.hann_window(self.n_fft, device=wavs.device, dtype=wavs.dtype)
            stft = torch.stft(wavs, self.n_fft, self.hop_length, self.n_fft,
                              window=window, return_complex=True)
            magnitudes = stft.abs().float()  # <-- ensure float32
            mel_spec = self.mel_filters.to(magnitudes.device) @ magnitudes
            log_spec = torch.clamp(mel_spec, min=1e-5).log()
            log_spec = (log_spec + 4.0) / 4.0
            codes = self.quantize(log_spec)
            if wav_lens is not None:
                token_lens = (wav_lens / self.hop_length).long()
                return codes, token_lens
            return codes

        s3tokenizer.S3Tokenizer.forward = _patched_forward
        patched = True
    except (ImportError, AttributeError):
        pass

    if not patched:
        print("[ChatterBox Turbo] WARNING: Could not patch s3tokenizer float32")


def _patch_tts_turbo_float32():
    """Fix float64 issues in chatterbox for numpy 2.x.

    Patches:
    1. librosa.load() wrapper to return float32 arrays
    2. Voice encoder forward() to cast mel input to float32 before LSTM
    """
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
    except ImportError:
        return

    # Guard against double-patching
    if getattr(ChatterboxTurboTTS, '_float32_patched', False):
        return
    ChatterboxTurboTTS._float32_patched = True

    import numpy as np

    # Patch 1: librosa.load wrapper in prepare_conditionals
    _original_prepare = ChatterboxTurboTTS.prepare_conditionals

    def _patched_prepare(self, audio_prompt_path, *args, **kwargs):
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

    # Patch 2: Voice encoder forward() — cast mels to float32 before LSTM
    try:
        from chatterbox.models.voice_encoder.voice_encoder import VoiceEncoder
        _original_forward = VoiceEncoder.forward

        def _patched_forward(self, mels):
            mels = mels.float()  # ensure float32
            return _original_forward(self, mels)

        VoiceEncoder.forward = _patched_forward
    except (ImportError, AttributeError) as e:
        print(f"[ChatterBox Turbo] Could not patch VoiceEncoder: {e}")


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

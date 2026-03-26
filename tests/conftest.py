"""
Shared test fixtures for batch synthesis tests.
"""
import struct
import wave
from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir():
    """Path to the test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Isolated temporary output directory per test."""
    out = tmp_path / "batch_output"
    out.mkdir()
    return out


@pytest.fixture
def sample_wav(tmp_path):
    """Create a minimal valid WAV file (1 second of silence at 24kHz)."""
    wav_path = tmp_path / "test.wav"
    with wave.open(str(wav_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(struct.pack("<h", 0) * 24000)
    return wav_path

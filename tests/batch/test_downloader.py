"""
Tests for batch.downloader module.
"""
import shutil
import zipfile
from pathlib import Path

import pytest

from batch.downloader import pack_to_zip


class TestPackToZip:
    """Tests for the pack_to_zip function."""

    def test_creates_valid_zip(self, tmp_path, sample_wav):
        # Prepare 3 wav files
        wav_files = []
        for name in ["a.wav", "b.wav", "c.wav"]:
            p = tmp_path / name
            shutil.copy(sample_wav, p)
            wav_files.append(p)

        zip_path = pack_to_zip(wav_files, output_dir=tmp_path / "zips")
        assert zip_path is not None
        assert zip_path.exists()
        assert zip_path.suffix == ".zip"

        with zipfile.ZipFile(zip_path) as zf:
            assert len(zf.namelist()) == 3
            assert set(zf.namelist()) == {"a.wav", "b.wav", "c.wav"}

    def test_preserves_filenames(self, tmp_path, sample_wav):
        p = tmp_path / "my_podcast.wav"
        shutil.copy(sample_wav, p)

        zip_path = pack_to_zip([p], output_dir=tmp_path / "zips")
        with zipfile.ZipFile(zip_path) as zf:
            assert "my_podcast.wav" in zf.namelist()

    def test_empty_list_returns_none(self, tmp_path):
        result = pack_to_zip([], output_dir=tmp_path / "zips")
        assert result is None

    def test_missing_file_skipped(self, tmp_path, sample_wav):
        existing = tmp_path / "exists.wav"
        shutil.copy(sample_wav, existing)
        missing = tmp_path / "does_not_exist.wav"

        zip_path = pack_to_zip(
            [existing, missing], output_dir=tmp_path / "zips"
        )
        assert zip_path is not None
        with zipfile.ZipFile(zip_path) as zf:
            assert len(zf.namelist()) == 1
            assert "exists.wav" in zf.namelist()

    def test_all_missing_returns_none(self, tmp_path):
        result = pack_to_zip(
            [tmp_path / "a.wav", tmp_path / "b.wav"],
            output_dir=tmp_path / "zips",
        )
        assert result is None

    def test_custom_zip_name(self, tmp_path, sample_wav):
        p = tmp_path / "audio" / "test.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(sample_wav, p)

        zip_path = pack_to_zip(
            [p], output_dir=tmp_path / "zips", zip_name="my_batch"
        )
        assert zip_path.name == "my_batch.zip"

    def test_creates_output_dir_if_missing(self, tmp_path, sample_wav):
        p = tmp_path / "audio" / "test.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(sample_wav, p)
        nested_dir = tmp_path / "a" / "b" / "c"

        zip_path = pack_to_zip([p], output_dir=nested_dir)
        assert zip_path is not None
        assert nested_dir.exists()

"""
Audio file packaging utilities for batch download.
"""
import logging
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def pack_to_zip(
    wav_paths: list[Path],
    output_dir: Path,
    zip_name: Optional[str] = None,
) -> Optional[Path]:
    """
    Pack multiple wav files into a single zip archive.

    Args:
        wav_paths: List of wav file paths to include.
        output_dir: Directory to write the zip file.
        zip_name: Optional zip filename (without extension).
                  Defaults to 'batch_{timestamp}'.

    Returns:
        Path to the created zip file, or None if wav_paths is empty.
    """
    # Filter to only existing files
    existing = [p for p in wav_paths if p.exists()]
    if not existing:
        return None

    if zip_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_name = f"batch_{timestamp}"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"{zip_name}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for wav_path in existing:
            zf.write(wav_path, arcname=wav_path.name)

    logger.info(f"Created zip: {zip_path} ({len(existing)} files)")
    return zip_path

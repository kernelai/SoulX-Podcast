"""
TXT file parser for batch podcast synthesis.

Reads .txt files, validates [S1]/[S2] tag format,
and returns structured ParsedScript objects.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ParsedScript:
    """Immutable result of parsing a single txt file."""
    filename: str
    dialogue_text: str
    is_valid: bool
    error_msg: Optional[str] = None


# Valid speaker tag pattern: [S1] through [S4], followed by non-empty content
_SPEAKER_TAG_RE = re.compile(r"^\[S[1-4]\].+")


def _read_file_content(file_path: Path) -> str:
    """
    Read file content with encoding auto-detection.
    Tries UTF-8 BOM -> UTF-8 -> GBK in order.

    Raises:
        ValueError: If no supported encoding can decode the file.
    """
    raw = file_path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError("无法识别文件编码，仅支持 UTF-8 / GBK")


def _validate_lines(lines: list[str]) -> tuple[bool, Optional[str]]:
    """
    Validate that every non-empty line starts with a valid speaker tag.

    Returns:
        (is_valid, error_msg) tuple.
    """
    if not lines:
        return False, "文件内容为空"

    has_any_tag = False
    for i, line in enumerate(lines, start=1):
        if not _SPEAKER_TAG_RE.match(line):
            return False, f"第 {i} 行格式错误：需要 [S1]~[S4] 标签后跟文本内容，当前：{line[:30]}"
        has_any_tag = True

    if not has_any_tag:
        return False, "缺少说话人标签 [S1]~[S4]"

    return True, None


def parse_txt_file(file_path: str | Path) -> ParsedScript:
    """
    Parse a single txt file for batch synthesis.

    Expected format: each non-empty line starts with [S1]~[S4] tag.

    Args:
        file_path: Path to the .txt file.

    Returns:
        ParsedScript with validation result.
    """
    file_path = Path(file_path)
    filename = file_path.name

    # Read file
    try:
        content = _read_file_content(file_path)
    except FileNotFoundError:
        return ParsedScript(
            filename=filename,
            dialogue_text="",
            is_valid=False,
            error_msg=f"文件不存在: {file_path}",
        )
    except ValueError as e:
        return ParsedScript(
            filename=filename,
            dialogue_text="",
            is_valid=False,
            error_msg=str(e),
        )

    # Split into non-empty lines
    lines = [line.strip() for line in content.splitlines()]
    lines = [line for line in lines if line]

    # Validate
    is_valid, error_msg = _validate_lines(lines)
    if not is_valid:
        return ParsedScript(
            filename=filename,
            dialogue_text="",
            is_valid=False,
            error_msg=error_msg,
        )

    # Join lines into single dialogue text with \n separator
    dialogue_text = "\n".join(lines)

    return ParsedScript(
        filename=filename,
        dialogue_text=dialogue_text,
        is_valid=True,
        error_msg=None,
    )


def parse_txt_files(file_paths: list[str | Path]) -> list[ParsedScript]:
    """
    Parse multiple txt files.

    Returns:
        List of ParsedScript, one per file. Invalid files are included
        with is_valid=False so the UI can display error info.
    """
    return [parse_txt_file(p) for p in file_paths]

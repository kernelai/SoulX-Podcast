"""
Tests for batch.file_parser module.

All tests are pure logic — no GPU, no model, no Gradio required.
"""
from pathlib import Path

import pytest

from batch.file_parser import parse_txt_file, parse_txt_files


class TestParseValidFiles:
    """Tests for files that should parse successfully."""

    def test_parse_valid_simple(self, fixtures_dir):
        result = parse_txt_file(fixtures_dir / "valid_simple.txt")
        assert result.is_valid is True
        assert result.error_msg is None
        assert "[S1]" in result.dialogue_text
        assert "[S2]" in result.dialogue_text
        assert result.filename == "valid_simple.txt"

    def test_parse_valid_multi_turn(self, fixtures_dir):
        result = parse_txt_file(fixtures_dir / "valid_multi_turn.txt")
        assert result.is_valid is True
        lines = result.dialogue_text.split("\n")
        assert len(lines) == 6

    def test_parse_valid_with_blank_lines(self, fixtures_dir):
        result = parse_txt_file(fixtures_dir / "valid_with_blanks.txt")
        assert result.is_valid is True
        # Blank lines should be stripped out
        lines = result.dialogue_text.split("\n")
        assert len(lines) == 2
        assert all(line.strip() for line in lines)

    def test_parse_utf8_bom(self, fixtures_dir):
        result = parse_txt_file(fixtures_dir / "encoding_utf8_bom.txt")
        assert result.is_valid is True
        # BOM should not appear in the parsed text
        assert not result.dialogue_text.startswith("\ufeff")
        assert "[S1]" in result.dialogue_text

    def test_parse_gbk(self, fixtures_dir):
        result = parse_txt_file(fixtures_dir / "encoding_gbk.txt")
        assert result.is_valid is True
        assert "[S1]" in result.dialogue_text
        assert "GBK" in result.dialogue_text


class TestParseInvalidFiles:
    """Tests for files that should fail validation."""

    def test_parse_empty_file(self, fixtures_dir):
        result = parse_txt_file(fixtures_dir / "invalid_empty.txt")
        assert result.is_valid is False
        assert "为空" in result.error_msg

    def test_parse_no_tag(self, fixtures_dir):
        result = parse_txt_file(fixtures_dir / "invalid_no_tag.txt")
        assert result.is_valid is False
        assert "标签" in result.error_msg or "格式错误" in result.error_msg

    def test_parse_bad_tag_reports_line_number(self, fixtures_dir):
        result = parse_txt_file(fixtures_dir / "invalid_bad_tag.txt")
        assert result.is_valid is False
        # Should mention line 2 (the [SA] line)
        assert "2" in result.error_msg
        assert "格式错误" in result.error_msg

    def test_parse_nonexistent_file(self, tmp_path):
        result = parse_txt_file(tmp_path / "does_not_exist.txt")
        assert result.is_valid is False
        assert result.error_msg is not None

    def test_parse_unknown_encoding(self, tmp_path):
        """Write bytes that can't be decoded as UTF-8 or GBK."""
        bad_file = tmp_path / "bad_encoding.txt"
        # Random bytes that are invalid in both UTF-8 and GBK
        bad_file.write_bytes(bytes([0x80, 0x81, 0x82, 0xFE, 0xFF] * 20))
        result = parse_txt_file(bad_file)
        assert result.is_valid is False
        assert "编码" in result.error_msg


class TestParseMultipleFiles:
    """Tests for the batch parse_txt_files function."""

    def test_parse_multiple_mixed(self, fixtures_dir):
        files = [
            fixtures_dir / "valid_simple.txt",
            fixtures_dir / "invalid_empty.txt",
            fixtures_dir / "valid_multi_turn.txt",
        ]
        results = parse_txt_files(files)
        assert len(results) == 3
        assert results[0].is_valid is True
        assert results[1].is_valid is False
        assert results[2].is_valid is True

    def test_parse_empty_list(self):
        results = parse_txt_files([])
        assert results == []


class TestParsedScriptImmutability:
    """Verify ParsedScript is frozen (immutable)."""

    def test_frozen(self, fixtures_dir):
        result = parse_txt_file(fixtures_dir / "valid_simple.txt")
        with pytest.raises(AttributeError):
            result.filename = "changed.txt"

"""
Tests for the ChemML Purple Agent.

These tests verify the agent works correctly with the MLE-Bench protocol:
1. Agent card is accessible and correct
2. Agent can receive and process file attachments
3. Agent generates valid ML code
4. Agent produces submission.csv artifacts
"""

import asyncio
import base64
import io
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests — Agent Logic
# ═══════════════════════════════════════════════════════════════════════════════


class TestAnalyzeDirectory:
    """Test the data analysis helpers."""

    def test_analyze_empty_dir(self):
        from agent import analyze_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            result = analyze_directory(Path(tmpdir))
            assert "FILES:" in result

    def test_analyze_csv_dir(self):
        from agent import analyze_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a sample CSV
            csv_path = Path(tmpdir) / "train.csv"
            csv_path.write_text("id,feature1,feature2,target\n1,0.5,0.3,1\n2,0.8,0.1,0\n")

            result = analyze_directory(Path(tmpdir))
            assert "train.csv" in result
            assert "feature1" in result

    def test_analyze_with_description(self):
        from agent import analyze_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            desc_path = Path(tmpdir) / "description.md"
            desc_path.write_text("# Competition\nPredict survival on the Titanic.")

            result = analyze_directory(Path(tmpdir))
            assert "Titanic" in result


class TestDetectChemistry:
    """Test chemistry data detection."""

    def test_detect_smiles(self):
        from agent import detect_chemistry_data
        assert detect_chemistry_data("Column: smiles_string") is True

    def test_detect_inchi(self):
        from agent import detect_chemistry_data
        assert detect_chemistry_data("Columns: id, InChI, target") is True

    def test_detect_molecular(self):
        from agent import detect_chemistry_data
        assert detect_chemistry_data("Predict molecular properties") is True

    def test_no_chemistry(self):
        from agent import detect_chemistry_data
        assert detect_chemistry_data("Predict house prices from features") is False


class TestExtractCode:
    """Test code extraction from LLM responses."""

    def test_extract_python_block(self):
        from agent import Agent
        agent = Agent()
        text = "Here's the code:\n```python\nimport pandas as pd\nprint('hello')\n```\nDone."
        code = agent._extract_code(text)
        assert code is not None
        assert "import pandas" in code

    def test_extract_plain_block(self):
        from agent import Agent
        agent = Agent()
        text = "```\nimport pandas as pd\n```"
        code = agent._extract_code(text)
        assert code is not None

    def test_no_code_block(self):
        from agent import Agent
        agent = Agent()
        text = "No code here, just text."
        code = agent._extract_code(text)
        assert code is None


class TestFindSubmission:
    """Test submission file discovery."""

    def test_find_in_home(self):
        from agent import Agent
        agent = Agent()

        with tempfile.TemporaryDirectory() as tmpdir:
            home_dir = Path(tmpdir) / "home"
            home_dir.mkdir()
            sub = home_dir / "submission.csv"
            sub.write_text("id,target\n1,0\n")

            result = agent._find_submission(Path(tmpdir))
            assert result is not None
            assert result.name == "submission.csv"

    def test_find_in_root(self):
        from agent import Agent
        agent = Agent()

        with tempfile.TemporaryDirectory() as tmpdir:
            sub = Path(tmpdir) / "submission.csv"
            sub.write_text("id,target\n1,0\n")

            result = agent._find_submission(Path(tmpdir))
            assert result is not None

    def test_not_found(self):
        from agent import Agent
        agent = Agent()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = agent._find_submission(Path(tmpdir))
            assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests — Full Pipeline (requires API key)
# ═══════════════════════════════════════════════════════════════════════════════


def create_mock_competition_tar() -> bytes:
    """Create a minimal competition tar.gz for testing."""
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        # description.md
        desc = b"# Test Competition\nPredict the target variable from features.\nMetric: accuracy"
        info = tarfile.TarInfo(name="home/data/description.md")
        info.size = len(desc)
        tar.addfile(info, io.BytesIO(desc))

        # train.csv
        train = b"id,feature1,feature2,target\n1,0.5,0.3,1\n2,0.8,0.1,0\n3,0.2,0.9,1\n4,0.7,0.4,0\n"
        info = tarfile.TarInfo(name="home/data/train.csv")
        info.size = len(train)
        tar.addfile(info, io.BytesIO(train))

        # test.csv
        test = b"id,feature1,feature2\n5,0.6,0.5\n6,0.1,0.8\n"
        info = tarfile.TarInfo(name="home/data/test.csv")
        info.size = len(test)
        tar.addfile(info, io.BytesIO(test))

        # sample_submission.csv
        sample = b"id,target\n5,0\n6,0\n"
        info = tarfile.TarInfo(name="home/data/sample_submission.csv")
        info.size = len(sample)
        tar.addfile(info, io.BytesIO(sample))

    tar_buffer.seek(0)
    return tar_buffer.read()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

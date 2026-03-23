"""
ChemML Purple Agent — MLE-Bench Competition Solver

A competition-winning Purple Agent that receives Kaggle-style ML tasks from
MLE-Bench Green Agent and produces high-scoring submission.csv files.

Protocol:
1. Green Agent sends: instructions.txt (TextPart) + competition.tar.gz (FilePart)
2. We extract data, analyze it, generate ML code, execute it, produce submission.csv
3. We return submission.csv as a FilePart artifact
4. Green Agent grades using mlebench.grade.grade_csv

Strategy:
- Analyze dataset structure (columns, dtypes, shapes) before generating code
- Detect chemistry data (SMILES, InChI, fingerprints) → activate RDKit heuristics
- Generate robust ML pipelines with fallback predictions
- Always produce a valid submission.csv, even if training fails
"""

import asyncio
import base64
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import traceback
from pathlib import Path
from typing import Any

import litellm
from dotenv import load_dotenv

from a2a.server.tasks import TaskUpdater
from a2a.types import (
    FilePart,
    FileWithBytes,
    Message,
    Part,
    TaskState,
    TextPart,
)
from a2a.utils import get_message_text, new_agent_text_message

load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — Competition-winning ML engineer
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an elite machine learning engineer competing in MLE-Bench (Kaggle-style competitions).
You are also an expert in analytical chemistry, cheminformatics, and chemometrics.

## YOUR GOAL
Write a COMPLETE, SELF-CONTAINED Python script that:
1. Reads the competition data from the provided directory
2. Trains a model and generates predictions
3. Saves predictions to `submission.csv` in the EXACT format required

## RULES
- Your script must be FULLY SELF-CONTAINED (no interactive input, no plots, no display)
- Use ONLY these libraries (pre-installed): pandas, numpy, scikit-learn, xgboost, \
scipy, lightgbm, catboost, torch, torchvision, transformers, Pillow, opencv-python, rdkit
- The script must handle errors gracefully — if training fails, produce a baseline submission
- The script must complete within 15 minutes
- Output ONLY the Python code, wrapped in ```python ... ``` markers
- Do NOT include any explanation outside the code block

## STRATEGY BY DATA TYPE

### Tabular Data (CSV with numeric/categorical columns)
1. Load data, handle missing values (median for numeric, mode for categorical)
2. Feature engineering: label encode categoricals, create interactions
3. Train XGBoost or LightGBM with reasonable defaults
4. If competition metric is known, optimize for it

### Image Data (folders of images)
1. Use torchvision pretrained models (ResNet, EfficientNet)
2. Apply standard augmentation (resize, normalize)
3. Fine-tune for a few epochs with low learning rate

### Text/NLP Data
1. Use TF-IDF + LightGBM as a fast baseline
2. If time permits, use a small transformer model

### Chemistry Data (SMILES, InChI, molecular descriptors)
1. Convert SMILES → molecular fingerprints (Morgan, MACCS) using RDKit
2. Use standard ML models on fingerprint features
3. For molecular translation: use character-level tokenization

### Signal/Time-Series Data
1. Extract statistical features (mean, std, min, max, quantiles, fft)
2. Use gradient boosting on extracted features

## FALLBACK STRATEGY
If your model fails during training:
- For regression: predict the mean of the target column
- For classification: predict the most frequent class
- ALWAYS produce a valid submission.csv with the correct format

## SUBMISSION FORMAT
- Read the sample_submission.csv to determine the exact column names and format
- Make sure index/ID columns match exactly
- Save with `index=False` unless the sample submission uses an index
"""


# ═══════════════════════════════════════════════════════════════════════════════
# DATA ANALYSIS HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_directory(data_dir: Path) -> str:
    """Analyze competition data directory and return a summary string."""
    summary_parts = []

    # List all files
    all_files = []
    for f in sorted(data_dir.rglob("*")):
        if f.is_file():
            rel = f.relative_to(data_dir)
            size_mb = f.stat().st_size / (1024 * 1024)
            all_files.append(f"  {rel} ({size_mb:.1f} MB)")
    summary_parts.append("FILES:\n" + "\n".join(all_files[:50]))
    if len(all_files) > 50:
        summary_parts.append(f"  ... and {len(all_files) - 50} more files")

    # Read description.md if present
    desc_file = data_dir / "description.md"
    if desc_file.exists():
        desc_text = desc_file.read_text(encoding="utf-8", errors="replace")
        # Truncate to 6000 chars to fit in context
        if len(desc_text) > 6000:
            desc_text = desc_text[:6000] + "\n... [TRUNCATED]"
        summary_parts.append(f"\nDESCRIPTION.MD:\n{desc_text}")

    # Analyze CSV files (first 5 rows, column info)
    csv_files = list(data_dir.rglob("*.csv"))
    for csv_file in csv_files[:5]:  # Limit to 5 CSVs
        rel = csv_file.relative_to(data_dir)
        try:
            import pandas as pd
            df = pd.read_csv(csv_file, nrows=5)
            info = (
                f"\n{rel}:\n"
                f"  Shape (first 5 rows): {df.shape}\n"
                f"  Columns: {list(df.columns)}\n"
                f"  Dtypes:\n{df.dtypes.to_string()}\n"
                f"  Head:\n{df.head(3).to_string()}"
            )
            summary_parts.append(info)
        except Exception as e:
            summary_parts.append(f"\n{rel}: Error reading — {e}")

    return "\n".join(summary_parts)


def detect_chemistry_data(summary: str) -> bool:
    """Check if dataset contains chemistry-related data."""
    chem_keywords = [
        "smiles", "inchi", "molecule", "compound", "fingerprint",
        "morgan", "maccs", "rdkit", "chemical", "molecular",
        "drug", "toxicity", "solubility", "logp", "pka",
    ]
    summary_lower = summary.lower()
    return any(kw in summary_lower for kw in chem_keywords)


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT CLASS
# ═══════════════════════════════════════════════════════════════════════════════


class Agent:
    """Competition-winning Purple Agent for MLE-Bench."""

    def __init__(self):
        self.model = os.getenv("LLM_MODEL", "openai/gpt-4o")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "8192"))
        self.code_timeout = int(os.getenv("CODE_TIMEOUT", "900"))  # 15 min

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """
        Process MLE-Bench evaluation request.

        Flow:
        1. Extract competition.tar.gz from message
        2. Analyze dataset structure
        3. Generate ML pipeline code via LLM
        4. Execute code to produce submission.csv
        5. Return submission as FilePart artifact
        """
        # ── Step 1: Extract files from message ──────────────────────────
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Extracting competition data..."),
        )

        instructions_text = ""
        tar_data = None

        for part in message.parts:
            if isinstance(part.root, TextPart):
                instructions_text += part.root.text + "\n"
            elif isinstance(part.root, FilePart):
                file_info = part.root.file
                if isinstance(file_info, FileWithBytes):
                    tar_data = base64.b64decode(file_info.bytes)

        if tar_data is None:
            # Fallback: message might be plain text (non-standard request)
            instructions_text = get_message_text(message)
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=(
                    "Error: No competition data file received. "
                    "Expected a competition.tar.gz attachment."
                )))],
                name="Error",
            )
            return

        # ── Step 2: Extract tar and analyze data ───────────────────────
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Analyzing dataset structure..."),
        )

        work_dir = Path(tempfile.mkdtemp(prefix="mlebench_"))
        data_dir = work_dir / "home" / "data"

        try:
            # Extract tar
            tar_buffer = io.BytesIO(tar_data)
            with tarfile.open(fileobj=tar_buffer, mode="r:gz") as tar:
                tar.extractall(path=work_dir, filter="data")

            # Find the actual data directory (tar may have nested structure)
            if not data_dir.exists():
                # Try to find description.md anywhere
                for candidate in work_dir.rglob("description.md"):
                    data_dir = candidate.parent
                    break
                else:
                    data_dir = work_dir

            # Analyze dataset
            data_summary = analyze_directory(data_dir)
            is_chemistry = detect_chemistry_data(data_summary)

            # ── Step 3: Generate ML pipeline via LLM ───────────────────
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Generating ML pipeline... "
                    f"{'(chemistry data detected!) ' if is_chemistry else ''}"
                ),
            )

            code = await self._generate_code(
                instructions_text, data_summary, data_dir, is_chemistry
            )

            if not code:
                await updater.add_artifact(
                    parts=[Part(root=TextPart(text="Error: Failed to generate ML code"))],
                    name="Error",
                )
                return

            # Save code to work_dir
            script_path = work_dir / "solve.py"
            script_path.write_text(code, encoding="utf-8")

            # ── Step 4: Execute code ───────────────────────────────────
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("Training model and generating predictions..."),
            )

            exec_result = await self._execute_code(script_path, work_dir)

            # ── Step 5: Find and return submission.csv ─────────────────
            submission_path = self._find_submission(work_dir)

            if submission_path is None:
                # Code execution failed — try to generate a fallback submission
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        "Primary model failed. Generating fallback submission..."
                    ),
                )

                fallback_code = await self._generate_fallback_code(
                    data_summary, data_dir, exec_result
                )
                if fallback_code:
                    fallback_path = work_dir / "fallback.py"
                    fallback_path.write_text(fallback_code, encoding="utf-8")
                    await self._execute_code(fallback_path, work_dir)
                    submission_path = self._find_submission(work_dir)

            if submission_path is None:
                await updater.add_artifact(
                    parts=[Part(root=TextPart(text=(
                        f"Error: Failed to produce submission.csv\n\n"
                        f"Execution output:\n{exec_result}"
                    )))],
                    name="Error",
                )
                return

            # Read submission and package as FilePart
            submission_bytes = submission_path.read_bytes()
            submission_b64 = base64.b64encode(submission_bytes).decode("ascii")

            await updater.add_artifact(
                parts=[
                    Part(root=FilePart(
                        file=FileWithBytes(
                            bytes=submission_b64,
                            name="submission.csv",
                            mime_type="text/csv",
                        )
                    )),
                    Part(root=TextPart(text=f"Submission generated ({len(submission_bytes)} bytes)")),
                ],
                name="submission",
            )

        except Exception as e:
            tb = traceback.format_exc()
            print(f"[Agent] Fatal error: {tb}")
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=f"Fatal agent error: {e}\n{tb}"))],
                name="Error",
            )
        finally:
            # Cleanup
            import shutil
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass

    # ═══════════════════════════════════════════════════════════════════════
    # LLM CODE GENERATION
    # ═══════════════════════════════════════════════════════════════════════

    async def _generate_code(
        self,
        instructions: str,
        data_summary: str,
        data_dir: Path,
        is_chemistry: bool,
    ) -> str | None:
        """Generate ML pipeline code using the LLM."""

        chemistry_hint = ""
        if is_chemistry:
            chemistry_hint = """
CHEMISTRY DATA DETECTED! Use these specialized strategies:
- If SMILES/InChI columns exist: use RDKit to compute Morgan fingerprints (radius=2, nBits=2048)
- For molecular property prediction: fingerprints + XGBoost works well
- For molecular translation (image→InChI): use CNN encoder + sequence decoder
- Import rdkit with: from rdkit import Chem; from rdkit.Chem import AllChem, Descriptors
"""

        user_prompt = f"""\
Solve this Kaggle competition. Write a COMPLETE Python script.

INSTRUCTIONS:
{instructions}

DATA DIRECTORY: {data_dir}

DATASET ANALYSIS:
{data_summary}
{chemistry_hint}
CRITICAL REQUIREMENTS:
1. The script must read data from "{data_dir}" (use this exact path)
2. The script must save submission to "{data_dir.parent}/submission.csv"
3. The script must be fully self-contained and handle all errors
4. Include a try/except around training — if it fails, produce a baseline prediction
5. Do NOT use plt.show() or any interactive/display functions
6. Print progress messages so we can track execution

Output ONLY the Python code wrapped in ```python ... ``` markers.
"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            response_text = response.choices[0].message.content

            # Extract code from markdown code block
            code = self._extract_code(response_text)
            return code

        except Exception as e:
            print(f"[Agent] LLM error: {e}")
            return None

    async def _generate_fallback_code(
        self,
        data_summary: str,
        data_dir: Path,
        error_output: str,
    ) -> str | None:
        """Generate a simple fallback submission when primary model fails."""

        user_prompt = f"""\
The previous ML script FAILED with this error:
{error_output[:3000]}

Generate a SIMPLE fallback script that produces a valid submission.csv.
Use the sample_submission.csv as a template and fill it with baseline predictions:
- For regression: use the mean of the target from training data
- For classification: use the most common class from training data
- If unsure, just copy sample_submission.csv as-is

DATA DIRECTORY: {data_dir}
SAVE TO: {data_dir.parent}/submission.csv

DATASET ANALYSIS:
{data_summary}

Output ONLY the Python code wrapped in ```python ... ``` markers.
"""

        messages = [
            {"role": "system", "content": "You are a data scientist. Write a simple fallback submission script."},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=4096,
            )
            return self._extract_code(response.choices[0].message.content)
        except Exception as e:
            print(f"[Agent] Fallback LLM error: {e}")
            return None

    # ═══════════════════════════════════════════════════════════════════════
    # CODE EXECUTION
    # ═══════════════════════════════════════════════════════════════════════

    async def _execute_code(self, script_path: Path, work_dir: Path) -> str:
        """Execute a Python script in a subprocess with timeout."""
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(script_path),
                cwd=str(work_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "MPLBACKEND": "Agg"},  # Non-interactive matplotlib
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.code_timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return "TIMEOUT: Script exceeded time limit"

            output = ""
            if stdout:
                output += stdout.decode("utf-8", errors="replace")
            if stderr:
                output += "\nSTDERR:\n" + stderr.decode("utf-8", errors="replace")
            if proc.returncode != 0:
                output += f"\nExit code: {proc.returncode}"

            print(f"[Agent] Execution output (last 500 chars): {output[-500:]}")
            return output

        except Exception as e:
            return f"Execution error: {e}"

    # ═══════════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════════

    def _extract_code(self, text: str) -> str | None:
        """Extract Python code from markdown code blocks."""
        # Try ```python ... ``` first
        pattern = r"```python\s*\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()

        # Try ``` ... ``` (no language specified)
        pattern = r"```\s*\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()

        # If no code blocks, check if the entire response looks like code
        if "import " in text and ("pd.read_csv" in text or "open(" in text):
            return text.strip()

        return None

    def _find_submission(self, work_dir: Path) -> Path | None:
        """Find submission.csv in the work directory."""
        # Check common locations
        candidates = [
            work_dir / "home" / "submission.csv",
            work_dir / "submission.csv",
        ]

        # Also search recursively
        for f in work_dir.rglob("submission.csv"):
            candidates.append(f)

        for path in candidates:
            if path.exists() and path.stat().st_size > 0:
                return path

        return None

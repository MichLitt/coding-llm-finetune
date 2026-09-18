"""Every entry-point script must at least start (`--help`) on a CPU-only machine."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


@pytest.mark.parametrize("script", [
    "prepare_sft_data.py", "check_contamination.py", "sft_train.py", "dpo_train.py",
    "generate_dpo_pairs.py", "run_eval.py", "compare_runs.py",
])
def test_help_runs(script):
    result = subprocess.run([sys.executable, str(SCRIPTS / script), "--help"],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-500:]
    assert "Usage" in result.stdout


def test_removed_targeted_scripts_stay_removed():
    for name in ("generate_targeted_data.py", "run_targeted_eval.py", "run_humaneval.py"):
        assert not (SCRIPTS / name).exists(), name

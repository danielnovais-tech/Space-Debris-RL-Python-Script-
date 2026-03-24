import subprocess
import sys

from space_debris_rl.cli import main


def test_cli_help_runs(capsys):
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0

    out = capsys.readouterr().out
    assert "space-debris-rl" in out

def test_help_shows_commands() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "space_debris_rl.cli", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    out = proc.stdout
    assert "train" in out
    assert "evaluate" in out
    assert "distributed" in out
    assert "hierarchical" in out
    assert "train-hierarchical" in out
    assert "federated" in out

def test_evaluate_help_shows_hierarchical_flags() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "space_debris_rl.cli", "evaluate", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    out = proc.stdout
    assert "--hierarchical" in out
    assert "--manager-model" in out
    assert "--worker-model" in out
    assert "--learned-worker" in out


def test_train_help_shows_hierarchical_flag() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "space_debris_rl.cli", "train", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    out = proc.stdout
    assert "--hierarchical" in out

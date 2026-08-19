"""Tests for the CLI surface (SPEC §10, Task 0 acceptance).

The command set is checked against SPEC §10 explicitly, so that the documented
interface and the real one cannot drift apart unnoticed.
"""

from __future__ import annotations

import pytest

from culturesim.cli import (
    EXIT_FAILED,
    EXIT_NOT_IMPLEMENTED,
    EXIT_OK,
    EXIT_USAGE,
    build_parser,
    main,
)


def test_help_exits_cleanly(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == EXIT_OK
    assert "culture-sim" in capsys.readouterr().out


def test_version_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == EXIT_OK


def test_no_command_prints_usage(capsys) -> None:
    assert main([]) == EXIT_USAGE
    assert "usage" in capsys.readouterr().err.lower()


def test_command_set_matches_spec_10() -> None:
    parser = build_parser()
    subparsers = next(
        action
        for action in parser._subparsers._group_actions
        if action.choices  # type: ignore[union-attr]
    )
    assert set(subparsers.choices) == {
        "simulate",
        "fingerprint",
        "fit",
        "validate",
        "report",
    }


def test_fit_has_coarse_and_sbi_stages() -> None:
    parser = build_parser()
    subparsers = next(
        action
        for action in parser._subparsers._group_actions
        if action.choices  # type: ignore[union-attr]
    )
    fit = subparsers.choices["fit"]
    stages = next(action for action in fit._subparsers._group_actions if action.choices)  # type: ignore[union-attr]
    assert set(stages.choices) == {"coarse", "sbi"}


def test_fit_without_a_stage_prints_usage(capsys) -> None:
    assert main(["fit"]) == EXIT_USAGE
    assert "usage" in capsys.readouterr().err.lower()


@pytest.fixture
def cli_workspace(tmp_path, monkeypatch, poisson_recording):
    """A working directory with the configs and a real recording to act on.

    The configs are symlinked so `culture-sim` resolves them the way it would in a
    checkout; the recording exists so the `fingerprint` command reaches
    `compute_fingerprint` instead of stopping at a missing input file.
    """
    from culturesim.config import DEFAULT_CONFIG_DIR

    (tmp_path / "configs").symlink_to(DEFAULT_CONFIG_DIR, target_is_directory=True)
    poisson_recording.to_hdf5(tmp_path / "run.h5")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize(
    "argv",
    [
        ["simulate", "--out", "run.h5"],
        ["fingerprint", "--input", "run.h5", "--out", "fp.json"],
        ["fit", "coarse", "--data", "wagenaar2006", "--out", "coarse.json"],
        ["fit", "sbi", "--data", "wagenaar2006", "--n-sims", "3000", "--out", "posterior.pkl"],
        ["validate", "--posterior", "posterior.pkl", "--test", "all"],
        ["report", "--out", "report.html"],
    ],
)
def test_documented_invocations_reach_a_handler(argv: list[str], cli_workspace, capsys) -> None:
    """Every SPEC §10 invocation parses, validates its inputs, and dispatches.

    They exit "not implemented" rather than 0 until their task lands. That exit code is
    distinct from failure on purpose: a half-built pipeline should not look broken, and
    a broken one should not look unbuilt.
    """
    if argv[0] == "validate":
        # validate requires an on-disk posterior; create a stub so Task 7 is reached.
        (cli_workspace / "posterior.pkl").write_bytes(b"stub")
    exit_code = main(argv)
    if argv[0] in {"simulate", "fingerprint"}:
        assert exit_code == EXIT_OK
        assert "wrote" in capsys.readouterr().out
    elif argv[0] == "fit":
        # Wagenaar loader / scale requirements fail without data/raw (exit 1).
        assert exit_code == EXIT_FAILED
        assert "error:" in capsys.readouterr().err
    else:
        assert exit_code == EXIT_NOT_IMPLEMENTED
        assert "not implemented" in capsys.readouterr().err


def test_fit_without_a_cached_dataset_is_a_missing_file(cli_workspace, capsys) -> None:
    """The loader exists; the bytes have to be fetched separately (data/raw is gitignored)."""
    from culturesim.cli import EXIT_FAILED, main

    exit_code = main(["fit", "coarse", "--data", "wagenaar2006", "--out", "c.json"])
    assert exit_code == EXIT_FAILED
    message = capsys.readouterr().err
    assert "error:" in message
    assert "fetch_wagenaar" in message


def test_fit_loads_a_local_wagenaar_file_then_needs_scale(cli_workspace, capsys, tmp_path) -> None:
    """A single recording is not enough to z-score (SPEC §8.1, §14)."""
    import bz2

    from culturesim.cli import EXIT_FAILED, main

    path = tmp_path / "1-1-14.spk.txt.bz2"
    path.write_bytes(bz2.compress(b"0.1 0\n0.2 1\n"))
    exit_code = main(["fit", "coarse", "--data", str(path), "--out", "c.json"])
    assert exit_code == EXIT_FAILED
    message = capsys.readouterr().err
    assert "error:" in message
    assert "across-culture" in message


def test_missing_required_argument_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["simulate"])
    assert exit_info.value.code == EXIT_USAGE


def test_sbi_warns_below_the_spec_minimum(cli_workspace, capsys) -> None:
    """SPEC §8.3 requires >= 3000 simulations for the Task 6 acceptance criterion."""
    main(["fit", "sbi", "--data", "x", "--n-sims", "10", "--out", "p.pkl"])
    assert "below the 3000 required" in capsys.readouterr().err


def test_bad_config_path_is_reported_not_traced(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    exit_code = main(["simulate", "--config", "nope.yaml", "--out", "run.h5"])
    assert exit_code != EXIT_OK
    assert "error:" in capsys.readouterr().err


def test_help_does_not_import_brian2_or_torch() -> None:
    """`culture-sim --help` must stay fast, so heavy imports live inside handlers."""
    import subprocess
    import sys

    code = (
        "import sys; from culturesim.cli import build_parser; build_parser().format_help(); "
        "print(sorted(m for m in ('brian2', 'torch', 'sbi') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"

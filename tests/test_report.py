"""Tests for the Task 8 HTML report (no Brian2)."""

from __future__ import annotations

import json
from pathlib import Path

from culturesim.report import gather_report_inputs, render_html, write_report


def test_write_report_from_fixture_artefacts(tmp_path: Path):
    artefact = tmp_path / "output"
    figures = tmp_path / "figures"
    artefact.mkdir()
    figures.mkdir()
    (artefact / "posterior.summary.json").write_text(
        json.dumps(
            {
                "summary": {
                    "names": ["p_conn", "rate_bg"],
                    "mean": [0.2, 4.0],
                    "std": [0.1, 1.0],
                    "prior_std": [0.2, 5.0],
                    "identified": {"p_conn": False, "rate_bg": True},
                    "correlations": [[1.0, 0.1], [0.1, 1.0]],
                },
                "identified": ["rate_bg"],
                "unidentified": ["p_conn"],
                "n_simulations": 10,
                "n_excluded": 1,
                "posterior_predictive": {
                    "bracketed_fraction": 0.5,
                    "n_bracketed": 1,
                    "n_checked": 2,
                    "coverage_interval": [5, 95],
                },
            }
        ),
        encoding="utf-8",
    )
    (artefact / "validation.json").write_text(
        json.dumps(
            {
                "any_passed": False,
                "heldout": {"passed": False, "notes": "heldout fail", "fraction_within_z": 0.2},
                "cross_culture": {"passed": False, "notes": "no culture B"},
                "perturbation": {
                    "passed": False,
                    "notes": "silent stim",
                    "psth_correlation": 0.0,
                    "amplitude_curve_error": 1.0,
                    "burst_probability_error": 0.3,
                },
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "report.html"
    write_report(
        out,
        artefact_dir=artefact,
        figure_dir=figures,
        regenerate_posterior_figures=True,
    )
    text = out.read_text(encoding="utf-8")
    assert "Purpose and scope" in text
    assert "rate_bg" in text
    assert "does not" in text
    assert "heldout fail" in text
    assert (figures / "task6_posterior_marginals.png").exists()


def test_write_report_archives_under_reports_dir(tmp_path: Path):
    from culturesim.report import write_report

    artefact = tmp_path / "output"
    figures = tmp_path / "figures"
    reports = tmp_path / "reports"
    artefact.mkdir()
    figures.mkdir()
    (artefact / "posterior.summary.json").write_text(
        json.dumps(
            {
                "summary": {
                    "names": ["rate_bg"],
                    "mean": [4.0],
                    "std": [1.0],
                    "prior_std": [5.0],
                    "identified": {"rate_bg": True},
                    "correlations": [[1.0]],
                },
                "identified": ["rate_bg"],
                "unidentified": [],
                "n_simulations": 3,
                "n_excluded": 0,
            }
        ),
        encoding="utf-8",
    )
    out = write_report(
        artefact_dir=artefact,
        figure_dir=figures,
        report_dir=reports,
        label="milestone",
        regenerate_posterior_figures=False,
    )
    assert out.parent == reports
    assert "milestone" in out.name
    assert (reports / "latest.html").exists()
    history = (reports / "history.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(history) == 1
    assert "milestone" in history[0]


def test_gather_report_inputs_notes_missing(tmp_path: Path):
    inputs = gather_report_inputs(artefact_dir=tmp_path, figure_dir=tmp_path)
    assert "posterior.summary.json" in inputs.missing
    html = render_html(inputs)
    assert "Missing" in html or "missing" in html.lower()

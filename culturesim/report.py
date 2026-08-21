"""HTML report generator (SPEC §10, §13 Task 8, §15).

One command rebuilds a single document that states what the model reproduces,
which parameters are identified, whether validation passed, and — crucially —
what it is *not* valid for (SPEC §0).
"""

from __future__ import annotations

import base64
import html
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import REPO_ROOT
from .manifest import git_commit, package_versions

__all__ = [
    "ReportInputs",
    "DEFAULT_REPORT_DIR",
    "archive_report_path",
    "gather_report_inputs",
    "load_scope_markdown",
    "write_report",
]

DEFAULT_ARTEFACT_DIR = REPO_ROOT / "output"
DEFAULT_FIGURE_DIR = REPO_ROOT / "figures"
# Cwd-relative so `culture-sim report` from a checkout (or a test workspace) archives
# next to the caller rather than always under the package install root.
DEFAULT_REPORT_DIR = Path("reports")
HISTORY_NAME = "history.jsonl"
LATEST_NAME = "latest.html"


def archive_report_path(
    *,
    label: str | None = None,
    when: datetime | None = None,
    report_dir: Path | None = None,
) -> Path:
    """Timestamped path under ``reports/`` for a new historical entry."""
    report_dir = _resolve_report_dir(report_dir)
    when = when or datetime.now(UTC)
    stamp = when.strftime("%Y-%m-%dT%H%M%SZ")
    slug = _slugify(label) if label else "report"
    return report_dir / f"{stamp}_{slug}.html"


def _resolve_report_dir(report_dir: Path | None) -> Path:
    path = Path(report_dir) if report_dir is not None else DEFAULT_REPORT_DIR
    return path if path.is_absolute() else Path.cwd() / path


def _slugify(label: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", label.strip().lower()).strip("-")
    return slug[:60] or "report"


def load_scope_markdown() -> str:
    """SPEC.md §0 body, from the heading through the CL-SDK relationship section."""
    text = (REPO_ROOT / "SPEC.md").read_text(encoding="utf-8")
    match = re.search(
        r"^## 0\. Purpose and Scope\n(?P<body>.*?)^---\n",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise RuntimeError("SPEC.md §0 not found; refuse to invent the scope statement")
    body = match.group("body").strip()
    # Drop the leading blank after the heading; keep subsections as written.
    return body


FIGURE_CANDIDATES = (
    "task1_bursts_vs_static.png",
    "task2_60_vs_1024.png",
    "task5_distance_landscape.png",
    "task6_posterior_marginals.png",
    "task6_posterior_correlations.png",
)


@dataclass(frozen=True)
class ReportInputs:
    artefact_dir: Path
    figure_dir: Path
    posterior_summary: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    coarse: dict[str, Any] | None = None
    task6_status: dict[str, Any] | None = None
    posterior_path: Path | None = None
    figure_paths: dict[str, Path] = field(default_factory=dict)
    missing: tuple[str, ...] = ()


def gather_report_inputs(
    *,
    artefact_dir: Path | None = None,
    figure_dir: Path | None = None,
    posterior: Path | None = None,
) -> ReportInputs:
    """Load whatever report artefacts already exist on disk."""
    artefact_dir = Path(artefact_dir) if artefact_dir is not None else DEFAULT_ARTEFACT_DIR
    figure_dir = Path(figure_dir) if figure_dir is not None else DEFAULT_FIGURE_DIR
    missing: list[str] = []

    summary_path = artefact_dir / "posterior.summary.json"
    validation_path = artefact_dir / "validation.json"
    coarse_path = artefact_dir / "coarse.json"
    status_path = artefact_dir / "task6_status.json"
    posterior_path = Path(posterior) if posterior is not None else artefact_dir / "posterior.pkl"

    posterior_summary = _load_json(summary_path, missing, "posterior.summary.json")
    validation = _load_json(validation_path, missing, "validation.json")
    coarse = _load_json(coarse_path, missing, "coarse.json")
    task6_status = _load_json(status_path, missing, "task6_status.json")
    if not posterior_path.exists():
        missing.append(str(posterior_path.name))
        posterior_path = None

    figures: dict[str, Path] = {}
    for name in FIGURE_CANDIDATES:
        path = figure_dir / name
        if path.exists():
            figures[name] = path
    return ReportInputs(
        artefact_dir=artefact_dir,
        figure_dir=figure_dir,
        posterior_summary=posterior_summary,
        validation=validation,
        coarse=coarse,
        task6_status=task6_status,
        posterior_path=posterior_path,
        figure_paths=figures,
        missing=tuple(missing),
    )


def write_report(
    out: Path | None = None,
    *,
    artefact_dir: Path | None = None,
    figure_dir: Path | None = None,
    posterior: Path | None = None,
    regenerate_posterior_figures: bool = True,
    label: str | None = None,
    report_dir: Path | None = None,
    update_latest: bool = True,
    record_history: bool = True,
) -> Path:
    """Build the HTML report and write it to ``out``.

    When ``out`` is omitted, a timestamped file is written under ``reports/``
    (see :func:`archive_report_path`). A copy is also written to
    ``reports/latest.html``, and one line is appended to ``reports/history.jsonl``.
    """
    report_dir = _resolve_report_dir(report_dir)
    out = Path(out) if out is not None else archive_report_path(label=label, report_dir=report_dir)
    inputs = gather_report_inputs(
        artefact_dir=artefact_dir,
        figure_dir=figure_dir,
        posterior=posterior,
    )
    if regenerate_posterior_figures and inputs.posterior_summary is not None:
        generated = _write_posterior_figures(
            inputs.posterior_summary,
            directory=inputs.figure_dir,
        )
        figure_paths = dict(inputs.figure_paths)
        figure_paths.update(generated)
        inputs = ReportInputs(
            artefact_dir=inputs.artefact_dir,
            figure_dir=inputs.figure_dir,
            posterior_summary=inputs.posterior_summary,
            validation=inputs.validation,
            coarse=inputs.coarse,
            task6_status=inputs.task6_status,
            posterior_path=inputs.posterior_path,
            figure_paths=figure_paths,
            missing=inputs.missing,
        )

    document = render_html(inputs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="utf-8")

    if update_latest and _is_under(out, report_dir):
        latest = report_dir / LATEST_NAME
        latest.write_text(document, encoding="utf-8")

    if record_history and _is_under(out, report_dir):
        _append_history(
            report_dir / HISTORY_NAME,
            {
                "written_at": datetime.now(UTC).isoformat(),
                "path": str(out.relative_to(REPO_ROOT)) if _is_under(out, REPO_ROOT) else str(out),
                "label": label,
                "git_commit": git_commit(),
                "missing_artefacts": list(inputs.missing),
            },
        )
    return out


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _append_history(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def render_html(inputs: ReportInputs) -> str:
    """Assemble the full HTML document as a string."""
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    commit = git_commit()
    versions = package_versions()
    executive = _executive_summary(inputs)
    body = "\n".join(
        [
            _section_scope(),
            _section_executive(executive),
            _section_fit(inputs),
            _section_posterior(inputs),
            _section_validation(inputs),
            _section_figures(inputs),
            _section_limitations(inputs),
            _section_provenance(commit, versions, inputs),
        ]
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>culture-sim report</title>
<style>
:root {{
  --ink: #1a1a1a;
  --muted: #5a5a5a;
  --line: #d8d8d8;
  --bg: #fafaf8;
  --card: #ffffff;
  --pass: #1b6b3a;
  --fail: #8b1e1e;
  --accent: #245b8a;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: "IBM Plex Sans", "Source Sans 3", "Segoe UI", sans-serif;
  color: var(--ink);
  background: var(--bg);
  line-height: 1.5;
}}
header {{
  background: linear-gradient(135deg, #0f2740 0%, #245b8a 55%, #3d7ea6 100%);
  color: #f5f8fb;
  padding: 2.5rem 1.5rem 2rem;
}}
header h1 {{ margin: 0 0 0.35rem; font-size: 1.85rem; font-weight: 650; }}
header p {{ margin: 0; max-width: 46rem; opacity: 0.92; }}
main {{ max-width: 920px; margin: 0 auto; padding: 1.5rem; }}
section {{
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 1.25rem 1.35rem 1.4rem;
  margin: 0 0 1.1rem;
}}
h2 {{ margin: 0 0 0.75rem; font-size: 1.2rem; color: var(--accent); }}
h3 {{ margin: 1rem 0 0.4rem; font-size: 1.02rem; }}
p, li {{ color: var(--ink); }}
.muted {{ color: var(--muted); }}
.badge {{
  display: inline-block;
  font-size: 0.75rem;
  font-weight: 650;
  letter-spacing: 0.03em;
  padding: 0.15rem 0.5rem;
  border-radius: 999px;
  border: 1px solid currentColor;
}}
.badge.pass {{ color: var(--pass); }}
.badge.fail {{ color: var(--fail); }}
.badge.warn {{ color: #8a5a00; }}
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.92rem;
  margin: 0.5rem 0 0.25rem;
}}
th, td {{
  text-align: left;
  padding: 0.4rem 0.55rem;
  border-bottom: 1px solid var(--line);
  vertical-align: top;
}}
th {{ color: var(--muted); font-weight: 600; }}
figure {{ margin: 1rem 0 0.25rem; }}
figure img {{
  max-width: 100%;
  height: auto;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fff;
}}
figcaption {{ font-size: 0.85rem; color: var(--muted); margin-top: 0.35rem; }}
code, .mono {{ font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 0.88em; }}
ul.compact {{ margin: 0.35rem 0; padding-left: 1.2rem; }}
.callout {{
  border-left: 3px solid var(--fail);
  background: #fbf4f4;
  padding: 0.75rem 0.9rem;
  margin: 0.75rem 0;
}}
.callout.ok {{ border-left-color: var(--pass); background: #f3f8f4; }}
footer {{
  max-width: 920px;
  margin: 0 auto 2rem;
  padding: 0 1.5rem;
  color: var(--muted);
  font-size: 0.85rem;
}}
</style>
</head>
<body>
<header>
  <h1>culture-sim — first fitted report</h1>
  <p>SPEC §15 definition of done. Generated {html.escape(generated)}
  from commit <span class="mono">{html.escape(commit)}</span>.</p>
</header>
<main>
{body}
</main>
<footer>
  Regenerated with
  <span class="mono">culture-sim report</span>
  (archived under <span class="mono">reports/</span>).
  Artefacts under <span class="mono">output/</span>;
  figures under <span class="mono">figures/</span>.
</footer>
</body>
</html>
"""


def _executive_summary(inputs: ReportInputs) -> dict[str, Any]:
    identified: list[str] = []
    unidentified: list[str] = []
    ppc = None
    if inputs.posterior_summary:
        identified = list(inputs.posterior_summary.get("identified") or [])
        unidentified = list(inputs.posterior_summary.get("unidentified") or [])
        ppc = inputs.posterior_summary.get("posterior_predictive")

    validation = inputs.validation or {}
    heldout = validation.get("heldout") or {}
    cross = validation.get("cross_culture") or {}
    pert = validation.get("perturbation") or {}

    coarse = inputs.coarse or {}
    improvement = coarse.get("improvement_fraction")
    dataset = "wagenaar2006 (culture 1-1-14, DIV 14 dense spontaneous)"

    reproduces = (
        f"Coarse fingerprint distance improved by {100 * float(improvement):.0f}% "
        f"vs the Task 1 baseline (artefact `coarse.json`)."
        if improvement is not None
        else "Coarse-fit artefact missing; spontaneous fingerprint match not summarised here."
    )
    if ppc:
        bracketed = ppc.get("bracketed_fraction")
        reproduces += (
            f" Posterior predictive check bracketed {100 * float(bracketed):.0f}% "
            "of fingerprint statistics (5–95% band)."
            if bracketed is not None
            else ""
        )

    return {
        "dataset": dataset,
        "reproduces": reproduces,
        "identified": identified,
        "unidentified": unidentified,
        "heldout_pass": bool(heldout.get("passed")),
        "cross_pass": bool(cross.get("passed")),
        "pert_pass": bool(pert.get("passed")),
        "evoked": "does not" if not pert.get("passed") else "does",
        "any_validation_pass": bool(validation.get("any_passed")),
    }


def _section_scope() -> str:
    return f"""<section id="scope">
<h2>0. Purpose and scope (SPEC §0, verbatim)</h2>
{_markdownish_to_html(load_scope_markdown())}
</section>"""


def _section_executive(summary: dict[str, Any]) -> str:
    id_s = ", ".join(f"<code>{html.escape(n)}</code>" for n in summary["identified"]) or "none"
    uid_s = ", ".join(f"<code>{html.escape(n)}</code>" for n in summary["unidentified"]) or "none"
    return f"""<section id="executive">
<h2>Definition of done (SPEC §15)</h2>
<div class="callout">
<p><strong>Dataset D:</strong> {html.escape(summary["dataset"])}</p>
<p><strong>What the model reproduces (quantified):</strong> {html.escape(summary["reproduces"])}</p>
<p><strong>Identified parameters:</strong> {id_s}</p>
<p><strong>Unidentified parameters:</strong> {uid_s}</p>
<p><strong>Evoked responses:</strong> the model <strong>{html.escape(summary["evoked"])}</strong>
predict evoked responses under the spontaneous-fit parameters (SPEC §9.3).</p>
<p><strong>Validity:</strong> it is <strong>not</strong> valid for the non-goals in SPEC §0
(morphology, glia, development, waves, 3-D organoids, etc.), and it is not yet validated
as a general culture model (cross-culture incomplete; held-out and perturbation failed).</p>
</div>
<p class="muted">Downstream use for closed-loop timing, decoder drift, or criticality control
should treat this build as an incomplete test bench until validation improves.</p>
</section>"""


def _section_fit(inputs: ReportInputs) -> str:
    coarse = inputs.coarse
    if coarse is None:
        return """<section id="coarse"><h2>Coarse fit (Task 5)</h2>
<p class="muted">Missing <code>output/coarse.json</code>.</p></section>"""
    best = coarse.get("best") or {}
    rows = "".join(
        f"<tr><td><code>{html.escape(k)}</code></td><td>{_fmt(v)}</td></tr>"
        for k, v in best.items()
    )
    return f"""<section id="coarse">
<h2>Coarse fit (Task 5)</h2>
<p>Baseline distance {_fmt(coarse.get("baseline_distance"))} → best
{_fmt(coarse.get("best_distance"))}
({100 * float(coarse.get("improvement_fraction") or 0):.0f}% improvement;
meets ≥50% criterion: <strong>{bool(coarse.get("meets_50_percent"))}</strong>).
Duration {_fmt(coarse.get("duration_s"))} s biological; scale from
{html.escape(str(coarse.get("scale_n_cultures")))} cultures.</p>
<table><thead><tr><th>parameter</th><th>point estimate</th></tr></thead>
<tbody>{rows}</tbody></table>
<p class="muted">A point estimate is a starting point for SBI, not a result.</p>
</section>"""


def _section_posterior(inputs: ReportInputs) -> str:
    summary_doc = inputs.posterior_summary
    if summary_doc is None:
        return """<section id="posterior"><h2>SBI posterior (Task 6)</h2>
<p class="muted">Missing <code>output/posterior.summary.json</code>.</p></section>"""
    summary = summary_doc.get("summary") or summary_doc
    names = list(summary.get("names") or [])
    mean = list(summary.get("mean") or [])
    std = list(summary.get("std") or [])
    prior_std = list(summary.get("prior_std") or [])
    identified = summary.get("identified") or {}
    rows = []
    for i, name in enumerate(names):
        ratio = (
            float(std[i]) / float(prior_std[i])
            if i < len(std) and i < len(prior_std) and float(prior_std[i]) > 0
            else float("nan")
        )
        flag = identified.get(name) if isinstance(identified, dict) else None
        if flag is None:
            flag = name in (summary_doc.get("identified") or [])
        badge = (
            '<span class="badge pass">identified</span>'
            if flag
            else '<span class="badge fail">flat</span>'
        )
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(name)}</code></td>"
            f"<td>{_fmt(mean[i] if i < len(mean) else None)}</td>"
            f"<td>{_fmt(std[i] if i < len(std) else None)}</td>"
            f"<td>{_fmt(ratio)}</td>"
            f"<td>{badge}</td>"
            "</tr>"
        )
    ppc = summary_doc.get("posterior_predictive") or {}
    n_sims = summary_doc.get("n_simulations")
    n_excl = summary_doc.get("n_excluded")
    return f"""<section id="posterior">
<h2>SBI posterior (Task 6)</h2>
<p>Kept simulations: <strong>{html.escape(str(n_sims))}</strong>
(excluded {html.escape(str(n_excl))}). Identifiability threshold: posterior std
&lt; 0.5 × prior std.</p>
<p>PPC: bracketed {html.escape(str(ppc.get("n_bracketed")))}/
{html.escape(str(ppc.get("n_checked")))} statistics
({100 * float(ppc.get("bracketed_fraction") or 0):.0f}% in the
{html.escape(str(ppc.get("coverage_interval")))} percentile band).</p>
<table>
<thead><tr><th>parameter</th><th>mean</th><th>std</th>
<th>std / prior_std</th><th>status</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
{_figure_block(inputs, "task6_posterior_marginals.png", "Posterior marginals vs prior width")}
{_figure_block(inputs, "task6_posterior_correlations.png", "Pairwise posterior correlations")}
</section>"""


def _section_validation(inputs: ReportInputs) -> str:
    validation = inputs.validation
    if validation is None:
        return """<section id="validation"><h2>Validation (Task 7)</h2>
<p class="muted">Missing <code>output/validation.json</code>. Run
<code>culture-sim validate --posterior output/posterior.pkl</code>.</p></section>"""

    def block(title: str, key: str) -> str:
        data = validation.get(key) or {}
        passed = bool(data.get("passed"))
        badge = (
            '<span class="badge pass">PASS</span>'
            if passed
            else '<span class="badge fail">FAIL</span>'
        )
        notes = html.escape(str(data.get("notes") or ""))
        extra = ""
        if key == "heldout" and data.get("fraction_within_z") is not None:
            extra = (
                f"<p>Fraction of held-out scalars with |z|&lt;2: "
                f"<strong>{100 * float(data['fraction_within_z']):.0f}%</strong>.</p>"
            )
        if key == "perturbation":
            extra = (
                f"<p>PSTH correlation {_fmt(data.get('psth_correlation'))}; "
                f"amplitude-curve error {_fmt(data.get('amplitude_curve_error'))}; "
                f"burst-probability error {_fmt(data.get('burst_probability_error'))}.</p>"
            )
        return f"""<h3>{html.escape(title)} {badge}</h3>
<p>{notes}</p>{extra}"""

    return f"""<section id="validation">
<h2>Validation suite (Task 7 / SPEC §9)</h2>
<p>All three tests are reported regardless of outcome.</p>
{block("Held-out statistics (§9.1)", "heldout")}
{block("Cross-culture posterior overlap (§9.2)", "cross_culture")}
{block("Perturbation / evoked response (§9.3)", "perturbation")}
</section>"""


def _section_figures(inputs: ReportInputs) -> str:
    blocks = [
        _figure_block(inputs, "task1_bursts_vs_static.png", "Task 1 — STP vs static synapses"),
        _figure_block(inputs, "task2_60_vs_1024.png", "Task 2 — 60 vs 1024 electrode observation"),
        _figure_block(inputs, "task5_distance_landscape.png", "Task 5 — coarse distance landscape"),
    ]
    present = [b for b in blocks if b]
    if not present:
        return """<section id="figures"><h2>Figures</h2>
<p class="muted">No figure PNGs found under <code>figures/</code>.</p></section>"""
    return f"""<section id="figures">
<h2>Supporting figures</h2>
{"".join(present)}
</section>"""


def _section_limitations(inputs: ReportInputs) -> str:
    return f"""<section id="limitations">
<h2>Stated limitations</h2>
<ul class="compact">
<li>Simulation duration for the SBI campaign was 60 s biological vs ~45 min Wagenaar
sessions — underpowers burst/IBI structure (see <code>docs/SBI_REFIT.md</code>).</li>
<li>Only <code>rate_bg</code> clears the identifiability threshold; STP and connectivity
parameters remain weakly constrained.</li>
<li>Cross-culture overlap needs an independent posterior for culture B.</li>
<li>Perturbation test failed: short stim simulation produced no electrode spikes while
the real Wagenaar stim recording shows evoked structure.</li>
<li>Statistics are electrode-level only (virtual MEA bottleneck). Neuron-level
comparisons are invalid.</li>
<li>Fingerprint order is frozen at version 1.0.0; adding statistics requires a version
bump and a re-fit.</li>
<li>Missing artefacts at generation time:
{", ".join(f"<code>{html.escape(m)}</code>" for m in inputs.missing) or "none"}.</li>
</ul>
</section>"""


def _section_provenance(
    commit: str,
    versions: dict[str, str],
    inputs: ReportInputs,
) -> str:
    ver_rows = "".join(
        f"<tr><td><code>{html.escape(k)}</code></td><td>{html.escape(v)}</td></tr>"
        for k, v in sorted(versions.items())
    )
    return f"""<section id="provenance">
<h2>Provenance</h2>
<p>Git commit: <code>{html.escape(commit)}</code>. Artefact directory:
<code>{html.escape(str(inputs.artefact_dir))}</code>.</p>
<table><thead><tr><th>package</th><th>version</th></tr></thead>
<tbody>{ver_rows}</tbody></table>
</section>"""


def _figure_block(inputs: ReportInputs, name: str, caption: str) -> str:
    path = inputs.figure_paths.get(name)
    if path is None or not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"""<figure>
<img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}"/>
<figcaption>{html.escape(caption)} (<code>{html.escape(name)}</code>)</figcaption>
</figure>"""


def _write_posterior_figures(
    summary_doc: dict[str, Any],
    *,
    directory: Path,
) -> dict[str, Path]:
    from .figures import apply_style, save_figure

    apply_style()
    import matplotlib.pyplot as plt

    summary = summary_doc.get("summary") or summary_doc
    names = list(summary.get("names") or [])
    mean = np.asarray(summary.get("mean"), dtype=np.float64)
    std = np.asarray(summary.get("std"), dtype=np.float64)
    prior_std = np.asarray(summary.get("prior_std"), dtype=np.float64)
    identified = summary.get("identified") or {}
    if isinstance(identified, dict):
        is_id = [bool(identified.get(n, False)) for n in names]
    else:
        id_set = set(summary_doc.get("identified") or [])
        is_id = [n in id_set for n in names]

    directory = Path(directory)
    out: dict[str, Path] = {}

    fig, axes = plt.subplots(2, 4, figsize=(10.5, 4.8), constrained_layout=True)
    for ax, name, _mu, sigma, pstd, flagged in zip(
        axes.ravel(), names, mean, std, prior_std, is_id, strict=False
    ):
        ax.barh(
            [0, 1],
            [float(pstd), float(sigma)],
            color=["#b8c4ce", "#2f6f9f" if flagged else "#a35454"],
            height=0.55,
        )
        ax.set_yticks([0, 1], ["prior σ", "post σ"])
        ax.set_title(name, loc="left", fontsize=9)
        ax.axvline(0.5 * float(pstd), color="0.35", ls="--", lw=0.8)
        ax.set_xlabel("std")
    fig.suptitle("Posterior vs prior width (dashed = 0.5 × prior σ)", fontsize=11)
    path = save_figure(fig, "task6_posterior_marginals", directory=directory)
    plt.close(fig)
    out[path.name] = path

    corr = np.asarray(summary.get("correlations"), dtype=np.float64)
    if corr.ndim == 2 and corr.shape[0] == len(names):
        fig, ax = plt.subplots(figsize=(5.8, 5.0), constrained_layout=True)
        image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
        ax.set_yticks(range(len(names)), names)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Posterior pairwise correlations", loc="left")
        path = save_figure(fig, "task6_posterior_correlations", directory=directory)
        plt.close(fig)
        out[path.name] = path
    return out


def _load_json(path: Path, missing: list[str], label: str) -> dict[str, Any] | None:
    if not path.exists():
        missing.append(label)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if not np.isfinite(number):
        return "—"
    if abs(number) >= 100 or (abs(number) > 0 and abs(number) < 0.01):
        return f"{number:.3g}"
    return f"{number:.3f}"


def _markdownish_to_html(text: str) -> str:
    """Tiny subset converter for the frozen scope block (no external Markdown dep)."""
    chunks: list[str] = []
    lines = text.strip().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("### "):
            chunks.append(f"<h3>{html.escape(line[4:])}</h3>")
            i += 1
            continue
        if line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{_inline(lines[i][2:])}</li>")
                i += 1
            chunks.append("<ul class='compact'>" + "".join(items) + "</ul>")
            continue
        if re.match(r"^\d+\. ", line):
            items = []
            while i < len(lines) and re.match(r"^\d+\. ", lines[i]):
                items.append(f"<li>{_inline(re.sub(r'^\\d+\\. ', '', lines[i]))}</li>")
                i += 1
            chunks.append("<ol class='compact'>" + "".join(items) + "</ol>")
            continue
        if line.strip() == "":
            i += 1
            continue
        para = [line]
        i += 1
        while (
            i < len(lines)
            and lines[i].strip()
            and not lines[i].startswith(("### ", "- "))
            and not re.match(r"^\d+\. ", lines[i])
        ):
            para.append(lines[i])
            i += 1
        chunks.append(f"<p>{_inline(' '.join(para))}</p>")
    return "\n".join(chunks)


def _inline(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    return escaped

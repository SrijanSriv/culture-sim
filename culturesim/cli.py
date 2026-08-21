"""Command-line interface (SPEC §10).

The command set is exactly the one in SPEC §10 -- no extras, so that the documented
interface and the real one cannot drift apart.

Every command writes a run manifest next to its output (SPEC §11). Heavy imports
(Brian2, torch, sbi) happen inside handlers rather than at module level, so
``culture-sim --help`` stays fast.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_NOT_IMPLEMENTED = 3
EXIT_FAILED = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="culture-sim",
        description=(
            "In-silico dissociated neuronal culture on an MEA. Reproduces a defined "
            "list of MEA statistics with quantified parameter uncertainty; see README "
            "§0 for what it does not model."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"culture-sim {__version__}")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase log verbosity (repeatable)",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # -- simulate ---------------------------------------------------------
    simulate = subparsers.add_parser(
        "simulate",
        help="run one simulation and write an electrode-level recording",
    )
    simulate.add_argument("--config", type=Path, default=Path("configs/model_default.yaml"))
    simulate.add_argument("--observation", type=Path, default=Path("configs/observation.yaml"))
    simulate.add_argument("--out", type=Path, required=True, help="output HDF5 path")
    simulate.add_argument("--duration", type=float, default=None, help="override duration, seconds")
    simulate.add_argument("--seed", type=int, default=None, help="override the master seed")
    simulate.add_argument(
        "--static-synapses",
        action="store_true",
        help="ablation: disable short-term plasticity (SPEC §13, Task 1)",
    )
    simulate.set_defaults(handler=_cmd_simulate)

    # -- fingerprint ------------------------------------------------------
    fingerprint = subparsers.add_parser(
        "fingerprint",
        help="compute the summary statistic vector for a recording",
    )
    fingerprint.add_argument("--input", type=Path, required=True, help="recording HDF5 path")
    fingerprint.add_argument("--out", type=Path, required=True, help="output JSON path")
    fingerprint.add_argument("--config", type=Path, default=Path("configs/fingerprint.yaml"))
    fingerprint.add_argument("--seed", type=int, default=None)
    fingerprint.set_defaults(handler=_cmd_fingerprint)

    # -- fit --------------------------------------------------------------
    fit = subparsers.add_parser("fit", help="fit the model to a real dataset")
    fit_sub = fit.add_subparsers(dest="stage", metavar="<stage>")

    fit_coarse = fit_sub.add_parser("coarse", help="grid search then local optimisation")
    fit_coarse.add_argument("--data", required=True, help="dataset key or path")
    fit_coarse.add_argument("--out", type=Path, required=True)
    fit_coarse.add_argument("--config", type=Path, default=Path("configs/model_default.yaml"))
    fit_coarse.add_argument("--seed", type=int, default=None)
    fit_coarse.set_defaults(handler=_cmd_fit_coarse)

    fit_sbi = fit_sub.add_parser("sbi", help="SNPE-C posterior over the 8 free parameters")
    fit_sbi.add_argument("--data", required=True, help="dataset key or path")
    fit_sbi.add_argument("--out", type=Path, required=True)
    fit_sbi.add_argument("--config", type=Path, default=Path("configs/fit_sbi.yaml"))
    fit_sbi.add_argument(
        "--n-sims",
        type=int,
        default=None,
        help="number of prior draws (SPEC §8.3 requires >= 3000)",
    )
    fit_sbi.add_argument("--seed", type=int, default=None)
    fit_sbi.add_argument(
        "--detach",
        action="store_true",
        help=(
            "start the SBI campaign in the background and return immediately; "
            "progress is in output/task6_status.json and the README Task 6 row"
        ),
    )
    fit_sbi.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore an existing checkpoint and start the training set from scratch",
    )
    fit_sbi.set_defaults(handler=_cmd_fit_sbi)
    fit.set_defaults(handler=_require_stage(fit))

    # -- validate ---------------------------------------------------------
    validate = subparsers.add_parser("validate", help="run the validation suite (SPEC §9)")
    validate.add_argument("--posterior", type=Path, required=True)
    validate.add_argument(
        "--test",
        default="all",
        choices=["all", "heldout", "cross_culture", "perturbation"],
    )
    validate.add_argument("--out", type=Path, default=Path("validation.json"))
    validate.add_argument("--seed", type=int, default=None)
    validate.set_defaults(handler=_cmd_validate)

    # -- report -----------------------------------------------------------
    report = subparsers.add_parser("report", help="regenerate the full HTML report")
    report.add_argument(
        "--out",
        type=Path,
        default=None,
        help="explicit output path; default is reports/<timestamp>_<label>.html",
    )
    report.add_argument(
        "--label",
        type=str,
        default=None,
        help="short slug for the archived filename (default: report)",
    )
    report.add_argument(
        "--posterior",
        type=Path,
        default=None,
        help="posterior to report; omit to report only what is already on disk",
    )
    report.set_defaults(handler=_cmd_report)

    return parser


def _require_stage(parser: argparse.ArgumentParser):
    def handler(_args: argparse.Namespace) -> int:
        parser.print_help(sys.stderr)
        return EXIT_USAGE

    return handler


# -- handlers -------------------------------------------------------------
# Each is thin on purpose: argument parsing here, science in the modules.


def _cmd_simulate(args: argparse.Namespace) -> int:
    from .config import load_config, merge_overrides
    from .manifest import record_run
    from .model.params import ModelParams
    from .model.runner import RunRequest, run_one
    from .observation.virtual_mea import ObservationConfig

    started = time.time()
    overrides: dict[str, Any] = {}
    if args.duration is not None:
        overrides["simulation.duration_s"] = args.duration
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.static_synapses:
        overrides["simulation.static_synapses"] = True

    model_config = merge_overrides(load_config(args.config), overrides)
    observation_config = load_config(args.observation)
    params = ModelParams.from_config(model_config)
    ObservationConfig.from_config(observation_config)  # fail fast on a bad layout

    result = run_one(RunRequest(params=params, observation_config=observation_config))
    result.recording.to_hdf5(args.out)
    record_run(
        command="simulate",
        configs={"model": model_config, "observation": observation_config},
        master_seed=params.seed,
        started_at=started,
        output=str(args.out),
        n_spikes=result.recording.n_spikes,
    ).write(args.out.with_suffix(".manifest.json"))
    print(f"wrote {args.out} ({result.recording!r})")
    return EXIT_OK


def _cmd_fingerprint(args: argparse.Namespace) -> int:
    from .manifest import record_run
    from .stats.fingerprint import FingerprintSpec, compute_fingerprint
    from .stats.spiketrains import load_recording

    started = time.time()
    recording = load_recording(args.input)
    spec = FingerprintSpec.load(args.config)
    fingerprint = compute_fingerprint(recording, spec)
    fingerprint.write_json(args.out)
    record_run(
        command="fingerprint",
        configs={"fingerprint": {"version": spec.version, "names_sha256": spec.names_sha256}},
        master_seed=args.seed if args.seed is not None else 0,
        started_at=started,
        input=str(args.input),
        output=str(args.out),
    ).write(args.out.with_suffix(".manifest.json"))
    print(f"wrote {args.out} ({fingerprint!r})")
    return EXIT_OK


def _load_target_fingerprint(data: str) -> Any:
    """Resolve ``--data`` to a real fingerprint (Task 4).

    A path that exists is loaded directly. A dataset key goes through the verified
    loader; missing local bytes are a usage error, not an unimplemented task.
    """
    from .data.loaders import DATASETS, load_wagenaar
    from .stats.fingerprint import compute_fingerprint
    from .stats.spiketrains import load_recording

    path = Path(data)
    if path.exists():
        suffix = path.name.lower()
        if suffix.endswith(".h5") or suffix.endswith(".hdf5"):
            recording = load_recording(path)
        else:
            recording = load_wagenaar(path)
        return compute_fingerprint(recording)

    info = DATASETS.get(data)
    if info is None:
        raise FileNotFoundError(f"unknown dataset or missing file: {data}")
    if info.access != "public":
        raise NotImplementedError(
            f"dataset {data!r} has access state {info.access!r}. SPEC §7 requires "
            f"verifying that {info.url} resolves and reading its data-availability "
            "statement before a loader is written. Do not substitute another dataset."
        )
    if data == "wagenaar2006":
        return compute_fingerprint(load_wagenaar())
    raise NotImplementedError(
        f"loader for {data!r} is not written; Wagenaar is the Task 4 fitting target"
    )


def _cmd_fit_coarse(args: argparse.Namespace) -> int:
    from .config import load_config
    from .fit.coarse import coarse_fit, scale_from_wagenaar
    from .manifest import record_run
    from .model.params import ModelParams

    started = time.time()
    base = ModelParams.from_config(load_config(args.config))
    if args.seed is not None:
        from dataclasses import replace

        base = replace(base, seed=int(args.seed))
    target = _load_target_fingerprint(args.data)
    scale = scale_from_wagenaar(fetch=False)
    figure_path = Path("figures") / "task5_distance_landscape.png"
    result = coarse_fit(
        target=target,
        base=base,
        scale=scale,
        figure_path=figure_path,
    )
    result.write_json(args.out)
    record_run(
        command="fit coarse",
        configs={"model": base.to_config()},
        master_seed=base.seed,
        started_at=started,
        output=str(args.out),
    ).write(args.out.with_suffix(".manifest.json"))
    print(
        f"wrote {args.out}  baseline={result.baseline_distance:.3g}  "
        f"best={result.best_distance:.3g}  improvement={result.improvement_fraction:.0%}"
    )
    return EXIT_OK


def _cmd_fit_sbi(args: argparse.Namespace) -> int:
    from .config import load_config
    from .fit.sbi_fit import run_sbi_fit
    from .fit.task_status import DEFAULT_STATUS_PATH, mark_running
    from .manifest import record_run
    from .model.params import ModelParams
    from .model.runner import default_workers

    config = load_config(args.config)
    n_sims = args.n_sims or int(config["inference"]["n_simulations"])
    if n_sims < 3000:
        print(
            f"warning: --n-sims {n_sims} is below the 3000 required by SPEC §8.3; "
            "the posterior will not meet the Task 6 acceptance criterion",
            file=sys.stderr,
        )

    if args.detach:
        return _detach_sbi(args, config, n_sims)

    started = time.time()
    observed = _load_target_fingerprint(args.data)
    base = ModelParams.load(config["simulator"]["model_config"])
    if args.seed is not None:
        from dataclasses import replace

        base = replace(base, seed=int(args.seed))

    duration_s = float(config["simulator"].get("duration_s", base.simulation.duration_s))
    mark_running(
        n_simulations=n_sims,
        duration_s=duration_s,
        batch_size=max(default_workers(), 8),
        out=args.out,
        log=args.out.with_suffix(".log"),
        checkpoint=args.out.with_suffix(".checkpoint.npz"),
        message="foreground run",
    )
    result = run_sbi_fit(
        observed=observed,
        base=base,
        config=config,
        n_simulations=n_sims,
        out=args.out,
        status_path=DEFAULT_STATUS_PATH,
        resume=not args.no_resume,
    )
    record_run(
        command="fit sbi",
        configs={"fit_sbi": config, "model": base.to_config()},
        master_seed=base.seed,
        started_at=started,
        output=str(args.out),
        n_simulations=result.n_simulations,
        n_excluded=result.n_excluded,
    ).write(args.out.with_suffix(".manifest.json"))
    print(
        f"wrote {args.out}  kept={result.n_simulations} excluded={result.n_excluded}  "
        f"identified={list(result.summary.identified_names())}  "
        f"unidentified={list(result.summary.unidentified_names())}"
    )
    return EXIT_OK


def _detach_sbi(args: argparse.Namespace, config: dict[str, Any], n_sims: int) -> int:
    """Spawn a background SBI worker and return immediately."""
    import os
    import subprocess

    from .fit.task_status import mark_running
    from .model.params import ModelParams
    from .model.runner import default_workers

    # Fail fast on missing data before detaching.
    _load_target_fingerprint(args.data)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix(".log")
    status_path = Path("output/task6_status.json")
    base = ModelParams.load(config["simulator"]["model_config"])
    duration_s = float(config["simulator"].get("duration_s", base.simulation.duration_s))

    # Do not Path.resolve() sys.executable: on macOS the venv python is a symlink
    # into Frameworks, and resolve() would look for culture-sim next to the system
    # interpreter instead of inside .venv/bin.
    venv_bin = Path(sys.executable).parent
    culture_sim = venv_bin / "culture-sim"
    if not culture_sim.exists():
        raise FileNotFoundError(
            f"cannot find culture-sim in {venv_bin}; is the package installed editable?"
        )
    worker_argv = [
        str(culture_sim),
        "fit",
        "sbi",
        "--data",
        str(args.data),
        "--out",
        str(out),
        "--config",
        str(args.config),
        "--n-sims",
        str(n_sims),
    ]
    if args.seed is not None:
        worker_argv.extend(["--seed", str(args.seed)])
    if args.no_resume:
        worker_argv.append("--no-resume")

    mark_running(
        n_simulations=n_sims,
        duration_s=duration_s,
        batch_size=max(default_workers(), 8),
        out=out,
        log=log_path,
        checkpoint=out.with_suffix(".checkpoint.npz"),
        message="detached; waiting for worker",
        path=status_path,
    )

    log_handle = log_path.open("a", encoding="utf-8")
    log_handle.write(f"\n--- detach launch {time.strftime('%Y-%m-%dT%H:%M:%S')} ---\n")
    log_handle.write(" ".join(worker_argv) + "\n")
    log_handle.flush()
    process = subprocess.Popen(  # noqa: S603 - argv is built from our CLI
        worker_argv,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd=str(Path.cwd()),
        env={**os.environ, "MPLBACKEND": "Agg", "HDF5_USE_FILE_LOCKING": "FALSE"},
    )
    print(
        f"started Task 6 SBI in the background (pid {process.pid})\n"
        f"  log:    {log_path}\n"
        f"  status: {status_path}\n"
        f"  check:  .venv/bin/python scripts/check_task6.py\n"
        f"The README Task 6 row updates as batches finish; leave the machine alone."
    )
    return EXIT_OK


def _cmd_validate(args: argparse.Namespace) -> int:
    import pickle

    from .validate.suite import run_validation, update_readme_validation

    if not Path(args.posterior).exists():
        raise FileNotFoundError(args.posterior)
    seed = int(args.seed) if args.seed is not None else 0
    tests = ("all",) if args.test == "all" else (args.test,)
    try:
        report = run_validation(Path(args.posterior), tests=tests, seed=seed)
    except (OSError, pickle.UnpicklingError, TypeError) as exc:
        raise ValueError(f"could not load posterior {args.posterior}: {exc}") from exc
    out = Path(args.out)
    report.save(out)
    update_readme_validation(report)
    print(f"wrote {out}", flush=True)
    # Always exit 0 once the suite has run: SPEC §9 says report failures, not hide them.
    return EXIT_OK


def _cmd_report(args: argparse.Namespace) -> int:
    from .report import write_report

    out = write_report(
        Path(args.out) if args.out is not None else None,
        posterior=Path(args.posterior) if args.posterior is not None else None,
        label=args.label,
    )
    print(f"wrote {out}", flush=True)
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "handler", None) is None:
        parser.print_help(sys.stderr)
        return EXIT_USAGE
    try:
        return int(args.handler(args))
    except NotImplementedError as exc:
        # Not yet built is a distinct outcome from broken, and the exit code says so.
        print(f"not implemented: {exc}", file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

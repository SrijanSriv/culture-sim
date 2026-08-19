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
    report.add_argument("--out", type=Path, required=True)
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
    from .fit.sbi_fit import simulate_training_set, train_posterior
    from .model.params import ModelParams

    config = load_config(args.config)
    n_sims = args.n_sims or int(config["inference"]["n_simulations"])
    if n_sims < 3000:
        print(
            f"warning: --n-sims {n_sims} is below the 3000 required by SPEC §8.3; "
            "the posterior will not meet the Task 6 acceptance criterion",
            file=sys.stderr,
        )
    observed = _load_target_fingerprint(args.data)
    base = ModelParams.load(config["simulator"]["model_config"])
    theta, fingerprints, _ = simulate_training_set(base, base.prior, n_sims)
    train_posterior(theta, fingerprints, observed, config)
    return EXIT_OK


def _cmd_validate(args: argparse.Namespace) -> int:
    from .fit.sbi_fit import SBIResult

    # The posterior is the input to all three tests (SPEC §9), so it loads first.
    SBIResult.load(args.posterior)
    raise NotImplementedError("Task 7 (SPEC §9) -- the validation suite")


def _cmd_report(args: argparse.Namespace) -> int:
    raise NotImplementedError("Task 8 (SPEC §10, §15) -- `culture-sim report`")


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

"""Command line entry point: ``python -m synthdata <command>``.

Four commands cover the whole workflow: profile a source table, generate a
synthetic copy with one of the two generators, evaluate a generated copy, or run
the whole pipeline and write the artefacts. Every command prints a small text
report so a run can be checked by eye without opening a notebook.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .config import AppConfig, load_config, to_ctgan_config
from .copula import GaussianCopula
from .ctgan import CTGANLite
from .evaluate import evaluate, format_report
from .profile import describe_profile, profile_frame
from .source import SourceSpec, make_source_table, read_source_csv, write_source_csv

DEFAULT_SOURCE_FILE: str = "source.csv"
DEFAULT_SYNTHETIC_FILE: str = "synthetic.csv"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser shared by every command."""
    parser = argparse.ArgumentParser(prog="synthdata", description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile_parser = subparsers.add_parser("profile", help="profile a source table")
    profile_parser.add_argument("--source", default=None, help="CSV to profile; generated if omitted")

    generate_parser = subparsers.add_parser("generate", help="generate a synthetic table")
    generate_parser.add_argument("--generator", choices=("copula", "ctgan"), default=None)
    generate_parser.add_argument("--rows", type=int, default=None)
    generate_parser.add_argument("--out", default=None)

    evaluate_parser = subparsers.add_parser("evaluate", help="score a synthetic table")
    evaluate_parser.add_argument("--source", default=DEFAULT_SOURCE_FILE)
    evaluate_parser.add_argument("--synthetic", default=DEFAULT_SYNTHETIC_FILE)
    evaluate_parser.add_argument("--json", action="store_true", help="print the summary as JSON")

    subparsers.add_parser("run-all", help="profile, generate with both generators, evaluate")
    subparsers.add_parser("config", help="print the effective configuration")
    return parser


def resolve_source(args: argparse.Namespace, config: AppConfig) -> pd.DataFrame:
    """Load the source CSV when one is given, otherwise build it from the config."""
    if getattr(args, "source", None) and Path(args.source).exists():
        return read_source_csv(args.source)
    spec = SourceSpec(n_rows=config.source.n_rows, seed=config.source.seed)
    return make_source_table(spec)


def generate_copula(frame: pd.DataFrame, config: AppConfig, n_rows: int) -> pd.DataFrame:
    """Fit and sample the Gaussian copula."""
    model = GaussianCopula(n_bins=config.copula.n_bins).fit(frame)
    return model.sample(n_rows, seed=config.evaluation.seed)


def generate_ctgan(frame: pd.DataFrame, config: AppConfig, n_rows: int) -> pd.DataFrame:
    """Fit and sample CTGAN-lite."""
    model = CTGANLite(config=to_ctgan_config(config.ctgan)).fit(frame)
    return model.sample(n_rows, seed=config.evaluation.seed)


def cmd_profile(args: argparse.Namespace, config: AppConfig) -> int:
    """Print the column profile of the source table."""
    frame = resolve_source(args, config)
    profile = profile_frame(frame)
    print(describe_profile(profile))
    return 0


def cmd_generate(args: argparse.Namespace, config: AppConfig) -> int:
    """Generate a synthetic table and write it to disk."""
    frame = resolve_source(args, config)
    generator = args.generator or config.generator
    n_rows = args.rows or config.evaluation.n_synthetic_rows
    synthetic = generate_copula(frame, config, n_rows) if generator == "copula" else generate_ctgan(frame, config, n_rows)
    output = Path(args.out or DEFAULT_SYNTHETIC_FILE)
    synthetic.to_csv(output, index=False)
    print(f"generator={generator} rows={len(synthetic)} columns={len(synthetic.columns)} -> {output}")
    return 0


def cmd_evaluate(args: argparse.Namespace, config: AppConfig) -> int:
    """Evaluate an existing synthetic table against its source."""
    real = read_source_csv(args.source)
    synthetic = read_source_csv(args.synthetic)
    result = evaluate(real, synthetic)
    if args.json:
        print(json.dumps(result.summary(), indent=2))
    else:
        print(format_report(result))
    return 0


def cmd_run_all(args: argparse.Namespace, config: AppConfig) -> int:
    """Run the full pipeline and persist every artefact."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = make_source_table(SourceSpec(n_rows=config.source.n_rows, seed=config.source.seed))
    write_source_csv(frame, str(output_dir / DEFAULT_SOURCE_FILE))
    print(describe_profile(profile_frame(frame)))
    for generator, factory in (("copula", generate_copula), ("ctgan", generate_ctgan)):
        synthetic = factory(frame, config, config.evaluation.n_synthetic_rows)
        path = output_dir / f"synthetic-{generator}.csv"
        synthetic.to_csv(path, index=False)
        result = evaluate(frame, synthetic)
        print(f"\n[{generator}] -> {path}")
        print(format_report(result))
        (output_dir / f"report-{generator}.json").write_text(
            json.dumps(result.summary(), indent=2), encoding="utf-8"
        )
    return 0


def cmd_config(args: argparse.Namespace, config: AppConfig) -> int:
    """Print the effective configuration as JSON."""
    print(
        json.dumps(
            {
                "generator": config.generator,
                "output_dir": config.output_dir,
                "source": config.source.__dict__,
                "copula": config.copula.__dict__,
                "ctgan": config.ctgan.__dict__,
                "evaluation": config.evaluation.__dict__,
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, load config, dispatch the command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    handlers = {
        "profile": cmd_profile,
        "generate": cmd_generate,
        "evaluate": cmd_evaluate,
        "run-all": cmd_run_all,
        "config": cmd_config,
    }
    return handlers[args.command](args, config)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

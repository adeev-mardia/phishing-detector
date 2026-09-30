"""Command-line interface for the phishing detector.

Installed as the `phishing-detector` console script (see pyproject.toml).

Examples:
    phishing-detector train
    phishing-detector train --csv real_uci_dataset.csv
    phishing-detector url http://paypa1-secure-login.tk/verify
    phishing-detector email suspicious_message.eml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import generate_synthetic_dataset
from .email_heuristics import analyze_email
from .model import DEFAULT_MODEL_PATH, predict_url, save_model, train_model


def _cmd_train(args: argparse.Namespace) -> int:
    clf, metrics, df = train_model(
        csv_path=args.csv,
        n_per_class=args.n_per_class,
        seed=args.seed,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
    )
    model_path = save_model(clf, Path(args.model_path))
    print(f"Trained on {'external CSV: ' + args.csv if args.csv else 'synthetic dataset'}")
    print(f"Dataset size: {len(df)} examples ({df['label'].sum()} phishing / {(df['label'] == 0).sum()} legitimate)")
    print()
    print(metrics.pretty())
    print(f"Model saved to: {model_path}")
    if args.metrics_out:
        Path(args.metrics_out).write_text(json.dumps(metrics.as_dict(), indent=2))
        print(f"Metrics JSON saved to: {args.metrics_out}")
    return 0


def _cmd_url(args: argparse.Namespace) -> int:
    try:
        result = predict_url(args.url, model_path=Path(args.model_path))
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"URL:        {result['url']}")
        print(f"Prediction: {result['prediction'].upper()}")
        if result["phishing_probability"] is not None:
            print(f"Phishing probability: {result['phishing_probability']:.4f}")
        if args.verbose:
            print("\nFeatures:")
            for k, v in result["features"].items():
                print(f"  {k}: {v}")
    return 0


def _cmd_email(args: argparse.Namespace) -> int:
    raw = Path(args.file).read_text(errors="replace")
    analysis = analyze_email(raw_email=raw)
    if args.json:
        print(json.dumps(analysis.as_dict(), indent=2))
    else:
        print(analysis.pretty())
    return 0


def _cmd_generate_dataset(args: argparse.Namespace) -> int:
    df = generate_synthetic_dataset(n_per_class=args.n_per_class, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phishing-detector", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train the URL phishing classifier")
    p_train.add_argument("--csv", default=None, help="Path to a real external CSV dataset (UCI/Kaggle-style) to train on instead of the synthetic generator")
    p_train.add_argument("--n-per-class", type=int, default=800, help="Synthetic examples per class (ignored with --csv)")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--test-size", type=float, default=0.2)
    p_train.add_argument("--n-estimators", type=int, default=300)
    p_train.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    p_train.add_argument("--metrics-out", default=None, help="Optional path to save metrics as JSON")
    p_train.set_defaults(func=_cmd_train)

    p_url = sub.add_parser("url", help="Classify a single URL")
    p_url.add_argument("url")
    p_url.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    p_url.add_argument("--json", action="store_true")
    p_url.add_argument("--verbose", "-v", action="store_true", help="Print all extracted features")
    p_url.set_defaults(func=_cmd_url)

    p_email = sub.add_parser("email", help="Analyze an .eml file for phishing signals")
    p_email.add_argument("file")
    p_email.add_argument("--json", action="store_true")
    p_email.set_defaults(func=_cmd_email)

    p_gen = sub.add_parser("generate-dataset", help="Write the synthetic dataset to a CSV file")
    p_gen.add_argument("--out", default="synthetic_dataset.csv")
    p_gen.add_argument("--n-per-class", type=int, default=800)
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.set_defaults(func=_cmd_generate_dataset)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

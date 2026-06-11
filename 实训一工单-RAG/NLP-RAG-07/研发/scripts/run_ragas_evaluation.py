from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.ragas_eval import DEFAULT_RAGAS_METRICS, RagasEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ragas evaluation for a golden-set CSV in this project.")
    parser.add_argument("--input-csv", required=True, help="CSV path containing question and gold answer columns.")
    parser.add_argument("--output-csv", default="", help="Optional CSV output path.")
    parser.add_argument("--dataset-jsonl", default="", help="Optional JSONL output path.")
    parser.add_argument("--summary-json", default="", help="Optional JSON summary path.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k citations used for each question.")
    parser.add_argument("--corpus", choices=["default", "uploaded"], default="default", help="Target corpus.")
    parser.add_argument("--upload-id", default="", help="Required when corpus=uploaded and you want a fixed upload.")
    parser.add_argument("--use-llm", choices=["true", "false"], default="true", help="Whether to evaluate generated answers.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_RAGAS_METRICS),
        help="ragas metrics to compute.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=60, help="Timeout passed to the ragas judge model.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluator = RagasEvaluator(project_root=PROJECT_ROOT)
    summary = evaluator.evaluate_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        dataset_jsonl=args.dataset_jsonl,
        summary_json=args.summary_json,
        top_k=args.top_k,
        corpus=args.corpus,
        upload_id=args.upload_id,
        use_llm=args.use_llm.lower() == "true",
        metrics=list(args.metrics),
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

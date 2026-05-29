from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.collectors.doctor_nhc import NhcDoctorCollector
from scripts.collectors.lawyer_manual import LawyerManualCollector
from scripts.collectors.history_wikipedia import WikipediaHistoryCollector
from scripts.collectors.history_zgbk import ZgbkHistoryCollector


async def _main_async(args: argparse.Namespace) -> None:
    role_id = args.role_id
    if role_id == "doctor_01":
        collector = NhcDoctorCollector(contact_email=args.contact_email, max_docs=args.max_docs)
    elif role_id == "lawyer_01":
        collector = LawyerManualCollector(max_docs=args.max_docs)
    elif role_id == "history_01":
        registry = json.loads((ROOT_DIR / "scripts" / "source_registry.json").read_text(encoding="utf-8"))
        collector_name = registry["history_01"]["collector"]
        if collector_name == "history_wikipedia":
            collector = WikipediaHistoryCollector(contact_email=args.contact_email, max_docs=args.max_docs)
        else:
            collector = ZgbkHistoryCollector(contact_email=args.contact_email, max_docs=args.max_docs)
    else:
        raise ValueError(f"Unsupported role for collector: {role_id}")

    docs = await collector.collect()
    print(f"{role_id} collected {len(docs)} documents")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-id", required=True, choices=["lawyer_01", "doctor_01", "history_01"])
    parser.add_argument("--contact-email", required=False, default="")
    parser.add_argument("--max-docs", type=int, default=20)
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()

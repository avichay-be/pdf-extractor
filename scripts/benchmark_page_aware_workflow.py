#!/usr/bin/env python3
"""
Benchmark the retained page-aware extraction service on one or more PDFs.

Examples:
    python3 scripts/benchmark_page_aware_workflow.py data/sample.pdf
    python3 scripts/benchmark_page_aware_workflow.py data/sample.pdf --repeat 3 --warmup 1
    python3 scripts/benchmark_page_aware_workflow.py data/sample.pdf --profiles baseline no_polish no_validation
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import settings
from src.core.logging import setup_logging
from src.services.page_aware_extraction_service import build_page_aware_extraction_service


PROFILE_DESCRIPTIONS = {
    "baseline": "Current defaults",
    "no_polish": "Disable Gemini polish",
    "no_validation": "Disable smart validation",
    "fastest": "Disable both validation and polish",
}


@contextmanager
def temporary_settings(*, polish_enabled: bool | None = None):
    original_polish = settings.SMART_EXTRACTION_POLISH_ENABLED
    try:
        if polish_enabled is not None:
            settings.SMART_EXTRACTION_POLISH_ENABLED = polish_enabled
        yield
    finally:
        settings.SMART_EXTRACTION_POLISH_ENABLED = original_polish


def build_profile(profile_name: str) -> dict[str, Any]:
    if profile_name == "baseline":
        return {"enable_validation": None, "polish_enabled": None}
    if profile_name == "no_polish":
        return {"enable_validation": None, "polish_enabled": False}
    if profile_name == "no_validation":
        return {"enable_validation": False, "polish_enabled": None}
    if profile_name == "fastest":
        return {"enable_validation": False, "polish_enabled": False}
    raise ValueError(f"Unknown profile: {profile_name}")


async def run_once(pdf_path: Path, profile_name: str) -> dict[str, Any]:
    profile = build_profile(profile_name)
    extraction_service = build_page_aware_extraction_service()

    with temporary_settings(polish_enabled=profile["polish_enabled"]):
        started = time.perf_counter()
        result = await extraction_service.extract_document(
            str(pdf_path),
            enable_validation=profile["enable_validation"],
        )
        wall_time_s = time.perf_counter() - started

    return {
        "profile": profile_name,
        "pdf": str(pdf_path),
        "wall_time_s": round(wall_time_s, 3),
        "timing": result.metadata.get("timing", {}),
        "workflow": result.metadata.get("workflow"),
        "ocr_mode": result.metadata.get("ocr_mode"),
        "total_pages": result.metadata.get("total_pages"),
        "page_types": result.metadata.get("page_types"),
        "content_chars": len(result.content or ""),
        "validated": bool(result.validation_report),
    }


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    wall_times = [run["wall_time_s"] for run in runs]
    timing_keys = set()
    for run in runs:
        timing_keys.update(run.get("timing", {}).keys())

    timing_avg = {}
    for key in sorted(timing_keys):
        values = [run["timing"][key] for run in runs if key in run.get("timing", {})]
        if values:
            timing_avg[key] = round(statistics.mean(values), 3)

    summary = {
        "runs": len(runs),
        "avg_wall_time_s": round(statistics.mean(wall_times), 3),
        "min_wall_time_s": round(min(wall_times), 3),
        "max_wall_time_s": round(max(wall_times), 3),
        "avg_timing": timing_avg,
        "sample": runs[-1],
    }
    if len(wall_times) > 1:
        summary["stdev_wall_time_s"] = round(statistics.pstdev(wall_times), 3)
    return summary


def print_summary(profile_name: str, summary: dict[str, Any]) -> None:
    print(f"\nProfile: {profile_name} ({PROFILE_DESCRIPTIONS[profile_name]})")
    print(f"  Runs: {summary['runs']}")
    print(
        "  Wall time: "
        f"avg={summary['avg_wall_time_s']:.3f}s "
        f"min={summary['min_wall_time_s']:.3f}s "
        f"max={summary['max_wall_time_s']:.3f}s"
    )
    if "stdev_wall_time_s" in summary:
        print(f"  Wall time stdev: {summary['stdev_wall_time_s']:.3f}s")

    avg_timing = summary.get("avg_timing", {})
    if avg_timing:
        print("  Phase averages:")
        for key, value in avg_timing.items():
            print(f"    {key}: {value:.3f}s")

    sample = summary["sample"]
    print(
        "  Sample result: "
        f"pages={sample.get('total_pages')} "
        f"ocr_mode={sample.get('ocr_mode')} "
        f"validated={sample.get('validated')} "
        f"chars={sample.get('content_chars')}"
    )


async def benchmark_pdf(
    pdf_path: Path,
    profiles: list[str],
    repeat: int,
    warmup: int,
) -> dict[str, Any]:
    print(f"\n=== Benchmarking {pdf_path} ===")
    results: dict[str, Any] = {"pdf": str(pdf_path), "profiles": {}}

    for profile_name in profiles:
        if warmup:
            for _ in range(warmup):
                await run_once(pdf_path, profile_name)

        runs = []
        for _ in range(repeat):
            runs.append(await run_once(pdf_path, profile_name))

        summary = summarize_runs(runs)
        results["profiles"][profile_name] = {
            "summary": summary,
            "runs": runs,
        }
        print_summary(profile_name, summary)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="+", help="PDF files to benchmark")
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=sorted(PROFILE_DESCRIPTIONS),
        default=["baseline", "no_polish", "no_validation", "fastest"],
        help="Benchmark profiles to run",
    )
    parser.add_argument("--repeat", type=int, default=2, help="Measured runs per profile")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per profile")
    parser.add_argument("--json-out", type=Path, help="Optional path to write raw JSON benchmark results")
    return parser.parse_args()


async def async_main() -> int:
    args = parse_args()
    setup_logging()

    pdf_paths = [Path(pdf).expanduser().resolve() for pdf in args.pdfs]
    missing = [str(path) for path in pdf_paths if not path.exists()]
    if missing:
        for path in missing:
            print(f"Missing PDF: {path}", file=sys.stderr)
        return 1

    results = []
    for pdf_path in pdf_paths:
        results.append(
            await benchmark_pdf(
                pdf_path=pdf_path,
                profiles=args.profiles,
                repeat=args.repeat,
                warmup=args.warmup,
            )
        )

    if args.json_out:
        args.json_out.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nWrote JSON results to {args.json_out}")

    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())

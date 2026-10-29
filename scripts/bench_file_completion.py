#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Measure ``@`` file-completion phase latency on a real or synthetic workspace.

Developer tool, not a production entrypoint.  It wraps the shipped functions —
``file_scanner``'s four discovery probes, ``_project_path_suggestions`` and
``ProjectPathIndex.build``/``query`` — so the numbers describe the real code
path rather than a copy of it, then prints per-phase P50/P95 next to an
environment header that makes a run reproducible.  Nothing here asserts
wall-clock time: performance data never gates CI (risk R4 of the LZHD-327
plan); pytest only covers this script's structure.

``--synthetic 50k|100k`` generates the paths in memory, which isolates the
scale-dependent phases (derived rows, index build, query) from disk and
antivirus noise.  Treat those numbers as a warm-page-cache upper bound, not a
prediction of how a real filesystem of that size behaves.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from chrys.app.tui.widgets.chrome import file_scanner as file_scanner_module
from chrys.app.tui.widgets.chrome.file_index import ProjectPathIndex
from chrys.app.tui.widgets.chrome.file_scanner import ProjectPathScanResult, scan_project_paths

# Synthetic workspace sizes, spelled as the plan names them.
SCALES = {"50k": 50_000, "100k": 100_000}

# Mirrors ``suggestions.py::_FILE_QUERY_LIMIT``: the menu renders 30 rows.
QUERY_LIMIT = 30

# One representative query per shape: empty (head), single character (worst
# case for the linear scan), directory prefix, common substring, file name.
QUERIES = ("", "e", "src/", "test_", "file_completion")

# Repeated per round so the P95 of a fast query is still a measurement.
QUERY_ITERATIONS = 10

# ``file_scanner`` module attribute -> reported phase name.
PROBE_PHASES = {
    "_scan_rg_files": "rg_files",
    "_append_git_cached_files": "git_cached",
    "_append_dot_entries": "dot_entries",
    "_scan_ignored_visible_files": "ignored_visible",
}
SYNC_PHASES = {"_project_path_suggestions": "derived_rows"}

# Phases that make up the attribution total, in table order.
PHASE_ORDER = (
    "scan_total",
    "rg_files",
    "git_cached",
    "dot_entries",
    "ignored_visible",
    "derived_rows",
    "index_build",
    "query",
)


@dataclass
class Recorder:
    """Per-phase wall-clock samples in seconds."""

    samples: dict[str, list[float]] = field(default_factory=dict)

    def record(self, phase: str, seconds: float) -> None:
        """Append one sample for *phase*."""
        self.samples.setdefault(phase, []).append(seconds)

    def summary(self) -> dict[str, dict[str, float | int]]:
        """Return millisecond statistics per recorded phase."""
        return {phase: summarize(values) for phase, values in self.samples.items()}


def _nearest_rank(ordered: Sequence[float], fraction: float) -> float:
    """Return the nearest-rank percentile of an ascending sample list."""
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def summarize(samples: Sequence[float]) -> dict[str, float | int]:
    """Return sample count plus millisecond min/max/mean/P50/P95 for *samples*."""
    ordered = sorted(samples)
    if not ordered:
        return {"samples": 0, "p50_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "mean_ms": 0.0}
    return {
        "samples": len(ordered),
        "p50_ms": round(statistics.median(ordered) * 1000, 3),
        "p95_ms": round(_nearest_rank(ordered, 0.95) * 1000, 3),
        "min_ms": round(ordered[0] * 1000, 3),
        "max_ms": round(ordered[-1] * 1000, 3),
        "mean_ms": round(statistics.fmean(ordered) * 1000, 3),
    }


def _timed_async(recorder: Recorder, phase: str, target: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap an async *target* so each call records its duration under *phase*."""

    async def timed(*args: Any, **kwargs: Any) -> Any:
        start = perf_counter()
        try:
            return await target(*args, **kwargs)
        finally:
            recorder.record(phase, perf_counter() - start)

    return timed


def _timed_sync(recorder: Recorder, phase: str, target: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a sync *target* so each call records its duration under *phase*."""

    def timed(*args: Any, **kwargs: Any) -> Any:
        start = perf_counter()
        try:
            return target(*args, **kwargs)
        finally:
            recorder.record(phase, perf_counter() - start)

    return timed


@contextmanager
def instrumented_scanner(recorder: Recorder) -> Generator[None]:
    """Record per-probe timings while the shipped ``scan_project_paths`` runs."""
    replacements: dict[str, Callable[..., Any]] = {}
    for name, phase in PROBE_PHASES.items():
        replacements[name] = _timed_async(recorder, phase, getattr(file_scanner_module, name))
    for name, phase in SYNC_PHASES.items():
        replacements[name] = _timed_sync(recorder, phase, getattr(file_scanner_module, name))
    originals = {name: getattr(file_scanner_module, name) for name in replacements}
    try:
        for name, replacement in replacements.items():
            setattr(file_scanner_module, name, replacement)
        yield
    finally:
        for name, original in originals.items():
            setattr(file_scanner_module, name, original)


async def _measure_queries(index: ProjectPathIndex, recorder: Recorder, detail: dict[str, list[float]]) -> None:
    """Time ``index.query`` for every query in :data:`QUERIES`."""
    for query in QUERIES:
        label = query or "<empty>"
        for _ in range(QUERY_ITERATIONS):
            start = perf_counter()
            await asyncio.to_thread(index.query, query, limit=QUERY_LIMIT)
            elapsed = perf_counter() - start
            recorder.record("query", elapsed)
            detail.setdefault(label, []).append(elapsed)


async def _measure_index(
    scan: ProjectPathScanResult, recorder: Recorder, detail: dict[str, list[float]]
) -> ProjectPathIndex:
    """Time index construction (normalisation) and the query phase for *scan*."""
    start = perf_counter()
    index = await asyncio.to_thread(ProjectPathIndex.build, scan)
    recorder.record("index_build", perf_counter() - start)
    await _measure_queries(index, recorder, detail)
    return index


async def bench_real(root: str, repeat: int, recorder: Recorder, detail: dict[str, list[float]]) -> dict[str, Any]:
    """Measure the real ``scan_project_paths`` -> index -> query path on *root*."""
    first_scan: ProjectPathScanResult | None = None
    for _ in range(repeat):
        gc.collect()
        start = perf_counter()
        with instrumented_scanner(recorder):
            scan = await scan_project_paths(root)
        recorder.record("scan_total", perf_counter() - start)
        if first_scan is None:
            first_scan = scan
        await _measure_index(scan, recorder, detail)
    if first_scan is None:  # pragma: no cover - repeat >= 1 is enforced by the CLI
        raise ValueError("repeat must be at least 1")
    return {
        "kind": "real",
        "root": os.path.abspath(root),
        "counts": {
            "files": first_scan.file_count,
            "suggestions": first_scan.suggestion_count,
            "truncated": first_scan.truncated,
            "source_truncations": dict(first_scan.source_truncations),
        },
    }


def generate_synthetic_paths(count: int) -> list[str]:
    """Return *count* unique source-shaped paths, one per file, for scale testing."""
    roots = ("src", "tests", "docs", "scripts", "examples")
    return [f"{roots[index % len(roots)]}/pkg{index % 500}/sub{index % 37}/mod_{index}.py" for index in range(count)]


async def bench_synthetic(
    count: int, repeat: int, recorder: Recorder, detail: dict[str, list[float]]
) -> dict[str, Any]:
    """Measure derived rows, index build and query over *count* in-memory paths."""
    paths = generate_synthetic_paths(count)
    build = await asyncio.to_thread(file_scanner_module._project_path_suggestions, paths)
    for _ in range(repeat):
        gc.collect()
        start = perf_counter()
        await asyncio.to_thread(file_scanner_module._project_path_suggestions, paths)
        recorder.record("derived_rows", perf_counter() - start)
        scan = ProjectPathScanResult.from_suggestions(root="<synthetic>", paths=build.suggestions)
        await _measure_index(scan, recorder, detail)
    return {
        "kind": "synthetic",
        "count": count,
        "counts": {
            "files": len(paths),
            "suggestions": len(build.suggestions),
            "truncated": build.truncated,
        },
    }


def _tool_version(executable: str | None, *args: str) -> str:
    """Return the first line of ``<executable> <args>`` output, or ``unavailable``."""
    if not executable:
        return "unavailable"
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
            [executable, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return "unavailable"
    lines = completed.stdout.strip().splitlines()
    return lines[0] if lines else "unavailable"


def environment_header() -> dict[str, Any]:
    """Return the machine facts that make a measurement run reproducible."""
    return {
        "os": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "interpreter": sys.executable,
        "cpu_count": os.cpu_count() or 0,
        "rg": _tool_version(file_scanner_module._rg_path(), "--version"),
        "git": _tool_version(shutil.which("git"), "--version"),
    }


def phase_shares(phases: dict[str, dict[str, float | int]]) -> dict[str, float]:
    """Return each attributed phase's P50 as a percentage of their P50 sum."""
    attributed = [phase for phase in PHASE_ORDER if phase in phases and phase != "scan_total"]
    total = sum(float(phases[phase]["p50_ms"]) for phase in attributed)
    if total <= 0:
        return {}
    return {phase: round(float(phases[phase]["p50_ms"]) / total * 100, 1) for phase in attributed}


def _phase_rows(phases: dict[str, dict[str, float | int]]) -> list[str]:
    """Return the fixed-order phase table body."""
    ordered = [phase for phase in PHASE_ORDER if phase in phases]
    ordered.extend(phase for phase in phases if phase not in PHASE_ORDER)
    return [phase for phase in ordered if phase in phases]


def format_report(report: dict[str, Any]) -> str:
    """Render *report* as the human-readable table."""
    env = report["environment"]
    counts = report["counts"]
    lines = [
        f"file-completion benchmark  mode={report['mode']}  repeat={report['repeat']}",
        (
            f"env: {env['os']}  machine={env['machine']}  cpus={env['cpu_count']}"
            f"  python={env['python']}  rg={env['rg']}  git={env['git']}"
        ),
        f"interpreter: {env['interpreter']}",
    ]
    if report["mode"] == "real":
        lines.append(
            f"root: {report['root']}  files={counts['files']}  suggestions={counts['suggestions']}"
            f"  truncated={counts['truncated']}  sources={counts['source_truncations']}"
        )
    else:
        lines.append(f"synthetic: files={counts['files']}  suggestions={counts['suggestions']}")
    lines.append("")
    header = f"{'phase':<17}{'p50 ms':>9}{'p95 ms':>9}{'min ms':>9}{'max ms':>9}{'n':>5}{'share':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    shares = phase_shares(report["phases"])
    for phase in _phase_rows(report["phases"]):
        stats = report["phases"][phase]
        share = f"{shares[phase]:.1f}%" if phase in shares else "-"
        lines.append(
            f"{phase:<17}{stats['p50_ms']:>9.2f}{stats['p95_ms']:>9.2f}{stats['min_ms']:>9.2f}"
            f"{stats['max_ms']:>9.2f}{stats['samples']:>5}{share:>8}"
        )
    if report["queries"]:
        lines.append("")
        lines.append(f"{'query':<17}{'p50 ms':>9}{'p95 ms':>9}{'n':>5}")
        for label, stats in report["queries"].items():
            lines.append(f"{label:<17}{stats['p50_ms']:>9.3f}{stats['p95_ms']:>9.3f}{stats['samples']:>5}")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Measure @ file-completion phase latency (P50/P95) with a reproducible environment header.",
    )
    parser.add_argument("--root", default=None, help="workspace root to scan (default: current directory)")
    parser.add_argument(
        "--synthetic",
        choices=sorted(SCALES),
        default=None,
        help="measure an in-memory workspace of this many files instead of a real root",
    )
    parser.add_argument("--repeat", type=int, default=6, help="measurement rounds (default: 6)")
    parser.add_argument("--json", action="store_true", help="print the machine-readable report instead of the table")
    return parser


def resolve_mode(args: argparse.Namespace) -> tuple[str, str | None]:
    """Return ``(mode, root)`` for *args*; raise ``ValueError`` on a conflicting request."""
    if args.synthetic is not None:
        if args.root is not None:
            raise ValueError("--root and --synthetic are mutually exclusive")
        return "synthetic", None
    return "real", args.root or "."


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and print the report."""
    args = build_parser().parse_args(argv)
    if args.repeat < 1:
        sys.stderr.write("error: --repeat must be at least 1\n")
        return 2
    try:
        mode, root = resolve_mode(args)
    except ValueError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2

    recorder = Recorder()
    detail: dict[str, list[float]] = {}
    if mode == "synthetic":
        payload = asyncio.run(bench_synthetic(SCALES[args.synthetic], args.repeat, recorder, detail))
    else:
        payload = asyncio.run(bench_real(str(root), args.repeat, recorder, detail))
    report: dict[str, Any] = {
        "tool": "bench_file_completion",
        "mode": mode,
        "repeat": args.repeat,
        "environment": environment_header(),
        "queries": {label: summarize(values) for label, values in detail.items()},
        "phases": recorder.summary(),
        **payload,
    }
    if args.json:
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    else:
        sys.stdout.write(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

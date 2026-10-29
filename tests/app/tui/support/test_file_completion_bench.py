# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the ``@`` file-completion benchmark script.

The script exists to report wall-clock phase latencies to a human; these tests
pin its structure only — argument handling, synthetic generation, percentile
arithmetic and report shape — and never assert a duration, because performance
data must not gate CI (risk R4 of the LZHD-327 plan).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from scripts import bench_file_completion as bench
from scripts.bench_file_completion import (
    PHASE_ORDER,
    Recorder,
    build_parser,
    format_report,
    generate_synthetic_paths,
    instrumented_scanner,
    main,
    phase_shares,
    resolve_mode,
    summarize,
)

GIT = shutil.which("git")


def _git_repo(root: Path, files: dict[str, str]) -> Path:
    """Write *files* into a fresh git repository at *root* and return it."""
    if GIT is None:
        pytest.skip("git is required for a repository-backed benchmark fixture")
    subprocess.run([GIT, "init"], cwd=root, check=True, capture_output=True)
    for name, content in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(content, encoding="utf-8")
    return root


def _report(argv: list[str], capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    """Run the script with *argv* and return its parsed JSON report."""
    assert main(argv) == 0
    return json.loads(capsys.readouterr().out)


class TestSyntheticPaths:
    def test_paths_are_unique_stable_and_source_shaped(self) -> None:
        paths = generate_synthetic_paths(200)

        assert len(paths) == 200
        assert len(set(paths)) == 200
        assert paths == generate_synthetic_paths(200)
        assert all(path.startswith(("src/", "tests/", "docs/", "scripts/", "examples/")) for path in paths)
        assert all(path.endswith(".py") for path in paths)

    def test_zero_paths_is_empty(self) -> None:
        assert generate_synthetic_paths(0) == []


class TestSummarize:
    def test_reports_median_p50_and_nearest_rank_p95(self) -> None:
        stats = summarize([0.001, 0.002, 0.003, 0.004, 0.005])

        assert stats == {
            "samples": 5,
            "p50_ms": 3.0,
            "p95_ms": 5.0,
            "min_ms": 1.0,
            "max_ms": 5.0,
            "mean_ms": 3.0,
        }

    def test_even_sample_count_median_averages_the_middle_pair(self) -> None:
        # Six rounds is the documented default, so the even-count P50 matters.
        assert summarize([0.348, 0.400, 0.450, 0.500, 0.600, 0.660])["p50_ms"] == 475.0

    def test_empty_samples_report_zeroes(self) -> None:
        assert summarize([]) == {
            "samples": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "mean_ms": 0.0,
        }


class TestPhaseShares:
    def test_shares_are_p50_percentages_of_the_attributed_phases(self) -> None:
        phases = {
            "rg_files": summarize([0.050]),
            "derived_rows": summarize([0.030]),
            "index_build": summarize([0.020]),
        }

        assert phase_shares(phases) == {"rg_files": 50.0, "derived_rows": 30.0, "index_build": 20.0}

    def test_scan_total_is_excluded_from_the_share_total(self) -> None:
        phases = {"scan_total": summarize([0.900]), "rg_files": summarize([0.100])}

        assert phase_shares(phases) == {"rg_files": 100.0}

    def test_no_attributed_samples_yields_no_shares(self) -> None:
        assert phase_shares({"scan_total": summarize([0.900])}) == {}


class TestArgumentHandling:
    def test_parser_defaults_to_the_current_directory_with_six_rounds(self) -> None:
        args = build_parser().parse_args([])

        assert resolve_mode(args) == ("real", ".")
        assert args.repeat == 6
        assert args.json is False

    def test_parser_accepts_a_real_root_and_a_scale(self) -> None:
        assert resolve_mode(build_parser().parse_args(["--root", "some/dir"])) == ("real", "some/dir")
        assert resolve_mode(build_parser().parse_args(["--synthetic", "50k"])) == ("synthetic", None)
        assert resolve_mode(build_parser().parse_args(["--synthetic", "100k"])) == ("synthetic", None)

    def test_parser_rejects_an_unknown_scale(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--synthetic", "10k"])

    def test_conflicting_root_and_synthetic_exits_with_usage_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--root", ".", "--synthetic", "50k"]) == 2
        assert "mutually exclusive" in capsys.readouterr().err

    def test_non_positive_repeat_exits_with_usage_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--repeat", "0"]) == 2
        assert "--repeat must be at least 1" in capsys.readouterr().err


class TestInstrumentedScanner:
    def test_wrapping_scanner_probes_is_reverted_afterwards(self) -> None:
        module = bench.file_scanner_module
        wrapped_names = [*bench.PROBE_PHASES, *bench.SYNC_PHASES]
        originals = {name: getattr(module, name) for name in wrapped_names}

        with instrumented_scanner(Recorder()):
            replaced = [getattr(module, name) is not original for name, original in originals.items()]
            assert replaced == [True] * len(wrapped_names)

        assert [getattr(module, name) is not original for name, original in originals.items()] == [False] * len(
            wrapped_names
        )

    def test_wrapping_is_reverted_when_the_body_raises(self) -> None:
        module = bench.file_scanner_module
        original = module._scan_rg_files

        with pytest.raises(RuntimeError), instrumented_scanner(Recorder()):
            raise RuntimeError("boom")

        assert module._scan_rg_files is original


class TestRealReport:
    def test_json_report_carries_the_environment_header_and_every_phase(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo = _git_repo(
            tmp_path,
            {"alpha.py": "alpha = 1\n", "beta.md": "# beta\n", "pkg/gamma.py": "gamma = 2\n"},
        )

        report = _report(["--root", str(repo), "--repeat", "1", "--json"], capsys)

        assert report["tool"] == "bench_file_completion"
        assert report["mode"] == "real"
        assert report["repeat"] == 1
        assert report["root"] == str(repo.resolve())
        assert report["counts"] == {"files": 3, "suggestions": 4, "truncated": False, "source_truncations": {}}

        environment = report["environment"]
        assert set(environment) == {
            "os",
            "system",
            "release",
            "machine",
            "python",
            "interpreter",
            "cpu_count",
            "rg",
            "git",
        }
        assert environment["python"].startswith("3.")

        phases = report["phases"]
        assert set(phases) == set(PHASE_ORDER)
        assert phases["scan_total"]["samples"] == 1
        assert phases["query"]["samples"] == bench.QUERY_ITERATIONS * len(bench.QUERIES)

        # Durations are reported, never asserted: only the shape is a contract.
        assert isinstance(phases["scan_total"]["p50_ms"], float)
        assert phases["scan_total"]["p50_ms"] >= 0.0

        assert set(report["queries"]) == {"<empty>", "e", "src/", "test_", "file_completion"}

    def test_table_report_names_the_environment_and_the_phase_rows(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo = _git_repo(tmp_path, {"alpha.py": "alpha = 1\n"})

        assert main(["--root", str(repo), "--repeat", "1"]) == 0
        out = capsys.readouterr().out

        assert out.startswith("file-completion benchmark  mode=real  repeat=1\n")
        assert "env: " in out
        assert f"root: {repo.resolve()}  files=1" in out
        for phase in ("scan_total", "rg_files", "derived_rows", "index_build", "query"):
            assert f"{phase:<17}" in out


class TestSyntheticReport:
    def test_synthetic_mode_skips_probes_and_reports_scale_phases(self, capsys: pytest.CaptureFixture[str]) -> None:
        report = _report(["--synthetic", "50k", "--repeat", "1", "--json"], capsys)

        assert report["mode"] == "synthetic"
        assert report["count"] == 50_000
        assert report["counts"]["files"] == 50_000
        assert report["counts"]["suggestions"] > 50_000
        phases = report["phases"]
        assert set(phases) == {"derived_rows", "index_build", "query"}
        assert "root" not in report


class TestFormatReport:
    def test_synthetic_table_labels_the_generated_workspace(self) -> None:
        report = {
            "mode": "synthetic",
            "repeat": 1,
            "environment": {
                "os": "test-os",
                "machine": "x",
                "cpu_count": 1,
                "python": "3.14",
                "interpreter": "python",
                "rg": "rg",
                "git": "git",
            },
            "counts": {"files": 10, "suggestions": 12, "truncated": False},
            "phases": {"derived_rows": summarize([0.002]), "index_build": summarize([0.001])},
            "queries": {"<empty>": summarize([0.001])},
        }

        rendered = format_report(report)

        assert "synthetic: files=10  suggestions=12" in rendered
        assert "derived_rows" in rendered
        assert "<empty>" in rendered
        assert "share" in rendered

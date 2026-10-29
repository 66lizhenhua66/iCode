# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""AC6 no-regression matrix for the ``@`` file-completion surface.

Pins what the index-lifecycle and scan-parallelism work (LZHD-329/LZHD-330)
must keep byte-identical, so a "faster" implementation cannot silently change
what the menu shows:

* the four ``fuzzy_filter`` reuse points — ``/`` slash commands, ``#`` agents,
  ``$`` models and ``@`` project paths — including the
  (match position, path length, path text) tiebreaks;
* ``source_truncations`` key names and the two reachable truncation triggers,
  plus the ``index.truncated`` gate the bounded-index row hangs on;
* the ``glob``/``grep`` tool output shape, which shares the scanner's ignore
  semantics and was frozen at ``c54b60c``.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from chrys.app.tui.widgets.chrome.file_index import ProjectPathIndex
from chrys.app.tui.widgets.chrome.file_scanner import (
    ProjectPathScanResult,
    ProjectPathSuggestion,
    fuzzy_filter,
    scan_project_paths,
)
from chrys.service.tools.builtins.search import glob, grep

GIT = shutil.which("git")

# The only ``source_truncations`` keys ``scan_project_paths`` may set (AC6).
TRUNCATION_SOURCES = frozenset(
    {"rg", "git_cached", "git_files", "os_walk", "dot_entries", "ignored_visible", "suggestions"}
)

# Candidates shaped like each reuse point's real list; the literals below are
# the frozen ordering contract, not a re-derivation of the implementation.
REUSE_POINT_CASES = [
    pytest.param(
        "/",
        ["clear", "compact", "help", "hooks", "model", "workflow"],
        "h",
        ["help", "hooks"],
        id="slash-commands",
    ),
    pytest.param(
        "/",
        ["clear", "compact", "help", "hooks", "model", "workflow"],
        "",
        ["clear", "compact", "help", "hooks", "model", "workflow"],
        id="slash-commands-empty-query-head",
    ),
    pytest.param("#", ["Code", "Architect", "Reviewer"], "r", ["Reviewer", "Architect"], id="agents"),
    pytest.param("#", ["Code", "Architect", "Reviewer"], "CODE", ["Code"], id="agents-casefolded-query"),
    pytest.param(
        "$",
        ["gpt-5.1", "claude-sonnet-4.5", "deepseek-v4.1"],
        "4",
        ["deepseek-v4.1", "claude-sonnet-4.5"],
        id="models",
    ),
    pytest.param(
        "@",
        ["src/chrys/", "src/chrys/app/", "src/chrys/app/tui/file.py", "docs/guide.md"],
        "app",
        ["src/chrys/app/", "src/chrys/app/tui/file.py"],
        id="project-paths",
    ),
    pytest.param(
        "@",
        ["src/chrys/", "src/chrys/app/"],
        "APP",
        ["src/chrys/app/"],
        id="project-paths-casefolded-query",
    ),
]

PATH_ORDERING_CASES = [
    pytest.param("b/x.py", "a/x.py", "x", ["a/x.py", "b/x.py"], id="equal-position-and-length-breaks-on-text"),
    pytest.param(
        "src/chrys/app/tui/file.py",
        "src/app.py",
        "app",
        ["src/app.py", "src/chrys/app/tui/file.py"],
        id="earlier-match-position-wins-over-length",
    ),
    pytest.param(
        "docs/guide.md",
        "app/short.py",
        "app",
        ["app/short.py"],
        id="miss-is-not-a-match",
    ),
]


def _index(*paths: str) -> ProjectPathIndex:
    suggestions = [
        ProjectPathSuggestion(path=path, kind="directory" if path.endswith("/") else "file") for path in paths
    ]
    return ProjectPathIndex.build(ProjectPathScanResult.from_suggestions(root="/repo", paths=suggestions))


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A git repository with four tracked-looking files (nothing ignored)."""
    if GIT is None:
        pytest.skip("git is required for scanner-backed truncation tests")
    subprocess.run([GIT, "init"], cwd=tmp_path, check=True, capture_output=True)
    for name in ("alpha.py", "beta.md", "pkg/gamma.py", "pkg/delta.py"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("value = 1\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def search_repo(tmp_path: Path) -> Iterator[Path]:
    """A git repository with searchable, hidden and gitignored files."""
    if GIT is None:
        pytest.skip("git is required for search-tool snapshot tests")
    subprocess.run([GIT, "init"], cwd=tmp_path, check=True, capture_output=True)
    files = {
        "a.py": "x = 1\n",
        "b.py": "y = 2\n",
        "c.txt": "hello\n",
        ".hidden.py": "hidden = True\n",
        "secret.py": "x = 1\n",
        ".gitignore": "secret.py\n",
    }
    for name, content in files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    yield tmp_path


class TestFuzzyFilterReusePoints:
    @pytest.mark.parametrize(("prefix", "candidates", "query", "expected"), REUSE_POINT_CASES)
    def test_prefix_query_ordering_is_unchanged(
        self, prefix: str, candidates: list[str], query: str, expected: list[str]
    ) -> None:
        assert fuzzy_filter(query, candidates) == expected, prefix


class TestProjectPathIndexOrdering:
    @pytest.mark.parametrize(("first", "second", "query", "expected"), PATH_ORDERING_CASES)
    def test_index_query_keeps_the_frozen_path_ordering(
        self, first: str, second: str, query: str, expected: list[str]
    ) -> None:
        index = _index(first, second)

        assert [suggestion.path for suggestion in index.query(query, limit=30)] == expected

    def test_empty_query_returns_the_insertion_order_head(self) -> None:
        index = _index("b/x.py", "a/x.py", "src/chrys/")

        assert [suggestion.path for suggestion in index.query("", limit=2)] == ["b/x.py", "a/x.py"]


class TestTruncationContract:
    async def test_full_budget_scan_reports_no_truncation(self, repo: Path) -> None:
        scan = await scan_project_paths(str(repo))

        assert scan.source_truncations == {}
        assert scan.truncated is False
        assert ProjectPathIndex.build(scan).truncated is False

    async def test_file_budget_sets_only_the_rg_key(self, repo: Path) -> None:
        scan = await scan_project_paths(str(repo), file_budget=2)

        assert scan.source_truncations == {"rg": True}
        assert scan.truncated is any(scan.source_truncations.values())
        index = ProjectPathIndex.build(scan)
        assert index.truncated is True
        assert index.file_count == 2

    async def test_suggestion_budget_sets_only_the_suggestions_key(self, repo: Path) -> None:
        scan = await scan_project_paths(str(repo), suggestion_budget=1)

        assert scan.source_truncations == {"suggestions": True}
        assert scan.truncated is any(scan.source_truncations.values())
        index = ProjectPathIndex.build(scan)
        assert index.truncated is True
        assert index.suggestion_count == 1

    async def test_emitted_source_keys_stay_inside_the_frozen_set(self, repo: Path) -> None:
        emitted: set[str] = set()
        for budgets in ({}, {"file_budget": 2}, {"suggestion_budget": 1}):
            scan = await scan_project_paths(str(repo), **budgets)
            emitted.update(scan.source_truncations)

        assert emitted == {"rg", "suggestions"}
        assert emitted <= TRUNCATION_SOURCES


class TestSearchToolSnapshots:
    """``grep``/``glob`` output shape frozen at ``c54b60c`` (files: same ignore rules as ``@``)."""

    async def test_glob_header_and_relative_rows_exclude_hidden_and_ignored_files(self, search_repo: Path) -> None:
        root = Path(search_repo).resolve()

        result = await glob("*.py", path=str(search_repo))

        header, *rows = result.splitlines()
        assert header == f"Found 2 file(s) matching '*.py' in {root}"
        assert sorted(row.strip() for row in rows) == ["a.py", "b.py"]

    async def test_glob_no_match_message_is_unchanged(self, search_repo: Path) -> None:
        root = Path(search_repo).resolve()

        assert await glob("*.rs", path=str(search_repo)) == f"No files matching '*.rs' found in {root}"

    async def test_grep_match_block_shape_is_unchanged(self, search_repo: Path) -> None:
        root = Path(search_repo).resolve()

        result = await grep("x = 1", path=str(search_repo))

        header, _, body = result.partition("\n")
        assert header == f"Found 1 match(es) in {root}"
        assert body.strip() == "a.py:1\n  >    1 | x = 1"

    async def test_grep_no_match_message_is_unchanged(self, search_repo: Path) -> None:
        root = Path(search_repo).resolve()

        assert await grep("nomatch_xyz", path=str(search_repo)) == f"No matches found for /nomatch_xyz/ in {root}"

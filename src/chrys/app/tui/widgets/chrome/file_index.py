# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""In-memory search index for project path suggestions.

``ProjectPathIndex`` is one immutable, pre-normalized snapshot of a scan.  Its
query returns a bounded top-k result with the same ordering semantics as
``fuzzy_filter``: match position, original path length, then original path text.

``WorkspacePathIndex`` owns that snapshot's lifecycle for one root: the ``@``
popup answers every keystroke from the installed snapshot, while rebuilds run
off the input path and only hot-swap the snapshot once they finish.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from bisect import bisect_left
from collections.abc import Awaitable, Callable, Iterable
from typing import TYPE_CHECKING

from chrys.app.tui.widgets.chrome.file_scanner import (
    ProjectPathScanResult,
    ProjectPathSuggestion,
    _has_non_referenceable_file_extension,
    _include_ignored_visible_path,
    _normalize_for_search,
)

if TYPE_CHECKING:
    from chrys.app.tui.widgets.chrome.file_scanner import ProjectPathOrdering

logger = logging.getLogger(__name__)

_DEFAULT_QUERY_LIMIT = 30

type ScanProjectPaths = Callable[[str], Awaitable[ProjectPathScanResult]]


def _default_scan(root: str) -> Awaitable[ProjectPathScanResult]:
    """Resolve the scanner at call time so tests and callers can substitute it."""
    from chrys.app.tui.widgets.chrome import file_scanner

    return file_scanner.scan_project_paths(root)


class ProjectPathIndex:
    """Pre-normalized substring index for ``@`` file suggestions."""

    def __init__(self, scan: ProjectPathScanResult) -> None:
        self.root = scan.root
        self.file_count = scan.file_count
        self.suggestion_count = scan.suggestion_count
        self.truncated = scan.truncated
        self.file_budget = scan.file_budget
        self.suggestion_budget = scan.suggestion_budget
        self.source_truncations = dict(scan.source_truncations)
        self.ordering = scan.ordering
        self._suggestions = list(scan.paths)
        self._normalized = [_normalize_for_search(suggestion.path) for suggestion in self._suggestions]

    @classmethod
    def build(cls, scan: ProjectPathScanResult) -> ProjectPathIndex:
        """Build an index from a scan result."""
        return cls(scan)

    def head(self, *, limit: int = _DEFAULT_QUERY_LIMIT) -> list[ProjectPathSuggestion]:
        """Return the first suggestions in insertion order."""
        if limit <= 0:
            return []
        return self._suggestions[:limit]

    def query(self, query: str, *, limit: int = _DEFAULT_QUERY_LIMIT) -> list[ProjectPathSuggestion]:
        """Return exact current fuzzy-filter ordering for the best file matches."""
        if limit <= 0:
            return []
        if not query:
            return self.head(limit=limit)

        normalized_query = _normalize_for_search(query)
        best: list[tuple[tuple[int, int, str], int]] = []
        for index, normalized_path in enumerate(self._normalized):
            match_index = normalized_path.find(normalized_query)
            if match_index < 0:
                continue
            suggestion = self._suggestions[index]
            entry = ((match_index, len(suggestion.path), suggestion.path), index)
            if len(best) < limit:
                best.insert(bisect_left(best, entry), entry)
            elif entry < best[-1]:
                best.insert(bisect_left(best, entry), entry)
                best.pop()
        return [self._suggestions[index] for _score, index in best]


def _ancestor_directories(relative_path: str) -> list[str]:
    """Return the ``@`` directory rows a file row implies, outermost first."""
    parts = relative_path.split("/")
    return ["/".join(parts[:index]) + "/" for index in range(1, len(parts))]


def _directory_reference_counts(files: Iterable[str]) -> dict[str, int]:
    """Count the files under each directory, so a directory row lives iff its count is positive."""
    counts: dict[str, int] = {}
    for relative_path in files:
        for directory in _ancestor_directories(relative_path):
            counts[directory] = counts.get(directory, 0) + 1
    return counts


class WorkspacePathIndex:
    """Resident ``@`` path snapshot for one workspace root.

    The popup only ever reads :attr:`snapshot`; nothing here scans on the input
    path.  ``dirty`` says the snapshot may be behind the filesystem and
    ``delta`` holds the changed absolute paths observed since it was installed,
    so ``mark_dirty()`` with no paths means the change set is unknown and a full
    rescan is required.  :meth:`refresh` is the single-flight rebuild: it
    replaces the snapshot only when it finishes for the same :attr:`epoch`,
    which is what makes a rebuild that outlives a workspace switch harmless.
    """

    def __init__(self, root: str) -> None:
        self.root = root
        self.epoch = 0
        self.dirty = False
        self.snapshot: ProjectPathIndex | None = None
        self.build_task: asyncio.Task[bool] | None = None
        self.delta: set[str] = set()
        self._files: list[str] = []
        self._file_set: set[str] = set()
        self._directory_counts: dict[str, int] = {}
        self._source_truncations: dict[str, bool] = {}
        self._file_budget = 0
        self._suggestion_budget = 0
        self._ordering: ProjectPathOrdering = "path-sorted"
        # Bumped on every recorded change, so a rebuild that straddled one knows
        # its scan predates the change and leaves the snapshot dirty.
        self._change_seq = 0
        # A change set we could not enumerate (shell tools, truncated detection)
        # rules out folding known paths in: only a rescan can cover it.
        self._unknown_changes = False

    @property
    def installed(self) -> bool:
        """Whether a snapshot is available for queries."""
        return self.snapshot is not None

    def query(self, query: str, *, limit: int = _DEFAULT_QUERY_LIMIT) -> list[ProjectPathSuggestion]:
        """Answer from the installed snapshot only; never scans."""
        snapshot = self.snapshot
        if snapshot is None:
            return []
        return snapshot.query(query, limit=limit)

    def head(self, *, limit: int = _DEFAULT_QUERY_LIMIT) -> list[ProjectPathSuggestion]:
        """Answer the empty query from the installed snapshot only."""
        snapshot = self.snapshot
        if snapshot is None:
            return []
        return snapshot.head(limit=limit)

    def install(self, scan: ProjectPathScanResult) -> None:
        """Install *scan* as the snapshot of this root."""
        self._install(scan, ProjectPathIndex.build(scan))

    def mark_dirty(self, changed: Iterable[str] | None = None) -> None:
        """Record that the snapshot may be stale.

        ``changed`` is the set of absolute paths the change came from; pass
        ``None`` when the change set is unknown, which forces a full rebuild.
        An empty set means the caller observed no change, so nothing is recorded.
        """
        if changed is None:
            self.dirty = True
            self._change_seq += 1
            self._unknown_changes = True
            self.delta.clear()
            return
        paths = {os.path.abspath(path) for path in changed}
        if not paths:
            return
        self.dirty = True
        self._change_seq += 1
        self.delta.update(paths)

    def apply_changes(self, paths: Iterable[str] = ()) -> bool:
        """Fold *paths* and every pending delta into the installed snapshot.

        Nothing is written until the whole batch is known to be representable, so
        a pending delta from an earlier failed batch is folded with this one
        instead of being dropped.

        Returns ``False`` when this root cannot be kept correct in memory — no
        snapshot, a truncated scan, an unenumerable change set, a rebuild already
        in flight, a directory, or a path whose visibility needs git evidence —
        so the caller upgrades to a full rebuild instead of guessing.
        """
        snapshot = self.snapshot
        if snapshot is None or snapshot.truncated or self._unknown_changes or self._building():
            return False

        pending = set(self.delta)
        pending.update(os.path.abspath(path) for path in paths)

        removals: list[str] = []
        additions: list[str] = []
        for path in pending:
            relative = self._relative_path(path)
            if relative is None:
                continue
            if os.path.isdir(path):
                return False
            if os.path.exists(path):
                if relative in self._file_set:
                    continue
                if not self._addition_is_visible(relative):
                    return False
                additions.append(relative)
            elif relative in self._file_set:
                removals.append(relative)

        if self._file_budget > 0 and len(self._files) + len(additions) > self._file_budget:
            # The scan caps file discovery; adding past it belongs to a rescan.
            return False

        if not removals and not additions:
            if self.dirty and not pending:
                # Stale with nothing pending to fold: only a rescan can prove the
                # snapshot current, so do not report it clean.
                return False
            self.dirty = False
            self.delta.clear()
            return True

        for relative in removals:
            self._remove_file(relative)
        for relative in additions:
            self._add_file(relative)
        self._reinstall_rows()
        return True

    def discard(self) -> None:
        """Drop the snapshot: the next ``@`` cold-builds for the new context."""
        self.epoch += 1
        self.snapshot = None
        self.dirty = False
        self.delta.clear()
        self._unknown_changes = False
        self._files = []
        self._file_set = set()
        self._directory_counts = {}
        self._source_truncations = {}
        self._file_budget = 0
        self._suggestion_budget = 0
        self._ordering = "path-sorted"

    async def refresh(self, scan: ScanProjectPaths | None = None) -> bool:
        """Rebuild the snapshot in the background; ``True`` when one was installed.

        Single-flight: concurrent callers await the same rebuild.  A failure
        keeps the installed snapshot (queries keep answering) and leaves
        ``dirty`` set so the next lifecycle event retries.
        """
        task = self.build_task
        if task is None or task.done():
            task = asyncio.ensure_future(self._rebuild(scan or _default_scan))
            self.build_task = task
        return await task

    def _building(self) -> bool:
        task = self.build_task
        return task is not None and not task.done()

    async def _rebuild(self, scan: ScanProjectPaths) -> bool:
        epoch = self.epoch
        change_seq = self._change_seq
        # Everything recorded before the scan starts is in its result; only what
        # arrives while it runs has to survive the install.
        covered_delta = set(self.delta)
        covered_unknown = self._unknown_changes
        try:
            result = await scan(self.root)
            index = await asyncio.to_thread(ProjectPathIndex.build, result)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Workspace path index rebuild failed for %s", self.root, exc_info=True)
            self.dirty = True
            return False
        if self.epoch != epoch:
            return False
        arrived_delta = set(self.delta) - covered_delta
        arrived_unknown = self._unknown_changes and not covered_unknown
        self._install(result, index)
        if self._change_seq != change_seq:
            # The scan may predate these changes, so the snapshot is not their
            # clean baseline: keep them pending for the next fold or rescan.
            self.delta.update(arrived_delta)
            self._unknown_changes = arrived_unknown
            self.dirty = True
        return True

    def _install(self, scan: ProjectPathScanResult, index: ProjectPathIndex) -> None:
        self.snapshot = index
        self.epoch += 1
        self._files = [suggestion.path for suggestion in scan.paths if suggestion.kind == "file"]
        self._file_set = set(self._files)
        self._directory_counts = _directory_reference_counts(self._files)
        self._source_truncations = dict(scan.source_truncations)
        self._file_budget = scan.file_budget
        self._suggestion_budget = scan.suggestion_budget
        self._ordering = scan.ordering
        self.delta.clear()
        self._unknown_changes = False
        self.dirty = False

    def _reinstall_rows(self) -> None:
        """Re-materialize the snapshot from the maintained file list."""
        rows = self._materialize_rows()
        suggestions_truncated = len(rows) > self._suggestion_budget
        if suggestions_truncated:
            rows = rows[: self._suggestion_budget]
        source_truncations = dict(self._source_truncations)
        if suggestions_truncated:
            source_truncations["suggestions"] = True
        scan = ProjectPathScanResult(
            root=self.root,
            paths=rows,
            file_count=sum(1 for row in rows if row.kind == "file"),
            suggestion_count=len(rows),
            truncated=any(source_truncations.values()),
            file_budget=self._file_budget,
            suggestion_budget=self._suggestion_budget,
            source_truncations=source_truncations,
            ordering=self._ordering,
        )
        self._install(scan, ProjectPathIndex.build(scan))

    def _materialize_rows(self) -> list[ProjectPathSuggestion]:
        """Return file rows plus one row per live directory, in insertion order."""
        rows: list[ProjectPathSuggestion] = []
        emitted: set[str] = set()
        for relative_path in self._files:
            for directory in _ancestor_directories(relative_path):
                if directory in emitted or self._directory_counts.get(directory, 0) <= 0:
                    continue
                emitted.add(directory)
                rows.append(ProjectPathSuggestion(path=directory, kind="directory"))
            rows.append(ProjectPathSuggestion(path=relative_path, kind="file"))
        return rows

    def _add_file(self, relative_path: str) -> None:
        self._files.append(relative_path)
        self._file_set.add(relative_path)
        for directory in _ancestor_directories(relative_path):
            self._directory_counts[directory] = self._directory_counts.get(directory, 0) + 1

    def _remove_file(self, relative_path: str) -> None:
        with contextlib.suppress(ValueError):
            self._files.remove(relative_path)
        self._file_set.discard(relative_path)
        for directory in _ancestor_directories(relative_path):
            remaining = self._directory_counts.get(directory, 0) - 1
            if remaining > 0:
                self._directory_counts[directory] = remaining
            else:
                self._directory_counts.pop(directory, None)

    def _relative_path(self, path: str) -> str | None:
        """Return *path* relative to this root in ``/`` form, or ``None`` when outside it."""
        try:
            relative = os.path.relpath(os.path.abspath(path), os.path.abspath(self.root))
        except ValueError:
            return None
        relative = relative.replace(os.sep, "/")
        if relative == ".." or relative.startswith("../"):
            return None
        return relative

    @staticmethod
    def _addition_is_visible(relative_path: str) -> bool:
        """Whether a new path is in the visible set whatever git says.

        The visible set is the union of the ripgrep, tracked, dot-entry and
        ignored-visible passes, so a non-dot path outside the noise top-level
        directories with a referenceable extension is listed in every case.
        Dot paths need git evidence and noise/blacklisted paths are listed only
        when they are not ignored — both are undeterminable here, so they fall
        back to a rescan instead of being guessed.
        """
        return _include_ignored_visible_path(relative_path) and not _has_non_referenceable_file_extension(relative_path)

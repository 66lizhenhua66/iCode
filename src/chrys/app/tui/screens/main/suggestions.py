# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Suggestion system — commands, files, agent profiles, models, and prompt history."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from rich.text import Text
from textual.content import Content

from chrys.app.tui.i18n import render_str
from chrys.app.tui.screens.main.buddy_command import BuddyCommandController
from chrys.app.tui.screens.main.commands import MainSlashCommandRegistry, SlashCommandActionPort
from chrys.app.tui.screens.main.model_indicator import is_model_selection_locked
from chrys.app.tui.screens.main.ports import BuddyCommandView, StartWorker, SuggestionPopupView
from chrys.app.tui.screens.main.state import MainScreenServices, MainScreenState
from chrys.app.tui.support.gc_freeze import GcFreezeBlockReason
from chrys.app.tui.support.prompt_history import DEFAULT_MAX_ENTRIES as _PROMPT_HISTORY_SEARCH_LIMIT
from chrys.app.tui.support.prompt_search import (
    LINE_BREAK,
    LINE_SEPARATOR_RE,
    PromptMatch,
    PromptSearchIndex,
    PromptSearchResult,
)
from chrys.app.tui.widgets.chrome.commands import (
    SlashCommandDef,
    is_slash_command_candidate,
)
from chrys.app.tui.widgets.chrome.file_index import ProjectPathIndex, WorkspacePathIndex
from chrys.app.tui.widgets.chrome.suggestion_list import SuggestionItem
from chrys.foundation.i18n import MessageRef, msg
from chrys.foundation.i18n.formatting import format_message, sanitize_legacy_scalar
from chrys.foundation.platform import safe_getcwd
from chrys.service.mutations.types import MutationProvenance, MutationSource
from chrys.service.profiles.models.schema import is_model_profile_selectable

_FILE_QUERY_DEBOUNCE_SECONDS = 0.016
# How many scans one background rebuild may chain before leaving the rest of the
# change set dirty for the next lifecycle event.
_FILE_INDEX_REBUILD_SCANS = 3
_FILE_QUERY_LIMIT = 30
# The list shows the best this many prompts; search covers every stored prompt.
_PROMPT_HISTORY_DISPLAY_LIMIT = 100
# A row shows one line and the highlighted row scrolls it, so a label longer than
# this only costs layout and marquee work on every keystroke; selection still
# inserts the whole prompt.
_PROMPT_HISTORY_LABEL_MAX_CHARS = 300
# Preview width used before the list has been laid out.
_PROMPT_HISTORY_FALLBACK_CELLS = 80
# The index outlives the list for an instant reopen only while the prompts hold
# at most this many characters; it keeps about three copies of their text.
_PROMPT_HISTORY_CACHE_MAX_CHARS = 2 * 1024 * 1024

_SUGGESTIONS_FILES_TITLE = msg("tui.suggestions.files_title", fallback="Files under {root}")
_SUGGESTIONS_COMMANDS_TITLE = msg("tui.suggestions.commands_title", fallback="Commands")
_SUGGESTIONS_AGENTS_TITLE = msg("tui.suggestions.agents_title", fallback="Agents")
_SUGGESTIONS_MODELS_TITLE = msg("tui.suggestions.models_title", fallback="Models")
_SUGGESTIONS_PROMPT_HISTORY_TITLE = msg("tui.suggestions.prompt_history_title", fallback="Prompt History")
_SUGGESTIONS_PROMPT_HISTORY_TITLE_MATCHES = msg(
    "tui.suggestions.prompt_history_title_matches",
    fallback="Prompt History · {count} match",
    plural_fallback="Prompt History · {count} matches",
)
_SUGGESTIONS_PROMPT_HISTORY_TITLE_SHOWN_MATCHES = msg(
    "tui.suggestions.prompt_history_title_shown_matches",
    fallback="Prompt History · {shown} of {count} match",
    plural_fallback="Prompt History · {shown} of {count} matches",
)
_SUGGESTIONS_PROMPT_HISTORY_EMPTY = msg("tui.suggestions.prompt_history_empty", fallback="No prompt history yet.")
_SUGGESTIONS_PROMPT_HISTORY_DISABLED = msg(
    "tui.suggestions.prompt_history_disabled",
    fallback="Prompt history is off. Turn on Save prompt history in settings and restart to use it.",
)
_SUGGESTIONS_PROMPT_HISTORY_NO_MATCH = msg(
    "tui.suggestions.prompt_history_no_match",
    fallback="No matching prompts. Try changing or clearing the input.",
)
_SUGGESTIONS_SYSTEM_COMMANDS = msg("tui.suggestions.system_commands", fallback="System Commands")
_SUGGESTIONS_LOADED_SKILLS = msg("tui.suggestions.loaded_skills", fallback="Loaded Skills")
_SUGGESTIONS_SHADOWED = msg("tui.suggestions.shadowed", fallback="shadowed by /{name}")
_SUGGESTIONS_BOUNDED_INDEX = msg(
    "tui.suggestions.bounded_index",
    fallback="More files exist outside this bounded index",
)
_SUGGESTIONS_INDEX_COUNTS = msg(
    "tui.suggestions.index_counts",
    fallback="{file_count} files / {suggestion_count} rows indexed",
)
_COMMAND_DISABLED_TITLE = msg("tui.commands.title.disabled", fallback="Command Disabled")
_COMMAND_UNAVAILABLE_RUNNING = msg(
    "tui.commands.unavailable_while_running",
    fallback="/{name} is not available while agent is running",
)

if TYPE_CHECKING:
    from textual.worker import Worker

    from chrys.app.tui.i18n import LocaleController
    from chrys.app.tui.widgets.chrome.file_scanner import ProjectPathSuggestion
    from chrys.foundation.events.types import RuntimeSkillDetails


def _newest_unique(history: Sequence[str]) -> list[str]:
    """Return *history* (oldest first) newest first, keeping only the newest copy of each prompt."""
    seen: set[str] = set()
    unique: list[str] = []
    for prompt in reversed(history):
        if prompt not in seen:
            seen.add(prompt)
            unique.append(prompt)
    return unique


@dataclass(frozen=True, slots=True)
class SuggestionCallbacks:
    """Screen-owned effects required by suggestions."""

    notify_warning: Callable[[MessageRef | str, MessageRef | str, float | None], None]
    start_worker: StartWorker
    submit_user_text: Callable[[str], object]
    start_agent_profile_switch: Callable[[str], object]
    start_model_profile_switch: Callable[[str], object]


class SuggestionHandler:
    """Manages the suggestion list popup lifecycle for /, @, #, and $ triggers.

    Owns suggestion-related state and the slash command definitions.
    Constructed once by the main-screen owner.
    """

    def __init__(
        self,
        *,
        state: MainScreenState,
        services: MainScreenServices,
        view: SuggestionPopupView,
        command_actions: SlashCommandActionPort,
        callbacks: SuggestionCallbacks,
        buddy_view: BuddyCommandView,
        locale_controller: LocaleController | None = None,
    ) -> None:
        self._state = state
        self._services = services
        self._view = view
        self._callbacks = callbacks
        self._locale_controller = locale_controller
        self._suggestion_mode: str | None = None
        self._file_index: WorkspacePathIndex | None = None
        self._file_query_revision = 0
        self._file_latest_query = ""
        self._file_query_pending: tuple[str, int] | None = None
        self._file_query_worker: Worker[None] | None = None
        self._file_warmup_requested_root: str | None = None
        self._file_warmup_worker: Worker[None] | None = None
        self._file_rebuild_worker: Worker[None] | None = None
        self._prompt_history_revision = 0
        self._prompt_history_index: PromptSearchIndex | None = None
        self._prompt_history_query_revision = 0
        self._prompt_history_pending: tuple[str, int] | None = None
        self._prompt_history_query_worker: Worker[None] | None = None
        self._prompt_history_enabled = True
        # Whether this opening's load finished: a cached index can lack a prompt just submitted.
        self._prompt_history_loaded = False
        # The query and index whose rows are shown; showing them again keeps the selected row.
        self._prompt_history_shown: tuple[str, PromptSearchIndex] | None = None
        # Survives closing the list so reopening it re-normalizes only new prompts;
        # None when the history is too large to keep in memory.
        self._prompt_history_index_cache: PromptSearchIndex | None = None
        # Up/Down and Enter/Tab pressed while a newer query is still ranking or
        # the history still loading, in order; replayed once both are done.
        self._prompt_history_deferred_keys: list[tuple[str, str]] = []
        self._slash_commands: list[SlashCommandDef] = []
        self._active_command: SlashCommandDef | None = None
        self._buddy_command = BuddyCommandController(
            buddy_view,
            render_message=self._render_message,
        )
        self._command_registry = MainSlashCommandRegistry(
            actions=command_actions,
            buddy=self._buddy_command,
            render_message=self._render_message,
        )

    # -------------------------------------------------------------- #
    # Slash command definitions
    # -------------------------------------------------------------- #

    def build_slash_commands(self) -> list[SlashCommandDef]:
        """Build slash command list with bound actions."""
        self._slash_commands = self._command_registry.build()
        return self._slash_commands

    @property
    def buddy_command(self) -> BuddyCommandController:
        """Return the stateful /buddy command controller."""
        return self._buddy_command

    def gc_freeze_block_reason(self) -> GcFreezeBlockReason | None:
        """Block while any suggestion popup is visible."""
        return GcFreezeBlockReason.SUGGESTIONS_VISIBLE if self._view.suggestions_active else None

    def prepare_for_gc_freeze(self) -> None:
        """Keep the acyclic warm file cache installed across freezes."""

    def after_gc_freeze(self) -> None:
        """Suggestion state has no mutable cyclic container to renew."""

    def abort_gc_freeze(self) -> None:
        """Suggestion state has no detached container to restore."""

    def dispatch_slash_command(self, text: str) -> bool:
        """Try to match and execute a slash command. Returns True if handled."""
        parts = text[1:].split(None, 1)
        if not parts:
            return False
        name = parts[0]
        arg = parts[1] if len(parts) > 1 else ""
        for cmd in self._slash_commands:
            if cmd.name == name or name in cmd.aliases:
                if cmd.name in self._disabled_commands():
                    self._callbacks.notify_warning(
                        _COMMAND_UNAVAILABLE_RUNNING.bind(name=cmd.name),
                        _COMMAND_DISABLED_TITLE.bind(),
                        3,
                    )
                    return True
                cmd.action(arg)
                return True
        return False

    # -------------------------------------------------------------- #
    # Suggestion display and filtering
    # -------------------------------------------------------------- #

    def _disabled_commands(self) -> set[str]:
        """Return command names that should be disabled in the current state."""
        if not self._state.run.agent_running:
            return set()
        return {cmd.name for cmd in self._slash_commands if not cmd.allow_while_running}

    def _visible_slash_commands(self) -> list[SlashCommandDef]:
        """Return slash commands that should appear in the suggestion list."""
        for cmd in self._slash_commands:
            if cmd.name == "resume":
                cmd.hidden = not self._services.state_store
        return [cmd for cmd in self._slash_commands if not cmd.hidden]

    def _show_suggestions(
        self,
        mode: str,
        items: Sequence[SuggestionItem | tuple[str, str | Content]],
        disabled: set[str] | None = None,
    ) -> None:
        """Show the suggestion list, optionally hiding the status bar."""
        if mode != "files":
            self._invalidate_file_query_results()
        if mode != "history":
            self._invalidate_prompt_history_results()

        self._suggestion_mode = mode
        self._view.show_suggestions(mode, list(items), disabled=disabled, title=self._suggestion_title(mode))

    def _show_suggestions_loading(self, mode: str) -> None:
        """Open *mode* immediately with no selectable rows until results are ready."""
        if mode != "files":
            self._invalidate_file_query_results()
        if mode != "history":
            self._invalidate_prompt_history_results()

        self._suggestion_mode = mode
        self._view.show_suggestions_loading(mode, title=self._suggestion_title(mode))

    def _suggestion_title(self, mode: str) -> str:
        """Border title for the suggestion popup; files mode names its scan root."""
        if mode == "files":
            return self._render_message(_SUGGESTIONS_FILES_TITLE.bind(root=self._current_file_root()))
        definition = {
            "commands": _SUGGESTIONS_COMMANDS_TITLE,
            "agents": _SUGGESTIONS_AGENTS_TITLE,
            "models": _SUGGESTIONS_MODELS_TITLE,
            "history": _SUGGESTIONS_PROMPT_HISTORY_TITLE,
        }.get(mode)
        return "" if definition is None else self._render_message(definition.bind())

    def _render_message(self, reference: MessageRef) -> str:
        controller = self._locale_controller
        if controller is None:
            return format_message(reference)
        return render_str(controller.localizer, reference)

    def dismiss_suggestions(self) -> None:
        """Hide the suggestion list and restore status bar."""
        self._invalidate_file_query_results()
        self._invalidate_prompt_history_results()
        self._view.hide_suggestions()
        self._suggestion_mode = None

    def _invalidate_prompt_history_results(self) -> None:
        """Invalidate an in-flight prompt-history load or query and drop loaded entries."""
        self._prompt_history_revision += 1
        self._prompt_history_index = None
        self._prompt_history_query_revision += 1
        self._prompt_history_pending = None
        self._prompt_history_deferred_keys = []
        self._prompt_history_loaded = False
        self._prompt_history_shown = None

    @staticmethod
    def _prompt_history_label(match: PromptMatch) -> tuple[str, tuple[tuple[int, int], ...]]:
        """Render the preview windows of a ranked prompt as one row and locate its matches in it.

        Line breaks show as ``↵`` (a trailing one is dropped, as ``splitlines``
        does), control characters are replaced, and an ellipsis marks text cut
        before, between or after the windows. The last window runs on to its
        tail, past the row's edge, so a highlighted row scrolls on from the
        match. The spans are ranges of the returned label.
        """
        prompt = match.prompt
        parts: list[str] = []
        spans: list[tuple[int, int]] = []
        length = 0

        def add(text: str, *, matched: bool = False) -> None:
            nonlocal length
            if matched:
                spans.append((length, length + len(text)))
            parts.append(text)
            length += len(text)

        def add_text(low: int, high: int) -> None:
            position = low
            for span_low, span_high in match.spans:
                span_low, span_high = max(span_low, position), min(span_high, high)
                if span_low >= span_high:
                    continue
                if span_low > position:
                    add(sanitize_legacy_scalar(prompt[position:span_low]))
                add(sanitize_legacy_scalar(prompt[span_low:span_high]), matched=True)
                position = span_high
            if position < high:
                add(sanitize_legacy_scalar(prompt[position:high]))

        previous_end = 0
        windows = [*match.windows[:-1], (match.windows[-1][0], match.tail_end)]
        for start, end in windows:
            if start > previous_end or (start > 0 and not parts):
                add("…")
            position = start
            for separator in LINE_SEPARATOR_RE.finditer(prompt, start, end):
                add_text(position, separator.start())
                position = separator.end()
                # The LF of a CRLF whose CR fell before the window was already shown with it.
                if position < len(prompt) and prompt[separator.start() - 1 : separator.end()] != "\r\n":
                    add(LINE_BREAK)
            add_text(position, end)
            previous_end = end
        if previous_end < len(prompt):
            add("…")
        return "".join(parts), tuple(spans)

    def _enqueue_prompt_history_query(self, query: str) -> None:
        """Record *query* as the latest history search and make sure a query loop will run it.

        Ranking scans every loaded entry and entries have no length limit, so it
        runs off the event loop; keystrokes that arrive meanwhile coalesce into
        the newest pending query, and results for any older query are dropped.
        """
        self._prompt_history_query_revision += 1
        self._prompt_history_pending = (query, self._prompt_history_query_revision)
        if self._prompt_history_index is None or self._prompt_history_query_active():
            return
        self._prompt_history_query_worker = self._view.run_suggestion_worker(
            self._run_prompt_history_query_loop(),
            name="prompt-history-query",
            group="prompt-history",
        )

    def _prompt_history_query_active(self) -> bool:
        """Return whether a history query loop is running; at most one ever is."""
        worker = self._prompt_history_query_worker
        return worker is not None and not worker.is_finished

    async def _run_prompt_history_query_loop(self) -> None:
        """Rank the newest pending history query in a thread until none is pending.

        Closing the list or a newer query makes the search in its thread stop
        at the next prompt, and the loop goes on to whatever is pending then,
        including the query of a list reopened meanwhile.
        """
        while self._prompt_history_pending is not None:
            query, revision = self._prompt_history_pending
            self._prompt_history_pending = None
            index = self._prompt_history_index
            if index is None:
                return
            result = await asyncio.to_thread(
                index.search,
                query,
                limit=_PROMPT_HISTORY_DISPLAY_LIMIT,
                cells=self._view.suggestion_row_cells() or _PROMPT_HISTORY_FALLBACK_CELLS,
                head_chars=_PROMPT_HISTORY_LABEL_MAX_CHARS,
                # Read from the search thread; the int read needs no lock.
                should_stop=lambda revision=revision: revision != self._prompt_history_query_revision,
            )
            if self._prompt_history_result_current(revision, index):
                items = await asyncio.to_thread(self._prompt_history_items, result.matches)
                if not self._prompt_history_result_current(revision, index):
                    continue
                self._show_prompt_history_matches(index, items, result, query=query)
                if self._prompt_history_pending is None and self._prompt_history_loaded:
                    self._replay_deferred_prompt_history_keys()

    def _prompt_history_result_current(self, revision: int, index: PromptSearchIndex) -> bool:
        """Return whether a result for *revision* on *index* may still update the open history list."""
        return (
            revision == self._prompt_history_query_revision
            and index is self._prompt_history_index
            and self._view.is_attached
            and self._suggestion_mode == "history"
        )

    @classmethod
    def _prompt_history_items(cls, matches: list[PromptMatch]) -> list[SuggestionItem]:
        """Build the rows for ranked *matches*; runs in a thread, off the event loop."""
        items: list[SuggestionItem] = []
        for match in matches:
            label, spans = cls._prompt_history_label(match)
            items.append(
                SuggestionItem(value=match.prompt, label=label, kind="history", marquee_start=0, match_spans=spans)
            )
        return items

    def _prompt_history_title(self, result: PromptSearchResult, *, shown: int, filtered: bool) -> str:
        """Name the list, and while a query filters it, how many prompts matched."""
        if not filtered or result.total == 0:
            return self._suggestion_title("history")
        if result.total > shown:
            return self._render_message(
                _SUGGESTIONS_PROMPT_HISTORY_TITLE_SHOWN_MATCHES.bind(count=result.total, shown=str(shown))
            )
        return self._render_message(_SUGGESTIONS_PROMPT_HISTORY_TITLE_MATCHES.bind(count=result.total))

    def _show_prompt_history_matches(
        self, index: PromptSearchIndex, items: list[SuggestionItem], result: PromptSearchResult, *, query: str
    ) -> None:
        """Show ranked rows from *index*, saying why when there are none.

        Rows refit to a new width keep the selected prompt and scroll position,
        so a held Enter picks what the user saw; a new query or newly loaded
        prompts start again from the best match.
        """
        shown = self._prompt_history_shown
        refit = shown is not None and shown[0] == query and shown[1] is index
        self._prompt_history_shown = (query, index)
        if not self._prompt_history_enabled:
            empty = _SUGGESTIONS_PROMPT_HISTORY_DISABLED
        elif len(index):
            empty = _SUGGESTIONS_PROMPT_HISTORY_NO_MATCH
        else:
            empty = _SUGGESTIONS_PROMPT_HISTORY_EMPTY
        self._view.update_suggestions(
            items,
            title=self._prompt_history_title(result, shown=len(items), filtered=bool(query.split())),
            empty_text=self._render_message(empty.bind()),
            keep_selection=refit,
        )

    def _defer_prompt_history_key(self, action: str, argument: str) -> bool:
        """Hold a list key until the newest history results are shown; return whether it was held.

        Results arrive after the keystroke that asked for them, and the first
        ones after the history loads, so moving or selecting at once would act
        on rows of an older query or on none at all. Rows from a cached index
        wait for the load too, since it can lack a prompt submitted just now.
        """
        if self._suggestion_mode != "history":
            return False
        loading = self._prompt_history_index is None or not self._prompt_history_loaded
        if not loading and self._prompt_history_pending is None and not self._prompt_history_query_active():
            return False
        self._prompt_history_deferred_keys.append((action, argument))
        return True

    def _replay_deferred_prompt_history_keys(self) -> None:
        """Apply held Up/Down and Enter/Tab, in order, to the rows just shown."""
        keys, self._prompt_history_deferred_keys = self._prompt_history_deferred_keys, []
        for action, argument in keys:
            if action == "move":
                self._view.move_suggestion_cursor(argument)
            else:
                # Selecting closes the list, so nothing held after it applies.
                self._view.select_highlighted_suggestion(execute=argument == "execute")
                return

    def start_prompt_history(self) -> int:
        """Open prompt history synchronously and return its load revision.

        With prompts indexed by an earlier opening, those are searched at once
        while the load checks for new ones, so reopening shows no loading state.
        """
        self._invalidate_prompt_history_results()
        revision = self._prompt_history_revision
        self._show_suggestions_loading("history")
        cached = self._prompt_history_index_cache
        if cached is not None:
            self._prompt_history_index = cached
            self._enqueue_prompt_history_query(self._view.input_value())
        return revision

    async def show_prompt_history_async(self, *, revision: int | None = None) -> None:
        """Load the newest global prompt-history entries and show those matching the input text."""
        if revision is None:
            revision = self.start_prompt_history()
        elif revision != self._prompt_history_revision or self._suggestion_mode != "history":
            return
        history = await self._view.load_prompt_history(max_entries=_PROMPT_HISTORY_SEARCH_LIMIT)
        if (
            revision != self._prompt_history_revision
            or not self._view.is_attached
            or self._suggestion_mode != "history"
        ):
            return
        self._prompt_history_enabled = self._view.prompt_history_enabled()
        prompts = _newest_unique(history)
        index = await asyncio.to_thread(partial(PromptSearchIndex, prompts, previous=self._prompt_history_index_cache))
        if (
            revision != self._prompt_history_revision
            or not self._view.is_attached
            or self._suggestion_mode != "history"
        ):
            return
        self._prompt_history_loaded = True
        shown = self._prompt_history_index
        if shown is not None and index.same_prompts(shown):
            # Nothing changed: keep the index being searched, whose results must still be shown.
            if self._prompt_history_pending is None and not self._prompt_history_query_active():
                self._replay_deferred_prompt_history_keys()
            return
        self._prompt_history_index = index
        small = sum(len(prompt) for prompt in prompts) <= _PROMPT_HISTORY_CACHE_MAX_CHARS
        self._prompt_history_index_cache = index if small else None
        # Typing during the load was recorded only as input text; search what is there now.
        self._enqueue_prompt_history_query(self._view.input_value())

    # -------------------------------------------------------------- #
    # Trigger handlers (called by thin @on forwarders in MainScreen)
    # -------------------------------------------------------------- #

    def on_slash_triggered(self) -> None:
        """Show slash command suggestions."""
        disabled = self._disabled_commands()
        items = self._command_suggestion_items(self._visible_slash_commands(), disabled)
        items.extend(self._runtime_skill_suggestion_items())
        self._show_suggestions("commands", items, disabled=disabled)

    def _command_suggestion_items(
        self,
        commands: list[SlashCommandDef],
        disabled: set[str],
    ) -> list[SuggestionItem]:
        """Build grouped slash-command suggestion items."""
        items: list[SuggestionItem] = []
        for cmd in commands:
            description = self._render_message(cmd.description)
            items.append(
                SuggestionItem(
                    value=cmd.name,
                    # The second separator space carries the dim style: when the
                    # row is highlighted (reverse video), the bright-to-gray
                    # boundary then splits evenly between name and description.
                    label=Content.assemble(f"/{cmd.name} ", (" " + description, "dim")),
                    section=self._render_message(_SUGGESTIONS_SYSTEM_COMMANDS.bind()),
                    kind="command",
                    disabled=cmd.name in disabled,
                    marquee_start=len(f"/{cmd.name}  ") if description else None,
                )
            )
        return items

    def _runtime_skill_suggestion_items(
        self,
        skills: list[RuntimeSkillDetails] | None = None,
    ) -> list[SuggestionItem]:
        """Build grouped runtime-skill suggestion items."""
        items: list[SuggestionItem] = []
        source_skills = skills if skills is not None else self._state.runtime.details.skill_details
        for skill in source_skills:
            shadowing_command = self._command_for_token(skill.name)
            items.append(
                SuggestionItem(
                    value=skill.name,
                    label=self._runtime_skill_label(skill.name, skill.description),
                    section=self._render_message(_SUGGESTIONS_LOADED_SKILLS.bind()),
                    kind="skill",
                    disabled=shadowing_command is not None,
                    disabled_reason=(
                        self._render_message(_SUGGESTIONS_SHADOWED.bind(name=shadowing_command.name))
                        if shadowing_command is not None
                        else ""
                    ),
                    marquee_start=len(f"/{skill.name}  ") if skill.description.strip() else None,
                )
            )
        return items

    @staticmethod
    def _runtime_skill_label(name: str, description: str) -> Content:
        """Build a literal-text label from user-authored skill metadata."""
        description = description.strip()
        label = Text.assemble(f"/{name}")
        if description:
            # Second separator space styled dim — see _command_suggestion_items.
            label.append(" ")
            label.append(Text(" " + description, style="dim"))
        return Content.from_rich_text(label)

    def _command_for_token(self, token: str) -> SlashCommandDef | None:
        """Return the slash command that would handle *token*, including aliases."""
        for cmd in self._slash_commands:
            if cmd.name == token or token in cmd.aliases:
                return cmd
        return None

    def on_file_triggered(self) -> None:
        """Show file search suggestions."""
        self._file_latest_query = ""
        self._invalidate_file_query_results()
        index = self._file_index_for(self._current_file_root())
        if index.installed:
            # Fast path: a resident snapshot answers the trigger with no scan.
            self._show_suggestions(
                "files",
                self._file_suggestion_items(index.head(limit=_FILE_QUERY_LIMIT), index=index.snapshot),
            )
            return
        self._show_suggestions_loading("files")
        self._callbacks.start_worker(self.show_file_suggestions_async)

    def on_agent_triggered(self) -> None:
        """Show agent profile suggestions."""
        if self._state.run.agent_running or self._state.run.agent_loading or self._services.execution_busy():
            return
        items, disabled = self._get_agent_items()
        self._show_suggestions("agents", items, disabled=disabled)

    def _get_agent_items(self) -> tuple[list[SuggestionItem], set[str]]:
        """Build suggestion items and disabled set from available agent profiles."""
        registry = self._services.agent_registry
        if registry is None:
            return [], set()
        profiles = registry.list_profiles()
        items: list[SuggestionItem] = []
        disabled: set[str] = set()
        for p in profiles:
            label = p.display_name or p.name
            is_current = label == self._state.runtime.profile
            prefix = "◦ " if is_current else "  "
            items.append(
                SuggestionItem(
                    value=p.name,
                    label=Content.assemble(f"{prefix}{label} ", (" " + p.description, "dim")),
                    marquee_start=len(f"{prefix}{label}  ") if p.description else None,
                )
            )
            if is_current:
                disabled.add(p.name)
        return items, disabled

    def _agent_query_matches(self, text: str) -> tuple[list[str], dict[str, SuggestionItem], set[str]]:
        """Return agent names matching the current # query plus labels and disabled names."""
        query = text.lstrip("#\uff03")
        all_items, disabled = self._get_agent_items()
        item_by_name = {item.value: item for item in all_items}
        if not query:
            return [item.value for item in all_items], item_by_name, disabled

        from chrys.app.tui.widgets.chrome.file_scanner import fuzzy_filter

        all_names = [item.value for item in all_items]
        matches = [value for value in fuzzy_filter(query, all_names) if value in item_by_name]
        return matches, item_by_name, disabled

    def _agent_fallback_should_submit(self, text: str) -> bool:
        """Return whether an unselected # entry should fall through to chat submit."""
        matches, _label_by_name, disabled = self._agent_query_matches(text)
        if not matches:
            return True
        return any(match not in disabled for match in matches)

    def on_model_triggered(self) -> None:
        """Show selectable model-profile suggestions when model selection is unlocked."""
        if (
            self._state.run.agent_running
            or self._state.run.agent_loading
            or self._services.execution_busy()
            or self._model_selection_is_locked()
        ):
            return
        items, disabled = self._get_model_items()
        self._show_suggestions("models", items, disabled=disabled)

    def _model_selection_is_locked(self) -> bool:
        """Return whether runtime policy prevents changing the active model."""
        return is_model_selection_locked(
            self._state.runtime.details.model,
            runtime_confirmed=self._state.runtime.details_confirmed,
        )

    def _current_model_profile_id(self) -> str:
        """Return the best-known current model-profile id for suggestion styling."""
        if self._state.runtime.details_confirmed:
            return self._state.runtime.details.model.profile_id
        return self._services.active_model_profile_id

    def _get_model_items(self) -> tuple[list[SuggestionItem], set[str]]:
        """Build suggestion items and disabled set from selectable model profiles."""
        registry = self._services.model_registry
        if registry is None:
            return [], set()
        current_profile_id = self._current_model_profile_id()
        items: list[SuggestionItem] = []
        disabled: set[str] = set()
        for profile in registry.list_profiles():
            if not is_model_profile_selectable(profile):
                continue
            is_current = profile.id == current_profile_id
            prefix = "◦ " if is_current else "  "
            items.append(
                SuggestionItem(
                    value=profile.id,
                    label=Content.assemble(f"{prefix}{profile.name} ", (" " + profile.model_id, "dim")),
                    kind="model",
                    marquee_start=len(f"{prefix}{profile.name}  ") if profile.model_id else None,
                )
            )
            if is_current:
                disabled.add(profile.id)
        return items, disabled

    def _model_query_matches(self, text: str) -> tuple[list[str], dict[str, SuggestionItem], set[str]]:
        """Return model profile ids matching the current $ query."""
        query = text.lstrip("$\uff04")
        all_items, disabled = self._get_model_items()
        item_by_id = {item.value: item for item in all_items}
        if not query:
            return [item.value for item in all_items], item_by_id, disabled

        registry = self._services.model_registry
        if registry is None:
            return [], item_by_id, disabled
        selectable_profiles = [profile for profile in registry.list_profiles() if profile.id in item_by_id]

        from chrys.app.tui.widgets.chrome.file_scanner import fuzzy_filter

        names = [profile.name for profile in selectable_profiles]
        matching_names = fuzzy_filter(query, names)
        ids_by_name: dict[str, list[str]] = {}
        for profile in selectable_profiles:
            ids_by_name.setdefault(profile.name, []).append(profile.id)
        matches = [ids_by_name[name].pop(0) for name in matching_names]
        return matches, item_by_id, disabled

    def _model_fallback_should_submit(self, text: str) -> bool:
        """Return whether an unselected $ entry should fall through to chat submit."""
        matches, _item_by_id, disabled = self._model_query_matches(text)
        if not matches:
            return True
        return any(match not in disabled for match in matches)

    async def show_file_suggestions_async(self) -> None:
        """Show file suggestions; ``on_file_triggered`` runs this in a screen worker.

        An installed snapshot answers immediately; only a root without one waits
        for the cold build, which runs in this worker rather than on input.
        """
        if self._suggestion_mode != "files" or not self._view.is_attached:
            return
        root = self._current_file_root()
        index = self._file_index_for(root)
        if index.installed:
            self._show_latest_file_query_or_head()
            return

        self._file_warmup_requested_root = root
        worker = self._file_warmup_worker
        if worker is None or worker.is_finished:
            worker = self._view.run_suggestion_worker(
                self._run_file_warmup_loop(),
                name="file-suggestion-warmup",
                group="file-suggestions",
            )
            self._file_warmup_worker = worker
        await worker.wait()

    def _current_file_root(self) -> str:
        """Return the cwd that file suggestions should be scoped to."""
        return self._state.workspace_marker.current_cwd or self._state.workspace.current_cwd or safe_getcwd()

    def _file_index_for(self, root: str) -> WorkspacePathIndex:
        """Return the resident index for *root*, replacing one kept for another root."""
        index = self._file_index
        if index is not None and index.root == root:
            return index
        if index is not None:
            index.discard()
        index = WorkspacePathIndex(root)
        self._file_index = index
        self._file_warmup_requested_root = None
        self._invalidate_file_query_results()
        return index

    def _file_snapshot(self) -> ProjectPathIndex | None:
        """Return the snapshot installed for the current root, without building one."""
        index = self._file_index
        if index is None or index.root != self._current_file_root():
            return None
        return index.snapshot

    def mark_file_index_dirty(self) -> None:
        """Mark the ``@`` snapshot stale when an agent turn ends.

        The popup keeps answering from the snapshot; the rebuild runs in the
        background and hot-swaps the snapshot only once it succeeded, so a turn
        that changed files costs the next ``@`` nothing.
        """
        index = self._file_index
        if index is None or not index.installed:
            return
        changed = self._turn_changed_paths()
        if changed is not None and index.apply_changes(changed):
            # A turn that only read files (empty set) also lands here: nothing to rebuild.
            return
        index.mark_dirty(changed)
        if index.dirty:
            self._schedule_file_index_rebuild(index)

    def invalidate_file_index(self) -> None:
        """Force a rescan after a rollback restored files behind the tracker's back.

        A rollback deletes the rolled-back periods from the mutation tracker, so
        the restored paths never reach :meth:`mark_file_index_dirty`, and a
        snapshot folded clean since then would keep listing a file the user just
        undid.  ``discard()`` would blank the popup until a cold build finishes,
        so the snapshot is marked dirty with an unknown change set instead: it
        keeps answering, and the rebuild hot-swaps the restored truth.
        """
        index = self._file_index
        if index is None or not index.installed:
            return
        index.mark_dirty()
        self._schedule_file_index_rebuild(index)

    def discard_file_index(self) -> None:
        """Drop the snapshot before another workspace or session takes over."""
        index = self._file_index
        self._file_index = None
        self._file_warmup_requested_root = None
        self._invalidate_file_query_results()
        if index is not None:
            index.discard()

    def _turn_changed_paths(self) -> set[str] | None:
        """Return the absolute paths the finished turn changed, or ``None`` when unknowable."""
        provider = self._services.engine_provider
        engine = provider() if provider is not None else None
        tracker = engine.mutation_tracker if engine is not None else None
        if tracker is None:
            return None
        period = tracker.current_period
        if period is None or period.detection_truncated:
            return None
        if any(
            mutation.source is MutationSource.SHELL and mutation.provenance is not MutationProvenance.FOREIGN
            for mutation in period.mutations
        ):
            # A shell command can create and delete files the tracker never sees.
            return None
        return tracker.get_changed_files_set()

    def _schedule_file_index_rebuild(self, index: WorkspacePathIndex) -> None:
        """Rebuild a stale snapshot off the input path."""
        if not self._view.is_attached:
            return
        worker = self._file_rebuild_worker
        if worker is not None and not worker.is_finished:
            return
        self._file_rebuild_worker = self._view.run_suggestion_worker(
            self._run_file_index_rebuild(index),
            name="file-suggestion-rebuild",
            group="file-suggestions",
        )

    async def _run_file_index_rebuild(self, index: WorkspacePathIndex) -> None:
        """Rebuild until the snapshot is provably current, within a bounded number of scans.

        A change that landed mid-scan leaves the installed snapshot dirty: an
        enumerable one folds in memory right here, an unknowable one forces one
        more scan.  The cap keeps a workspace that changes faster than it can be
        scanned from re-scanning forever; whatever is still pending stays dirty
        for the next lifecycle event.
        """
        for _attempt in range(_FILE_INDEX_REBUILD_SCANS):
            if not await index.refresh():
                return
            if self._file_index is not index:
                return
            settled = index.apply_changes()
            if self._suggestion_mode == "files":
                self._show_latest_file_query_or_head()
            if settled or not index.dirty:
                return

    async def _run_file_warmup_loop(self) -> None:
        """Coalesced cold build; a root that already has a snapshot never lands here."""
        while self._file_warmup_requested_root is not None:
            root = self._file_warmup_requested_root
            self._file_warmup_requested_root = None
            index = self._file_index_for(root)
            if index.installed:
                self._show_latest_file_query_or_head()
                continue
            if not self._file_warmup_context_current(root):
                continue

            installed = await index.refresh()
            if self._file_index is not index or not self._file_warmup_context_current(root):
                self._request_latest_file_warmup_if_active()
                continue
            if not installed:
                return
            self._show_latest_file_query_or_head()

    def _request_latest_file_warmup_if_active(self) -> None:
        """Request one follow-up warmup for the current root if file suggestions are still active."""
        if self._suggestion_mode == "files" and self._view.is_attached:
            self._file_warmup_requested_root = self._current_file_root()

    def _file_warmup_context_current(self, root: str) -> bool:
        """Return whether a completed cold build may still be shown for *root*."""
        return self._view.is_attached and self._suggestion_mode == "files" and self._current_file_root() == root

    def _file_suggestion_items(
        self,
        paths: list[ProjectPathSuggestion],
        *,
        index: ProjectPathIndex | None = None,
    ) -> list[SuggestionItem]:
        """Build suggestion rows for project file and directory paths."""
        items = [SuggestionItem(value=path.path, label=path.path, kind=path.kind) for path in paths]
        if index is not None and index.truncated:
            items.append(
                SuggestionItem(
                    value="__chrys_file_results_truncated__",
                    label=self._render_message(_SUGGESTIONS_BOUNDED_INDEX.bind()),
                    kind="status",
                    disabled=True,
                    disabled_reason=self._render_message(
                        _SUGGESTIONS_INDEX_COUNTS.bind(
                            file_count=f"{index.file_count:,}",
                            suggestion_count=f"{index.suggestion_count:,}",
                        )
                    ),
                )
            )
        return items

    def _show_latest_file_query_or_head(self) -> None:
        """Display the current file query, preserving immediate empty-query head behavior."""
        if self._suggestion_mode != "files" or not self._view.is_attached:
            return
        snapshot = self._file_snapshot()
        if snapshot is None:
            return
        query = self._file_latest_query
        if query:
            self._enqueue_file_query(query)
        else:
            self._invalidate_file_query_results()
            self._show_or_update_file_suggestions(snapshot.head(limit=_FILE_QUERY_LIMIT))

    def _show_or_update_file_suggestions(self, paths: list[ProjectPathSuggestion]) -> None:
        """Show or update the file suggestion list with *paths*."""
        if self._suggestion_mode != "files" or not self._view.is_attached:
            return

        items = self._file_suggestion_items(paths, index=self._file_snapshot())
        if self._view.suggestions_active:
            self._view.update_suggestions(items, title=self._suggestion_title("files"))
        else:
            self._show_suggestions("files", items)

    def _invalidate_file_query_results(self) -> None:
        """Invalidate pending/returned file query results."""
        self._file_query_revision += 1
        self._file_query_pending = None

    def _enqueue_file_query(self, query: str) -> None:
        """Record and schedule the latest file query."""
        self._file_latest_query = query
        self._file_query_revision += 1
        revision = self._file_query_revision
        if self._file_snapshot() is None:
            # Fast path only: without a snapshot the query waits for the cold build.
            return
        self._file_query_pending = (query, revision)
        if self._file_query_worker is None or self._file_query_worker.is_finished:
            self._file_query_worker = self._view.run_suggestion_worker(
                self._run_file_query_loop(),
                name="file-suggestion-query",
                group="file-suggestions",
            )

    async def _run_file_query_loop(self) -> None:
        """Debounced, coalesced file-query worker loop reading the installed snapshot."""
        while self._file_query_pending is not None:
            query, revision = self._file_query_pending
            self._file_query_pending = None
            await asyncio.sleep(_FILE_QUERY_DEBOUNCE_SECONDS)
            if self._file_query_pending is not None:
                continue

            index = self._file_index
            snapshot = self._file_snapshot()
            if index is None or snapshot is None:
                continue
            epoch = index.epoch
            matches = await asyncio.to_thread(snapshot.query, query, limit=_FILE_QUERY_LIMIT)
            if self._file_query_context_current(revision, index, epoch):
                self._apply_file_query_result(matches)

    def _file_query_context_current(self, revision: int, index: WorkspacePathIndex, epoch: int) -> bool:
        """Return whether a query result is still allowed to update the UI."""
        return (
            revision == self._file_query_revision
            and self._view.is_attached
            and self._suggestion_mode == "files"
            and self._file_index is index
            and index.epoch == epoch
        )

    def _apply_file_query_result(self, matches: list[ProjectPathSuggestion]) -> None:
        """Apply current file-query results to the suggestion list."""
        if not self._view.is_attached:
            return
        self._show_or_update_file_suggestions(matches)

    # -------------------------------------------------------------- #
    # Filtering as user types
    # -------------------------------------------------------------- #

    def on_text_changed(self, text: str) -> None:
        """Re-filter suggestion list as user types."""
        if self._suggestion_mode is None:
            return

        if self._suggestion_mode == "commands":
            if text not in ("/", "\uff0f") and not is_slash_command_candidate(text):
                self.dismiss_suggestions()
                return
            query = text.lstrip("/\uff0f")
            if " " in query:
                # Support second-level suggestions when user types "/cmd ..." directly.
                parts = query.split(None, 1)
                cmd_name = parts[0] if parts else ""
                sub_query = parts[1] if len(parts) > 1 else ""
                cmd = next(
                    (c for c in self._slash_commands if c.name == cmd_name or cmd_name in c.aliases),
                    None,
                )
                if not cmd or not cmd.subcommands:
                    self.dismiss_suggestions()
                    return
                self._suggestion_mode = "subcommands"
                self._active_command = cmd
                from chrys.app.tui.widgets.chrome.file_scanner import fuzzy_filter

                all_subs = cmd.subcommands()
                if sub_query:
                    matches = fuzzy_filter(sub_query, [v for v, _ in all_subs])
                    sub_by_value = dict(all_subs)
                    items = [(m, sub_by_value[m]) for m in matches if m in sub_by_value]
                    self._view.show_subcommand_suggestions(items)
                else:
                    initial = cmd.initial() if cmd.initial else None
                    self._view.show_subcommand_suggestions(all_subs, initial=initial)
                return
            from chrys.app.tui.widgets.chrome.file_scanner import fuzzy_filter

            visible = self._visible_slash_commands()
            all_names: list[str] = []
            name_to_cmd: dict[str, SlashCommandDef] = {}
            for cmd in visible:
                for n in [cmd.name, *cmd.aliases]:
                    all_names.append(n)
                    name_to_cmd[n] = cmd
            matches = fuzzy_filter(query, all_names, max_results=20)
            # Preserve fuzzy_filter order (prefix matches before contains matches)
            seen: set[str] = set()
            ordered_cmds: list[SlashCommandDef] = []
            for m in matches:
                cmd = name_to_cmd.get(m)
                if cmd and cmd.name not in seen:
                    seen.add(cmd.name)
                    ordered_cmds.append(cmd)
            disabled = self._disabled_commands()
            items = self._command_suggestion_items(ordered_cmds, disabled)
            skills = self._state.runtime.details.skill_details
            if query:
                skill_names = [skill.name for skill in skills]
                skill_matches = fuzzy_filter(query, skill_names, max_results=20)
                skill_by_name = {skill.name: skill for skill in skills}
                ordered_skills = [skill_by_name[name] for name in skill_matches if name in skill_by_name]
            else:
                ordered_skills = skills
            items.extend(self._runtime_skill_suggestion_items(list(ordered_skills)))
            self._view.update_suggestions(
                items,
                disabled=self._disabled_commands(),
                title=self._suggestion_title("commands"),
            )

        elif self._suggestion_mode == "subcommands":
            cmd = self._active_command
            if not cmd or not cmd.subcommands or not is_slash_command_candidate(text):
                self.dismiss_suggestions()
                return
            parts = text.split(None, 1)
            query = parts[1] if len(parts) > 1 else ""
            from chrys.app.tui.widgets.chrome.file_scanner import fuzzy_filter

            all_subs = cmd.subcommands()
            if query:
                matches = fuzzy_filter(query, [v for v, _ in all_subs])
                sub_by_value = dict(all_subs)
                items = [(m, sub_by_value[m]) for m in matches if m in sub_by_value]
                self._view.update_suggestions(items, title=self._suggestion_title("subcommands"))
            else:
                initial = cmd.initial() if cmd.initial else None
                self._view.show_subcommand_suggestions(all_subs, initial=initial)

        elif self._suggestion_mode == "files":
            idx = max(text.rfind("@"), text.rfind("\uff20"))
            if idx < 0:
                self.dismiss_suggestions()
                return
            query = text[idx + 1 :]
            if query:
                self._enqueue_file_query(query)
            else:
                self._file_latest_query = ""
                self._invalidate_file_query_results()
                snapshot = self._file_snapshot()
                if snapshot is not None:
                    self._view.update_suggestions(
                        self._file_suggestion_items(snapshot.head(limit=_FILE_QUERY_LIMIT), index=snapshot),
                        title=self._suggestion_title("files"),
                    )

        elif self._suggestion_mode == "agents":
            if "#" not in text and "\uff03" not in text:
                self.dismiss_suggestions()
                return
            matches, item_by_name, disabled = self._agent_query_matches(text)
            items = [item_by_name[name] for name in matches]
            self._view.update_suggestions(items, disabled=disabled, title=self._suggestion_title("agents"))

        elif self._suggestion_mode == "models":
            if "$" not in text and "\uff04" not in text:
                self.dismiss_suggestions()
                return
            matches, item_by_id, disabled = self._model_query_matches(text)
            items = [item_by_id[profile_id] for profile_id in matches]
            self._view.update_suggestions(items, disabled=disabled, title=self._suggestion_title("models"))

        elif self._suggestion_mode == "history":
            # The whole input is the search query; entries arriving later read it on load.
            self._enqueue_prompt_history_query(text)

    # -------------------------------------------------------------- #
    # Navigation and selection
    # -------------------------------------------------------------- #

    def on_prompt_history_shortcut_while_open(self) -> None:
        """Ctrl+R again while history is open moves to the next older match, as in shells."""
        self.on_suggestion_navigate("down")

    def on_suggestion_row_width_changed(self) -> None:
        """Re-fit history previews to the new row width."""
        if self._suggestion_mode == "history" and self._prompt_history_index is not None:
            self._enqueue_prompt_history_query(self._view.input_value())

    def on_suggestion_navigate(self, direction: str) -> None:
        if self._defer_prompt_history_key("move", direction):
            return
        self._view.move_suggestion_cursor(direction)

    def on_suggestion_select(self, execute: bool) -> bool:
        """Handle Tab/Enter on suggestion list. Returns True if handled."""
        if self._defer_prompt_history_key("select", "execute" if execute else "insert"):
            return True
        selected = self._view.select_highlighted_suggestion(execute=execute)
        if not selected and execute:
            mode = self._suggestion_mode
            if mode not in {"commands", "subcommands", "agents", "models"}:
                return True

            text = self._view.input_value()
            if mode == "agents" and not self._agent_fallback_should_submit(text):
                return True
            if mode == "models" and not self._model_fallback_should_submit(text):
                return True
            self.dismiss_suggestions()
            self._view.set_input_value("")
            if text and (mode in {"agents", "models"} or not self.dispatch_slash_command(text)):
                self._callbacks.submit_user_text(text)
            return True
        return bool(selected)

    def on_suggestion_selected(self, mode: str, text: str, execute: bool, kind: str = "") -> None:
        """Handle a confirmed suggestion selection."""
        if mode == "commands":
            if kind == "skill":
                self.dismiss_suggestions()
                if execute:
                    self._view.set_input_value("")
                    self._callbacks.submit_user_text(f"/{text}")
                else:
                    self._view.replace_input_trigger_text("/", f"/{text} ")
                return
            cmd = next((c for c in self._slash_commands if c.name == text), None)
            if cmd is None:
                self.dismiss_suggestions()
                return
            if cmd.name in self._disabled_commands():
                return
            if cmd.subcommands:
                self._view.replace_input_trigger_text("/", f"/{text} ")
                self._suggestion_mode = "subcommands"
                self._active_command = cmd
                sub_items = cmd.subcommands()
                initial = cmd.initial() if cmd.initial else None
                self._view.show_subcommand_suggestions(sub_items, initial=initial)
                return
            if execute:
                self.dismiss_suggestions()
                self._view.set_input_value("")
                cmd.action("")
            else:
                self._view.replace_input_trigger_text("/", f"/{text} ")
                self.dismiss_suggestions()

        elif mode == "subcommands":
            cmd = self._active_command
            if not cmd or cmd.name in self._disabled_commands():
                self.dismiss_suggestions()
                return
            # For subcommands that require parameters (e.g., "name"), keep input open
            if text == "name":
                self._view.replace_input_trigger_text(f"/{cmd.name} ", f"/{cmd.name} name ")
                self._suggestion_mode = None
                self._active_command = None
                self.dismiss_suggestions()
            else:
                # Execute immediately for parameterless subcommands
                self.dismiss_suggestions()
                self._view.set_input_value("")
                cmd.action(text)

        elif mode == "files":
            if kind == "status":
                return
            selected_path = text
            if kind == "directory" and not selected_path.endswith(("/", "\\")):
                selected_path = f"{selected_path}/"
            path = f'"{selected_path}"' if " " in selected_path else selected_path
            self._view.replace_input_trigger_text("@", f"@{path} ")
            self.dismiss_suggestions()

        elif mode == "agents":
            self.dismiss_suggestions()
            self._view.set_input_value("")
            self._callbacks.start_agent_profile_switch(text)

        elif mode == "models":
            self.dismiss_suggestions()
            self._view.set_input_value("")
            self._callbacks.start_model_profile_switch(text)

        elif mode == "history":
            self.dismiss_suggestions()
            self._view.set_input_value(text)

    @property
    def suggestion_mode(self) -> str | None:
        return self._suggestion_mode

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pure protocol logic for verifier-guided multi-turn ARC-AGI episodes."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any


MAX_GRID_DIM = 30
Grid = list[list[int]]

COLOR_MAPPING = """Each cell is an integer naming a color:
0 = black, 1 = blue, 2 = red, 3 = green, 4 = yellow,
5 = gray, 6 = magenta, 7 = orange, 8 = azure, 9 = maroon.
When naming colors, use this mapping exactly."""

_FORBIDDEN_MODEL_REQUEST_KEYS = frozenset(
    {
        "expected",
        "expected_output",
        "expected_outputs",
        "hidden_output",
        "hidden_outputs",
        "target",
        "targets",
        "test_output",
        "test_outputs",
        "test_targets",
    }
)


class PredictionParseError(ValueError):
    """Raised when an executor response violates the strict grid contract."""


class TransformDescriptionParseError(ValueError):
    """Raised when a proposer response violates its single-block contract."""


class EpisodePhase(str, Enum):
    """The next action expected by the ARC refinement state machine."""

    PROPOSER = "proposer"
    EXECUTOR_TRAIN = "executor_train"
    EXECUTOR_TEST = "executor_test"
    TERMINATED = "terminated"


class TerminationReason(str, Enum):
    """Terminal outcomes across both episode protocols."""

    TRAIN_VERIFIED = "train_verified"
    ALL_SOLVED = "all_solved"
    CANDIDATE_SELECTED = "candidate_selected"
    CONTEXT_EXHAUSTED = "context_exhausted"
    EXECUTOR_FORMAT_FAILURE = "executor_format_failure"
    EXECUTOR_CONTEXT_EXHAUSTED = "executor_context_exhausted"
    AGENT_ERROR = "agent_error"
    EMERGENCY_ROUND_CAP = "emergency_round_cap"


@dataclass(frozen=True)
class GridVerification:
    """Deterministic comparison of one predicted grid with its target."""

    grid_id: str
    predicted: Grid
    correct: Grid
    exact: bool
    shape_match: bool
    cell_accuracy: float
    feedback: str


@dataclass(frozen=True)
class ContextBudget:
    """Inputs to the proactive proposer context-window check."""

    model_context_limit: int
    reserved_proposer_output_tokens: int
    chat_template_margin: int

    def __post_init__(self) -> None:
        if self.model_context_limit <= 0:
            raise ValueError("model_context_limit must be positive")
        if self.reserved_proposer_output_tokens <= 0:
            raise ValueError("reserved_proposer_output_tokens must be positive")
        if self.chat_template_margin < 0:
            raise ValueError("chat_template_margin must not be negative")
        if self.reserved_proposer_output_tokens + self.chat_template_margin >= self.model_context_limit:
            raise ValueError("reserved proposer output plus template margin must fit within the context limit")

    def permits_revision(self, *, current_proposer_tokens: int, next_feedback_tokens: int) -> bool:
        """Return whether another complete proposer turn fits the configured limit."""
        if current_proposer_tokens < 0 or next_feedback_tokens < 0:
            raise ValueError("token counts must not be negative")
        return (
            current_proposer_tokens
            + next_feedback_tokens
            + self.reserved_proposer_output_tokens
            + self.chat_template_margin
            <= self.model_context_limit
        )


@dataclass(frozen=True)
class EpisodeState:
    """Small explicit state machine used by the agent and unit tests."""

    phase: EpisodePhase
    round_index: int
    format_retry_used: bool
    termination_reason: TerminationReason | None

    @classmethod
    def initial(cls) -> EpisodeState:
        return cls(
            phase=EpisodePhase.PROPOSER,
            round_index=0,
            format_retry_used=False,
            termination_reason=None,
        )

    def description_generated(self) -> EpisodeState:
        self._require_phase(EpisodePhase.PROPOSER)
        return replace(self, phase=EpisodePhase.EXECUTOR_TRAIN, format_retry_used=False)

    def executor_format_failed(self) -> EpisodeState:
        if self.phase not in (EpisodePhase.EXECUTOR_TRAIN, EpisodePhase.EXECUTOR_TEST):
            raise RuntimeError(f"unexpected executor format failure in phase {self.phase.value}")
        if self.format_retry_used:
            return self.terminate(TerminationReason.EXECUTOR_FORMAT_FAILURE)
        return replace(self, format_retry_used=True)

    def demos_verified(self) -> EpisodeState:
        """Enter the hidden-test phase after every demo grid verified exactly."""
        self._require_phase(EpisodePhase.EXECUTOR_TRAIN)
        return replace(self, phase=EpisodePhase.EXECUTOR_TEST, format_retry_used=False)

    def demo_loop_exhausted(self) -> EpisodeState:
        """Enter the hidden-test phase with the current rule after a budget guard.

        The deployment-shaped protocol always answers the hidden test with the
        best rule it has, even when the demo-refinement loop ends early.
        """
        self._require_phase(EpisodePhase.PROPOSER)
        return replace(self, phase=EpisodePhase.EXECUTOR_TEST, format_retry_used=False)

    def eval_grid_solved(self, *, all_solved: bool) -> EpisodeState:
        """Advance the evaluation-grid sequence after an exact solve."""
        self._require_phase(EpisodePhase.EXECUTOR_TRAIN)
        if all_solved:
            return self.terminate(TerminationReason.ALL_SOLVED)
        return replace(self, format_retry_used=False)

    def eval_grid_failed(self) -> EpisodeState:
        """Return to the proposer for a revision of the current rule."""
        self._require_phase(EpisodePhase.EXECUTOR_TRAIN)
        return replace(
            self,
            phase=EpisodePhase.PROPOSER,
            round_index=self.round_index + 1,
            format_retry_used=False,
        )

    def terminate(self, reason: TerminationReason) -> EpisodeState:
        if self.phase is EpisodePhase.TERMINATED:
            raise RuntimeError("episode is already terminated")
        return replace(
            self,
            phase=EpisodePhase.TERMINATED,
            termination_reason=reason,
        )

    def _require_phase(self, expected: EpisodePhase) -> None:
        if self.phase is not expected:
            raise RuntimeError(f"expected episode phase {expected.value}, got {self.phase.value}")


def validate_grid(value: Any, *, grid_id: str) -> Grid:
    """Validate and copy a non-empty rectangular ARC grid."""
    if not isinstance(value, list) or not value:
        raise PredictionParseError(f"{grid_id} must be a non-empty list of rows")
    if len(value) > MAX_GRID_DIM:
        raise PredictionParseError(f"{grid_id} has more than {MAX_GRID_DIM} rows")

    width: int | None = None
    grid: Grid = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or not row:
            raise PredictionParseError(f"{grid_id} row {row_index} must be a non-empty list")
        if width is None:
            width = len(row)
            if width > MAX_GRID_DIM:
                raise PredictionParseError(f"{grid_id} has more than {MAX_GRID_DIM} columns")
        elif len(row) != width:
            raise PredictionParseError(f"{grid_id} is ragged at row {row_index}")

        clean_row: list[int] = []
        for column_index, cell in enumerate(row):
            if type(cell) is not int or not 0 <= cell <= 9:
                raise PredictionParseError(
                    f"{grid_id}[{row_index}][{column_index}] must be an integer from 0 through 9"
                )
            clean_row.append(cell)
        grid.append(clean_row)
    return grid


# The canonical proposer<->executor rule schema, in render order. Matches the
# NVARC ingestion schema used for executor training rows, so a co-trained
# policy proposes rules in exactly the format it learned to execute.
CANONICAL_SECTIONS = (
    "rules_summary",
    "solution_steps",
    "key_insight",
    "puzzle_concepts",
)

_CANONICAL_SECTION_RES = {name: re.compile(rf"<{name}>\s*(.*?)\s*</{name}>", re.DOTALL) for name in CANONICAL_SECTIONS}


def parse_canonical_rule(text: str) -> str:
    """Extract and re-render the proposer's canonical 4-section rule.

    Every canonical section must be present and non-empty; when a tag repeats,
    the last occurrence wins (the same final-answer convention the answer-grid
    parser uses, so scratchpad text cannot displace the real rule). The
    sections are re-rendered in canonical order, which is byte-identical to
    the executor-training rule rendering.
    """
    sections: dict[str, str] = {}
    for name in CANONICAL_SECTIONS:
        matches = _CANONICAL_SECTION_RES[name].findall(text)
        if not matches or not matches[-1].strip():
            raise TransformDescriptionParseError(f"response must contain one non-empty <{name}> section")
        sections[name] = matches[-1].strip()
    rendered = "\n\n".join(f"<{name}>\n{sections[name]}\n</{name}>" for name in CANONICAL_SECTIONS)
    if re.search(r"```\s*python|\bdef\s+\w+\s*\(", rendered, flags=re.IGNORECASE):
        raise TransformDescriptionParseError("rule sections must not contain Python")
    return rendered


def compare_grid(*, grid_id: str, predicted: Grid, correct: Grid) -> GridVerification:
    """Compare one grid and render deterministic shape or predicted-correct feedback."""
    predicted_shape = (len(predicted), len(predicted[0]))
    correct_shape = (len(correct), len(correct[0]))
    if predicted_shape != correct_shape:
        feedback = (
            f"Example {grid_id}: shape mismatch; predicted {predicted_shape[0]}x{predicted_shape[1]}, "
            f"correct {correct_shape[0]}x{correct_shape[1]}."
        )
        return GridVerification(
            grid_id=grid_id,
            predicted=predicted,
            correct=correct,
            exact=False,
            shape_match=False,
            cell_accuracy=0.0,
            feedback=feedback,
        )

    matching = sum(
        predicted[row][column] == correct[row][column]
        for row in range(correct_shape[0])
        for column in range(correct_shape[1])
    )
    total = correct_shape[0] * correct_shape[1]
    exact = matching == total
    if exact:
        feedback = f"Example {grid_id}: exact."
    else:
        rows = [
            " ".join(f"{predicted[row][column]}-{correct[row][column]}" for column in range(correct_shape[1]))
            for row in range(correct_shape[0])
        ]
        feedback = f"Example {grid_id}: mismatch\nDiff (predicted-correct):\n" + "\n".join(rows)
    return GridVerification(
        grid_id=grid_id,
        predicted=predicted,
        correct=correct,
        exact=exact,
        shape_match=True,
        cell_accuracy=matching / total,
        feedback=feedback,
    )


def conservative_text_token_bound(text: str) -> int:
    """Return a tokenizer-independent upper bound for byte-fallback tokenizers."""
    return len(text.encode("utf-8"))


def assert_model_request_safe(payload: Any) -> None:
    """Reject verifier-only field names anywhere in a request sent to a model."""
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()
    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized_key = str(key).lower()
            if normalized_key in _FORBIDDEN_MODEL_REQUEST_KEYS:
                raise ValueError(f"model request contains verifier-only field {key!r}")
            assert_model_request_safe(value)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            assert_model_request_safe(value)


def format_grid(grid: Grid) -> str:
    """Render a grid with explicit cell boundaries."""
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


# The single-grid executor contract. This text must stay byte-identical to
# NeMo-RL's examples/prompts/nvarc_executor.txt (rendered there with
# task_data_spec.prompt.format(task_body), no system message), so a policy
# trained on native executor rows sees exactly the same task through this
# server. The trailing newline is part of the contract.
NVARC_EXECUTOR_PROMPT_TEMPLATE = """You are an exact grid-transformation executor. Apply the supplied transformation
mechanically to the input grid. Do not infer, alter, critique, or explain the
transformation.

Reason only as much as needed to execute the supplied rule. Stop reasoning as
soon as the output grid is determined.

Each grid is written one row per line with cells separated by single spaces.
Cells are integer colors: 0=black, 1=blue, 2=red, 3=green, 4=yellow, 5=gray,
6=magenta, 7=orange, 8=azure, 9=maroon.

{}

Return exactly one answer block in this form:

<answer>
0 1 2
3 4 5
</answer>

The block must contain only the transformed grid. Do not use JSON, prose, or a
Markdown code fence. Preserve the exact output shape implied by the operation.
"""


def build_single_executor_prompt(*, description: str, input_grid: Grid) -> str:
    """Render one rule plus one grid as the shared single-grid executor task."""
    task_body = f"<transformation>\n{description}\n</transformation>\n<input>\n{format_grid(input_grid)}\n</input>"
    return NVARC_EXECUTOR_PROMPT_TEMPLATE.format(task_body)


def build_single_grid_format_retry_prompt() -> str:
    """Request only a corrected rendering of the same single-grid answer.

    Wording matches NeMo-RL's executor benchmark retry so the one format-only
    retry is the same event in training, benchmarking, and episodes.
    """
    return (
        "Your previous response did not contain a parseable grid inside an "
        "<answer> block. Do not change or reinterpret the transformation. Return "
        "exactly one grid using this form:\n\n"
        "<answer>\n"
        "0 1 2\n"
        "3 4 5\n"
        "</answer>\n\n"
        "Do not include JSON, prose, or a Markdown code fence."
    )


def build_nvarc_proposer_prompt(*, demo_pairs: list[dict[str, Grid]]) -> str:
    """Build the persistent proposer prompt for the evaluation-grid sequence.

    Shows only the demonstration pairs -- the held-out evaluation inputs are
    not revealed up front, so the rule must generalize rather than fit them.
    """
    sections = [
        "Infer one general transformation rule from the paired ARC training examples.",
        COLOR_MAPPING,
    ]
    for index, pair in enumerate(demo_pairs):
        sections.append(
            f"Training example demo_{index}\nInput:\n{format_grid(pair['input'])}\nOutput:\n{format_grid(pair['output'])}"
        )
    sections.append(
        "Reason carefully but concisely. Do not restate complete grids in prose. Stop once one rule accounts "
        "for every example. First briefly describe the important objects, colors, shapes, symmetries, and "
        "spatial relationships; "
        "compare inputs and outputs to identify invariants and changes; then infer a rule that an independent "
        "executor can apply to a new input grid. Generalize beyond the displayed grids. Do not emit Python or a "
        "predicted grid. Return the rule as exactly these four sections:\n"
        "<rules_summary>\n...\n</rules_summary>\n"
        "<solution_steps>\n...\n</solution_steps>\n"
        "<key_insight>\n...\n</key_insight>\n"
        "<puzzle_concepts>\n...\n</puzzle_concepts>"
    )
    return "\n\n".join(sections)


def build_eval_feedback_prompt(
    *,
    grid_id: str,
    input_grid: Grid,
    predicted: Grid | None,
    expected: Grid,
    diff_feedback: str,
) -> str:
    """Render the behavioral evidence for one failed evaluation grid.

    The proposer receives only the input, the executor's output, the expected
    output, and the deterministic diff -- never executor reasoning.
    """
    predicted_text = format_grid(predicted) if predicted is not None else "(no parseable grid)"
    return (
        f"An independent executor applied your rule to a held-out grid {grid_id}. It failed.\n\n"
        f"Input:\n{format_grid(input_grid)}\n\n"
        f"Executor output:\n{predicted_text}\n\n"
        f"Expected output:\n{format_grid(expected)}\n\n"
        f"{diff_feedback}\n\n"
        "Return a complete replacement rule, not a patch and not a grid. Return exactly the four sections "
        "<rules_summary>, <solution_steps>, <key_insight>, and <puzzle_concepts>."
    )

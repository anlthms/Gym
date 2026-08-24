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

import json
import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any


MAX_GRID_DIM = 30
Grid = list[list[int]]

COLOR_MAPPING = """Each cell is an integer naming a color:
0 = black, 1 = blue, 2 = red, 3 = green, 4 = yellow,
5 = gray, 6 = fuchsia, 7 = orange, 8 = teal, 9 = brown."""

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
    """Terminal outcomes emitted by the first implementation track."""

    TRAIN_VERIFIED = "train_verified"
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
class BatchVerification:
    """Aggregate verification for all requested executor grids."""

    records: tuple[GridVerification, ...]
    all_exact: bool
    exact_fraction: float
    shape_match_fraction: float
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
        self._require_phase(EpisodePhase.EXECUTOR_TRAIN)
        if self.format_retry_used:
            return self.terminate(TerminationReason.EXECUTOR_FORMAT_FAILURE)
        return replace(self, format_retry_used=True)

    def training_verified(self, *, all_exact: bool) -> EpisodeState:
        self._require_phase(EpisodePhase.EXECUTOR_TRAIN)
        if all_exact:
            return replace(self, phase=EpisodePhase.EXECUTOR_TEST, format_retry_used=False)
        return replace(
            self,
            phase=EpisodePhase.PROPOSER,
            round_index=self.round_index + 1,
            format_retry_used=False,
        )

    def test_answered(self) -> EpisodeState:
        self._require_phase(EpisodePhase.EXECUTOR_TEST)
        return self.terminate(TerminationReason.TRAIN_VERIFIED)

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


def parse_tagged_grids(text: str, *, tag: str, expected_ids: list[str]) -> dict[str, Grid]:
    """Parse one strict tagged JSON object whose keys exactly match ``expected_ids``."""
    if not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("expected_ids must be non-empty and unique")
    pattern = rf"\s*<{re.escape(tag)}>\s*(\{{.*\}})\s*</{re.escape(tag)}>\s*"
    match = re.fullmatch(pattern, text, flags=re.DOTALL)
    if match is None:
        raise PredictionParseError(f"response must contain only one <{tag}> JSON object")
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as error:
        raise PredictionParseError(f"<{tag}> contains invalid JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise PredictionParseError(f"<{tag}> must contain a JSON object")

    actual_ids = set(payload)
    required_ids = set(expected_ids)
    if actual_ids != required_ids:
        missing = sorted(required_ids - actual_ids)
        extra = sorted(actual_ids - required_ids)
        raise PredictionParseError(f"<{tag}> keys do not match; missing={missing}, extra={extra}")
    return {grid_id: validate_grid(payload[grid_id], grid_id=grid_id) for grid_id in expected_ids}


def parse_transform_description(text: str) -> str:
    """Extract the proposer's single explicit transformation-description artifact."""
    match = re.fullmatch(
        r"\s*<transform_description>\s*(.*?)\s*</transform_description>\s*",
        text,
        flags=re.DOTALL,
    )
    if match is None or not match.group(1).strip():
        raise TransformDescriptionParseError("response must contain only one non-empty <transform_description> block")
    description = match.group(1).strip()
    if re.search(r"```\s*python|\bdef\s+\w+\s*\(", description, flags=re.IGNORECASE):
        raise TransformDescriptionParseError("transform description must not contain Python")
    return description


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


def verify_predictions(*, predictions: dict[str, Grid], correct: dict[str, Grid]) -> BatchVerification:
    """Verify an exactly-keyed prediction mapping in target insertion order."""
    if set(predictions) != set(correct):
        raise ValueError("prediction and target keys must match exactly before verification")
    records = tuple(
        compare_grid(grid_id=grid_id, predicted=predictions[grid_id], correct=target)
        for grid_id, target in correct.items()
    )
    count = len(records)
    if count == 0:
        raise ValueError("at least one grid is required for verification")
    return BatchVerification(
        records=records,
        all_exact=all(record.exact for record in records),
        exact_fraction=sum(record.exact for record in records) / count,
        shape_match_fraction=sum(record.shape_match for record in records) / count,
        cell_accuracy=sum(record.cell_accuracy for record in records) / count,
        feedback="\n\n".join(record.feedback for record in records),
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


def build_proposer_prompt(
    *,
    train_pairs: list[dict[str, Grid]],
    test_inputs: list[Grid],
) -> str:
    """Build the initial persistent proposer prompt without hidden test outputs."""
    sections = [
        "Infer one general transformation rule from the paired ARC training examples.",
        COLOR_MAPPING,
    ]
    for index, pair in enumerate(train_pairs):
        sections.append(
            f"Training example train_{index}\nInput:\n{format_grid(pair['input'])}\nOutput:\n{format_grid(pair['output'])}"
        )
    for index, grid in enumerate(test_inputs):
        sections.append(f"Test input test_{index} (do not execute it yet):\n{format_grid(grid)}")
    sections.append(
        "Describe an operational rule that identifies selected cells or objects, the ordered operations, output shape, "
        "colors and positions, and any stopping conditions. Generalize beyond the displayed grids. Do not emit Python "
        "or a predicted test grid. Return exactly:\n"
        "<transform_description>\ncomplete replacement description\n</transform_description>"
    )
    return "\n\n".join(sections)


def build_executor_prompt(*, description: str, inputs: dict[str, Grid], tag: str) -> str:
    """Build a fresh executor request that contains inputs but no expected outputs."""
    sections = [
        "Apply the supplied transformation exactly. Do not infer or revise a different rule.",
        COLOR_MAPPING,
        f"Transformation:\n<transform_description>\n{description}\n</transform_description>",
    ]
    for grid_id, grid in inputs.items():
        sections.append(f"Input {grid_id}:\n{format_grid(grid)}")
    example_items = ", ".join(f'"{grid_id}": [[0]]' for grid_id in inputs)
    sections.append(
        f"Return grids only, with no prose, using valid JSON and exactly these keys:\n"
        f"<{tag}>\n{{{example_items}}}\n</{tag}>"
    )
    return "\n\n".join(sections)


def build_revision_prompt(feedback: str) -> str:
    """Ask for a complete replacement rule after deterministic verification."""
    return (
        "The executor applied your description to the training inputs. Deterministic verification found:\n\n"
        f"{feedback}\n\n"
        "Return a complete replacement description, not a patch and not a test answer. Return exactly one "
        "<transform_description> block."
    )


def build_test_followup_prompt(*, test_inputs: dict[str, Grid]) -> str:
    """Request hidden-test execution only after the training gate passes."""
    sections = [
        "Every training prediction passed exact verification. Apply the same supplied transformation to these test "
        "inputs. Do not infer or revise a different rule."
    ]
    for grid_id, grid in test_inputs.items():
        sections.append(f"Input {grid_id}:\n{format_grid(grid)}")
    example_items = ", ".join(f'"{grid_id}": [[0]]' for grid_id in test_inputs)
    sections.append(
        "Return grids only, with no prose, using valid JSON and exactly these keys:\n"
        f"<answers>\n{{{example_items}}}\n</answers>"
    )
    return "\n\n".join(sections)


def build_format_retry_prompt(*, tag: str, expected_ids: list[str], error: str) -> str:
    """Request one format-only executor retry without changing the rule."""
    return (
        f"Your previous response could not be parsed: {error}\n"
        "Do not change or reinterpret the transformation. Reformat the same predictions as valid JSON with exactly "
        f"the keys {expected_ids}, inside one <{tag}>...</{tag}> block, and emit no other text."
    )

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
"""Answer-grid extraction and gain-over-echo reward scoring for ARC grids.

A behavior-faithful port of NeMo-RL's ``nemo_rl/environments/arc_agi_grid.py``
scorer, so a policy trained on single-turn executor rows and one evaluated
through this server face the same contract: the response must carry an
``<answer>`` block of digit rows, dense similarity terms are paid as *gain
over echoing the input* (an echo earns exactly zero and a non-answer echo is
pinned to the unparseable floor), and exact match dominates all shaping. Keep
the two implementations in sync; the tests here lock the shared behavior.

Pure Python on purpose (the source uses numpy only for the alignment search;
grids are at most 30x30, so the sliding search is cheap in a verifier).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# An ARC grid: rectangular, symbols 0-9, at most 30x30.
Grid = list[list[int]]

ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"
MAX_GRID_DIM = 30

# Row boundary marker for edit distance. Outside 0-9 so it can never match a cell.
_ROW_SENTINEL = -1

_ANSWER_BLOCK_RE = re.compile(re.escape(ANSWER_OPEN) + r"(.*?)" + re.escape(ANSWER_CLOSE), re.DOTALL)


@dataclass(frozen=True)
class RewardWeights:
    """Per-term reward weights.

    No field defaults on purpose -- the defaults live on the resources-server
    config, and a second set here would be a second source of truth.
    """

    exact: float
    cell: float
    edit: float
    color: float
    extraneous: float
    shape: float
    format: float


def serialize_grid(grid: Grid) -> str:
    """Render a grid as one line per row, cells separated by single spaces.

    The separator is not cosmetic: without it a row is a single run of digits
    that the tokenizer merges into arbitrary multi-cell chunks, so cell
    boundaries are invisible to the model.
    """
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def _parse_row(row: str) -> list[int] | None:
    """Parse one serialized row, accepting spaced or contiguous digits."""
    if any(char.isspace() for char in row):
        cells = row.split()
        if not all(len(cell) == 1 and cell.isdigit() for cell in cells):
            return None
        return [int(cell) for cell in cells]
    if not row.isdigit():
        return None
    return [int(char) for char in row]


def parse_answer_grid(text: str) -> Grid | None:
    """Parse digit-rows into a grid, or return None if it is not well formed.

    Rejects empty grids, ragged rows, non-digit characters, and anything larger
    than ``MAX_GRID_DIM`` in either dimension. Strict on purpose: an
    over-permissive parser inflates reward.
    """
    lines = [line.strip() for line in text.strip().splitlines()]
    lines = [line for line in lines if line]
    if not lines or len(lines) > MAX_GRID_DIM:
        return None
    grid: Grid = []
    for line in lines:
        row = _parse_row(line)
        if row is None or not row or len(row) > MAX_GRID_DIM:
            return None
        if grid and len(row) != len(grid[0]):
            return None
        grid.append(row)
    return grid


def extract_answer_grid(response: str) -> Grid | None:
    """Extract the final answer grid from a model response.

    Scans every ``<answer>...</answer>`` block and returns the last one that
    parses; falls back to the text after a final unclosed ``<answer>`` because
    generation stops on the closing delimiter and a token-capped response has
    no closing tag either.
    """
    for block in reversed(_ANSWER_BLOCK_RE.findall(response)):
        grid = parse_answer_grid(block)
        if grid is not None:
            return grid

    _, delimiter, tail = response.rpartition(ANSWER_OPEN)
    if delimiter and ANSWER_CLOSE not in tail:
        return parse_answer_grid(tail)
    return None


def grid_shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def overlay_cell_accuracy(pred: Grid, target: Grid) -> float:
    """Fraction of cells that agree when ``pred`` is centered over ``target``.

    The denominator is the larger of the two areas, so padding the prediction
    out to a huge grid that happens to cover the target cannot inflate the
    score.
    """
    pred_h, pred_w = grid_shape(pred)
    target_h, target_w = grid_shape(target)
    row_offset = (target_h - pred_h) // 2
    col_offset = (target_w - pred_w) // 2

    matches = 0
    for row in range(target_h):
        pred_row = row - row_offset
        if not 0 <= pred_row < pred_h:
            continue
        for col in range(target_w):
            pred_col = col - col_offset
            if 0 <= pred_col < pred_w and pred[pred_row][pred_col] == target[row][col]:
                matches += 1

    return matches / max(pred_h * pred_w, target_h * target_w)


def best_alignment_cell_accuracy(pred: Grid, target: Grid) -> float:
    """Cell agreement at the best valid-mode placement of one grid in the other.

    Slide the smaller grid entirely inside the larger and count agreeing cells
    at each placement, so an odd-sized near-miss is not decided by the centered
    overlay's coin-flip alignment. The denominator stays ``max`` of the two
    areas. Falls back to the centered overlay when neither grid fits inside
    the other (one taller, the other wider), where valid mode has no placement.
    """
    pred_h, pred_w = grid_shape(pred)
    target_h, target_w = grid_shape(target)

    if pred_h <= target_h and pred_w <= target_w:
        small, big = pred, target
    elif target_h <= pred_h and target_w <= pred_w:
        small, big = target, pred
    else:
        return overlay_cell_accuracy(pred, target)

    small_h, small_w = grid_shape(small)
    big_h, big_w = grid_shape(big)
    denominator = max(pred_h * pred_w, target_h * target_w)
    best = 0
    for row in range(big_h - small_h + 1):
        for col in range(big_w - small_w + 1):
            matches = sum(small[i][j] == big[row + i][col + j] for i in range(small_h) for j in range(small_w))
            best = max(best, matches)
    return best / denominator


def color_recall(pred: Grid, target: Grid) -> float:
    """Fraction of the target's colors that appear anywhere in the prediction."""
    target_colors = {cell for row in target for cell in row}
    pred_colors = {cell for row in pred for cell in row}
    return len(target_colors & pred_colors) / len(target_colors)


def extraneous_color_fraction(pred: Grid, target: Grid) -> float:
    """Fraction of the prediction's colors that the target does not use.

    Paired with ``color_recall``: without this penalty, emitting all ten colors
    would max out recall for free.
    """
    target_colors = {cell for row in target for cell in row}
    pred_colors = {cell for row in pred for cell in row}
    return len(pred_colors - target_colors) / len(pred_colors)


def _flatten(grid: Grid) -> list[int]:
    """Flatten a grid to a cell sequence with a sentinel between rows.

    The row sentinel is what makes edit distance shape-aware: without it,
    dropping a row would look like a handful of substitutions instead of a
    deleted row.
    """
    sequence: list[int] = []
    for index, row in enumerate(grid):
        if index:
            sequence.append(_ROW_SENTINEL)
        sequence.extend(row)
    return sequence


def _levenshtein(left: list[int], right: list[int]) -> int:
    """Edit distance between two cell sequences, two-row DP."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_cell in enumerate(left, start=1):
        current = [i]
        for j, right_cell in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_cell != right_cell),
                )
            )
        previous = current
    return previous[-1]


def edit_similarity(pred: Grid, target: Grid) -> float:
    """1 - normalized edit distance between the two grids, in [0, 1].

    Complements ``best_alignment_cell_accuracy``: cell accuracy scores a
    prediction that is right but shifted by one row as almost entirely wrong,
    while edit distance charges it for one insertion.
    """
    left, right = _flatten(pred), _flatten(target)
    return 1.0 - _levenshtein(left, right) / max(len(left), len(right))


def gain_over_baseline(score: float, baseline: float) -> float:
    """Rescale an absolute score to its improvement over a baseline, into [-1, 1].

    This is what stops copying the input from paying: an echo scores ~0.61
    cell accuracy on the ARC-AGI-2 evaluation split because backgrounds
    usually survive the transformation, but measured against a copy of the
    same task's input it is worth exactly zero. Both directions are normalized
    by the room available in that direction.
    """
    if score >= baseline:
        headroom = 1.0 - baseline
        return 1.0 if headroom <= 0.0 else (score - baseline) / headroom
    return (score - baseline) / baseline if baseline > 0.0 else -1.0


def shape_mismatch(pred: Grid, target: Grid) -> float:
    """Normalized magnitude of the shape error, in [0, 1]."""
    pred_h, pred_w = grid_shape(pred)
    target_h, target_w = grid_shape(target)
    error = abs(target_h - pred_h) + abs(target_w - pred_w)
    return min(1.0, error / (target_h + target_w))


def reward_floor(weights: RewardWeights) -> float:
    """The worst reward any response can earn.

    Two responses sit here: one that could not be parsed, and one that echoed
    the input without solving it. Both must stay strictly below the worst
    *genuine* parseable answer, which bottoms out one format bonus above --
    without that gap the format term cannot bootstrap.
    """
    return -(weights.cell + weights.edit + weights.extraneous + weights.shape)


def score_grid(pred: Grid | None, target: Grid, test_input: Grid, weights: RewardWeights) -> dict[str, float]:
    """Score one extracted grid (or a parse failure) against the target.

    The two similarity terms are paid on the *gain over echoing ``test_input``*,
    not on their absolute value. Returns the total plus every term separately;
    the breakdown is the primary diagnostic for whether reward growth is real
    (exact match) or shaping.
    """
    if pred is None:
        return {
            "reward": reward_floor(weights),
            "grid_match": 0.0,
            "cell_match": 0.0,
            "cell_gain": -1.0,
            "edit_similarity": 0.0,
            "edit_gain": -1.0,
            "color_recall": 0.0,
            "extraneous_colors": 1.0,
            "shape_mismatch": 1.0,
            "format_valid": 0.0,
            "copied_input": 0.0,
        }

    copy_cell = best_alignment_cell_accuracy(test_input, target)
    copy_edit = edit_similarity(test_input, target)

    terms = {
        "grid_match": float(pred == target),
        "cell_match": best_alignment_cell_accuracy(pred, target),
        "edit_similarity": edit_similarity(pred, target),
        "color_recall": color_recall(pred, target),
        "extraneous_colors": extraneous_color_fraction(pred, target),
        "shape_mismatch": shape_mismatch(pred, target),
        "format_valid": 1.0,
        # Not scored. Logged because "is the policy converging on an echo" is
        # the question the gain-over-echo scoring exists to answer.
        "copied_input": float(pred == test_input),
    }
    terms["cell_gain"] = gain_over_baseline(terms["cell_match"], copy_cell)
    terms["edit_gain"] = gain_over_baseline(terms["edit_similarity"], copy_edit)

    if terms["copied_input"] and not terms["grid_match"]:
        # An inexact echo still earns color and format credit after its
        # similarity gain is zeroed, so put it on the non-answer floor.
        terms["reward"] = reward_floor(weights)
    else:
        terms["reward"] = (
            weights.exact * terms["grid_match"]
            + weights.cell * terms["cell_gain"]
            + weights.edit * terms["edit_gain"]
            + weights.color * terms["color_recall"]
            - weights.extraneous * terms["extraneous_colors"]
            - weights.shape * terms["shape_mismatch"]
            + weights.format * terms["format_valid"]
        )
    return terms


def score_answer_response(response: str, target: Grid, test_input: Grid, weights: RewardWeights) -> dict[str, float]:
    """Extract the ``<answer>`` grid from a response and score it."""
    return score_grid(extract_answer_grid(response), target, test_input, weights)

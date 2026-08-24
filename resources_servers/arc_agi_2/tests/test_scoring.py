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
"""Lock the gain-over-echo scorer to NeMo-RL's arc_agi_grid behavior."""

import pytest

from resources_servers.arc_agi_2.scoring import (
    RewardWeights,
    best_alignment_cell_accuracy,
    extract_answer_grid,
    gain_over_baseline,
    reward_floor,
    score_answer_response,
    score_grid,
    serialize_grid,
)


WEIGHTS = RewardWeights(exact=1.0, cell=0.20, edit=0.10, color=0.05, extraneous=0.05, shape=0.05, format=0.05)


def _answer(grid: list[list[int]]) -> str:
    return f"<answer>\n{serialize_grid(grid)}\n</answer>"


def test_extract_answer_grid_takes_last_parsing_block() -> None:
    response = "reasoning <answer>\nnot a grid\n</answer> more <answer>\n1 2\n3 4\n</answer>"
    assert extract_answer_grid(response) == [[1, 2], [3, 4]]


def test_extract_answer_grid_reads_unclosed_final_block() -> None:
    assert extract_answer_grid("thinking...\n<answer>\n5 6\n7 8") == [[5, 6], [7, 8]]


def test_extract_answer_grid_accepts_contiguous_digits() -> None:
    assert extract_answer_grid("<answer>\n12\n34\n</answer>") == [[1, 2], [3, 4]]


@pytest.mark.parametrize(
    "text",
    [
        "no answer block at all",
        "<answer>\n1 2\n3\n</answer>",  # ragged
        "<answer>\n1 a\n</answer>",  # non-digit
        "<answer>\n</answer>",  # empty
    ],
)
def test_extract_answer_grid_rejects_malformed(text: str) -> None:
    assert extract_answer_grid(text) is None


def test_exact_answer_earns_dominant_reward() -> None:
    target = [[1, 2], [3, 4]]
    terms = score_answer_response(_answer(target), target, [[0, 0], [0, 0]], WEIGHTS)
    assert terms["grid_match"] == 1.0
    assert terms["reward"] > sum((WEIGHTS.cell, WEIGHTS.edit, WEIGHTS.color, WEIGHTS.format))


def test_unparseable_response_sits_on_the_floor() -> None:
    terms = score_answer_response("gibberish", [[1]], [[2]], WEIGHTS)
    assert terms["reward"] == reward_floor(WEIGHTS)
    assert terms["format_valid"] == 0.0


def test_inexact_echo_is_pinned_to_the_floor() -> None:
    test_input = [[1, 1], [1, 0]]
    target = [[1, 1], [1, 2]]  # mostly the same as the input
    terms = score_answer_response(_answer(test_input), target, test_input, WEIGHTS)
    assert terms["copied_input"] == 1.0
    assert terms["reward"] == reward_floor(WEIGHTS)


def test_echo_gain_is_zero_by_construction() -> None:
    assert gain_over_baseline(0.61, 0.61) == 0.0
    assert gain_over_baseline(1.0, 0.61) == 1.0
    assert gain_over_baseline(0.0, 0.61) < 0.0


def test_worst_parseable_answer_beats_the_floor() -> None:
    # A completely wrong but parseable answer must stay above the floor by
    # the format bonus, or the format term cannot bootstrap.
    target = [[1, 1], [1, 1]]
    terms = score_answer_response(_answer([[0]]), target, [[2, 2], [2, 2]], WEIGHTS)
    assert terms["reward"] > reward_floor(WEIGHTS)


def test_best_alignment_slides_to_the_matching_placement() -> None:
    target = [[0, 0, 0], [0, 5, 0], [0, 0, 0]]
    pred = [[5]]
    assert best_alignment_cell_accuracy(pred, target) == 1 / 9


def test_score_grid_handles_none_prediction() -> None:
    terms = score_grid(None, [[1]], [[2]], WEIGHTS)
    assert terms["reward"] == reward_floor(WEIGHTS)
    assert terms["cell_gain"] == -1.0

import pytest

from resources_servers.arc_agi_2.logic import (
    ContextBudget,
    EpisodePhase,
    EpisodeState,
    PredictionParseError,
    TerminationReason,
    TransformDescriptionParseError,
    assert_model_request_safe,
    build_eval_feedback_prompt,
    build_executor_prompt,
    build_nvarc_proposer_prompt,
    build_proposer_prompt,
    build_single_executor_prompt,
    compare_grid,
    conservative_text_token_bound,
    parse_canonical_rule,
    parse_tagged_grids,
    parse_transform_description,
    verify_predictions,
)


def test_parse_tagged_grids_accepts_exact_contract() -> None:
    parsed = parse_tagged_grids(
        '<predictions>{"train_0": [[0, 2], [2, 0]]}</predictions>',
        tag="predictions",
        expected_ids=["train_0"],
    )
    assert parsed == {"train_0": [[0, 2], [2, 0]]}


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ('prose <predictions>{"train_0": [[1]]}</predictions>', "contain only"),
        ('<predictions>{"train_0": [[1]], "extra": [[1]]}</predictions>', "extra"),
        ('<predictions>{"train_0": [[true]]}</predictions>', "integer"),
        ('<predictions>{"train_0": [[10]]}</predictions>', "integer"),
        ('<predictions>{"train_0": [[1], [2, 3]]}</predictions>', "ragged"),
        ('<predictions>{"train_0": []}</predictions>', "non-empty"),
    ],
)
def test_parse_tagged_grids_rejects_invalid_contract(text: str, match: str) -> None:
    with pytest.raises(PredictionParseError, match=match):
        parse_tagged_grids(text, tag="predictions", expected_ids=["train_0"])


def test_parse_transform_description_is_single_non_python_block() -> None:
    assert (
        parse_transform_description("<transform_description>Reverse every row.</transform_description>")
        == "Reverse every row."
    )
    with pytest.raises(ValueError, match="only one"):
        parse_transform_description("Explanation: <transform_description>Reverse rows.</transform_description>")
    with pytest.raises(ValueError, match="Python"):
        parse_transform_description(
            "<transform_description>```python\ndef solve(grid): pass\n```</transform_description>"
        )


def test_compare_grid_renders_every_aligned_cell_as_predicted_correct() -> None:
    record = compare_grid(
        grid_id="train_1",
        predicted=[[0, 2, 0], [4, 4, 1]],
        correct=[[0, 3, 0], [4, 4, 1]],
    )
    assert not record.exact
    assert record.shape_match
    assert record.cell_accuracy == pytest.approx(5 / 6)
    assert record.feedback == ("Example train_1: mismatch\nDiff (predicted-correct):\n0-0 2-3 0-0\n4-4 4-4 1-1")


def test_compare_grid_reports_shape_without_cell_diff() -> None:
    record = compare_grid(
        grid_id="train_1",
        predicted=[[1, 2, 3], [4, 5, 6]],
        correct=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    )
    assert not record.shape_match
    assert "predicted 2x3, correct 3x3" in record.feedback
    assert "Diff" not in record.feedback


def test_verify_predictions_aggregates_all_requested_grids() -> None:
    result = verify_predictions(
        predictions={"train_0": [[1]], "train_1": [[0, 1]]},
        correct={"train_0": [[1]], "train_1": [[0, 2]]},
    )
    assert not result.all_exact
    assert result.exact_fraction == 0.5
    assert result.shape_match_fraction == 1.0
    assert result.cell_accuracy == 0.75


def test_episode_state_machine_success_and_format_failure() -> None:
    state = EpisodeState.initial()
    assert state.phase is EpisodePhase.PROPOSER
    state = state.description_generated()
    state = state.executor_format_failed()
    assert state.format_retry_used
    state = state.executor_format_failed()
    assert state.phase is EpisodePhase.TERMINATED
    assert state.termination_reason is TerminationReason.EXECUTOR_FORMAT_FAILURE

    state = EpisodeState.initial().description_generated()
    state = state.training_verified(all_exact=False)
    assert state.phase is EpisodePhase.PROPOSER
    assert state.round_index == 1
    state = state.description_generated().training_verified(all_exact=True)
    assert state.phase is EpisodePhase.EXECUTOR_TEST
    state = state.test_answered()
    assert state.termination_reason is TerminationReason.TRAIN_VERIFIED


def test_episode_state_machine_rejects_invalid_transition() -> None:
    with pytest.raises(RuntimeError, match="expected episode phase executor_test"):
        EpisodeState.initial().test_answered()


def test_context_budget_accounts_for_feedback_output_and_margin() -> None:
    budget = ContextBudget(
        model_context_limit=100,
        reserved_proposer_output_tokens=20,
        chat_template_margin=5,
    )
    assert budget.permits_revision(current_proposer_tokens=60, next_feedback_tokens=15)
    assert not budget.permits_revision(current_proposer_tokens=61, next_feedback_tokens=15)
    assert conservative_text_token_bound("abc") == 3


def test_leakage_guard_rejects_nested_verifier_fields() -> None:
    assert_model_request_safe({"input": [{"role": "user", "content": "safe"}]})
    with pytest.raises(ValueError, match="expected_output"):
        assert_model_request_safe({"input": [{"metadata": {"expected_output": [[9]]}}]})


def test_prompts_separate_public_and_hidden_fields() -> None:
    train = [{"input": [[1, 0]], "output": [[0, 1]]}]
    proposer = build_proposer_prompt(train_pairs=train, test_inputs=[[[2, 0]]])
    executor = build_executor_prompt(
        description="Reverse each row.",
        inputs={"train_0": [[1, 0]]},
        tag="predictions",
    )
    assert "0 1" in proposer
    assert "Test input test_0" in proposer
    assert "0 1" not in executor
    assert "train_0" in executor


def test_parse_canonical_rule_rerenders_in_canonical_order() -> None:
    text = (
        "scratchpad thoughts\n"
        "<puzzle_concepts>symmetry</puzzle_concepts>\n"
        "<rules_summary>Mirror the grid.</rules_summary>\n"
        "<solution_steps>1. Flip horizontally.</solution_steps>\n"
        "<key_insight>The axis is vertical.</key_insight>"
    )
    rendered = parse_canonical_rule(text)
    assert rendered == (
        "<rules_summary>\nMirror the grid.\n</rules_summary>\n\n"
        "<solution_steps>\n1. Flip horizontally.\n</solution_steps>\n\n"
        "<key_insight>\nThe axis is vertical.\n</key_insight>\n\n"
        "<puzzle_concepts>\nsymmetry\n</puzzle_concepts>"
    )


def test_parse_canonical_rule_takes_last_occurrence() -> None:
    text = (
        "<rules_summary>draft</rules_summary>"
        "<solution_steps>s</solution_steps><key_insight>k</key_insight>"
        "<puzzle_concepts>p</puzzle_concepts>"
        "<rules_summary>final</rules_summary>"
    )
    assert "final" in parse_canonical_rule(text)
    assert "draft" not in parse_canonical_rule(text)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("<rules_summary>only one section</rules_summary>", "solution_steps"),
        (
            "<rules_summary>r</rules_summary><solution_steps></solution_steps>"
            "<key_insight>k</key_insight><puzzle_concepts>p</puzzle_concepts>",
            "solution_steps",
        ),
        (
            "<rules_summary>r</rules_summary><solution_steps>def f(x):</solution_steps>"
            "<key_insight>k</key_insight><puzzle_concepts>p</puzzle_concepts>",
            "Python",
        ),
    ],
)
def test_parse_canonical_rule_rejects_invalid(text: str, match: str) -> None:
    with pytest.raises(TransformDescriptionParseError, match=match):
        parse_canonical_rule(text)


def test_single_executor_prompt_matches_the_training_contract() -> None:
    prompt = build_single_executor_prompt(description="Reverse each row.", input_grid=[[1, 0]])
    assert prompt.startswith("You are an exact grid-transformation executor.")
    assert "<transformation>\nReverse each row.\n</transformation>" in prompt
    assert "<input>\n1 0\n</input>" in prompt
    assert prompt.endswith("Preserve the exact output shape implied by the operation.\n")
    assert "JSON" in prompt  # the no-JSON instruction is part of the contract


def test_nvarc_proposer_prompt_shows_demos_but_never_eval_grids() -> None:
    prompt = build_nvarc_proposer_prompt(demo_pairs=[{"input": [[1, 0]], "output": [[0, 1]]}])
    assert "demo_0" in prompt
    assert "1 0" in prompt and "0 1" in prompt
    assert "rules_summary" in prompt and "puzzle_concepts" in prompt
    assert "Test input" not in prompt


def test_eval_feedback_renders_all_behavioral_evidence() -> None:
    feedback = build_eval_feedback_prompt(
        grid_id="test_1",
        input_grid=[[1, 0]],
        predicted=[[1, 1]],
        expected=[[0, 1]],
        diff_feedback="Example test_1: mismatch",
    )
    assert "test_1" in feedback
    assert "Input:\n1 0" in feedback
    assert "Executor output:\n1 1" in feedback
    assert "Expected output:\n0 1" in feedback
    assert "mismatch" in feedback
    assert "replacement rule" in feedback


def test_eval_sequence_state_transitions() -> None:
    state = EpisodeState.initial().description_generated()
    state = state.eval_grid_solved(all_solved=False)
    assert state.phase is EpisodePhase.EXECUTOR_TRAIN
    failed = state.eval_grid_failed()
    assert failed.phase is EpisodePhase.PROPOSER
    assert failed.round_index == 1
    done = state.eval_grid_solved(all_solved=True)
    assert done.termination_reason is TerminationReason.ALL_SOLVED
    reset = state.next_grid()
    assert not reset.format_retry_used

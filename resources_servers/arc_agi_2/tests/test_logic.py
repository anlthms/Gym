import pytest

from resources_servers.arc_agi_2.logic import (
    ContextBudget,
    EpisodePhase,
    EpisodeState,
    PredictionParseError,
    TerminationReason,
    assert_model_request_safe,
    build_executor_prompt,
    build_proposer_prompt,
    compare_grid,
    conservative_text_token_bound,
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

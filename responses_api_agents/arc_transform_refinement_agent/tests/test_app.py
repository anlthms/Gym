import json
from unittest.mock import AsyncMock, MagicMock

from fastapi import Request

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from responses_api_agents.arc_transform_refinement_agent.app import (
    ArcTransformRefinementAgent,
    ArcTransformRefinementAgentConfig,
    ArcTransformRunRequest,
)


def _config(**overrides) -> ArcTransformRefinementAgentConfig:
    values = {
        "host": "0.0.0.0",
        "port": 8080,
        "entrypoint": "app.py",
        "name": "arc_refinement",
        "resources_server": ResourcesServerRef(type="resources_servers", name="arc_resource"),
        "proposer_model_server": ModelServerRef(type="responses_api_models", name="proposer"),
        "executor_model_server": ModelServerRef(type="responses_api_models", name="executor"),
        "model_context_limit": 1024,
        "reserved_proposer_output_tokens": 128,
        "chat_template_margin": 32,
        "proposer_max_output_tokens": 128,
        "executor_max_output_tokens": 128,
        "max_rounds": 8,
    }
    values.update(overrides)
    return ArcTransformRefinementAgentConfig(**values)


def _response(text: str, *, prompt_tokens: int, generation_tokens: int) -> NeMoGymResponse:
    return NeMoGymResponse.model_validate(
        {
            "id": f"response-{text[:8]}",
            "created_at": 0.0,
            "model": "mock",
            "object": "response",
            "output": [
                {
                    "id": "message",
                    "content": [{"annotations": [], "text": text, "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                    "prompt_token_ids": list(range(prompt_tokens)),
                    "generation_token_ids": list(range(generation_tokens)),
                    "generation_log_probs": [-0.1] * generation_tokens,
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
    )


class _SeedResponse:
    ok = True
    cookies = {"session": "seeded"}


def test_config_rejects_invalid_context_reservation() -> None:
    try:
        _config(model_context_limit=100, reserved_proposer_output_tokens=90, chat_template_margin=10)
    except ValueError as error:
        assert "must fit" in str(error)
    else:
        raise AssertionError("invalid context reservation was accepted")


CANONICAL_RULE = (
    "<rules_summary>Reverse each row.</rules_summary>\n"
    "<solution_steps>Reverse the cell order of every row.</solution_steps>\n"
    "<key_insight>Only the horizontal order changes.</key_insight>\n"
    "<puzzle_concepts>reversal</puzzle_concepts>"
)


class MockEvalAgent(ArcTransformRefinementAgent):
    def set_model_responses(self, responses: list[NeMoGymResponse]) -> None:
        object.__setattr__(self, "model_responses", list(responses))
        object.__setattr__(self, "model_requests", [])

    async def _call_model(self, *, request, server, params, cookies, run_body):
        self.model_requests.append((server.name, params))
        return self.model_responses.pop(0), {}

    async def _call_resource(self, *, url_path, payload, cookies):
        self.resource_requests.append((url_path, payload))
        if url_path == "/verify_eval_grid":
            return self.eval_results.pop(0), cookies
        if url_path == "/finalize":
            response = {
                "responses_create_params": payload["responses_create_params"],
                "response": payload["response"],
                "reward": 0.0,
                "termination_reason": payload["termination_reason"],
                "loss_masked": payload["loss_masked"],
                "proposer_format_failure": payload["proposer_format_failure"],
                "protocol": payload["protocol"],
                "trace": payload["trace"],
            }
            return response, cookies
        raise AssertionError(f"unexpected resource path {url_path}")


def _eval_agent(
    model_responses,
    eval_results,
    *,
    config: ArcTransformRefinementAgentConfig | None = None,
) -> MockEvalAgent:
    agent = MockEvalAgent(
        config=config or _config(protocol="eval_sequence"),
        server_client=MagicMock(spec=ServerClient),
    )
    agent.server_client.post = AsyncMock(return_value=_SeedResponse())
    agent.set_model_responses(model_responses)
    object.__setattr__(agent, "eval_results", list(eval_results))
    object.__setattr__(agent, "resource_requests", [])
    return agent


def _eval_body() -> ArcTransformRunRequest:
    return ArcTransformRunRequest(
        responses_create_params={"input": [], "temperature": 0.0},
        train=[{"input": [[1, 0]], "output": [[0, 1]]}],
        test=[
            {"input": [[2, 0]], "output": [[0, 2]]},
            {"input": [[3, 0]], "output": [[0, 3]]},
        ],
        task_id="eval-task",
    )


async def test_eval_sequence_advances_on_solve_and_sweeps_with_the_final_rule() -> None:
    evidence = "EVIDENCE test_1: expected differs"
    agent = _eval_agent(
        [
            _response(CANONICAL_RULE, prompt_tokens=100, generation_tokens=20),
            _response("<answer>\n0 2\n</answer>", prompt_tokens=60, generation_tokens=6),
            _response("<answer>\n3 3\n</answer>", prompt_tokens=60, generation_tokens=6),
            _response(REVISED_RULE, prompt_tokens=160, generation_tokens=20),
            _response("<answer>\n0 3\n</answer>", prompt_tokens=60, generation_tokens=6),
            # Final sweep: test_0 was last attempted with the FIRST rule, so
            # the final rule answers it once more before finalize.
            _response("<answer>\n0 2\n</answer>", prompt_tokens=60, generation_tokens=6),
        ],
        [
            {"grid_id": "test_0", "format_valid": True, "exact": True, "feedback": "exact", "revision_feedback": None},
            {
                "grid_id": "test_1",
                "format_valid": True,
                "exact": False,
                "feedback": "mismatch",
                "revision_feedback": evidence,
            },
            {"grid_id": "test_1", "format_valid": True, "exact": True, "feedback": "exact", "revision_feedback": None},
            {"grid_id": "test_0", "format_valid": True, "exact": True, "feedback": "exact", "revision_feedback": None},
        ],
    )
    request = MagicMock(spec=Request)
    request.cookies = {}

    result = await agent.run(request, _eval_body())

    assert result.termination_reason == "all_solved"
    assert not result.loss_masked
    roles = [name for name, _ in agent.model_requests]
    assert roles == ["proposer", "executor", "executor", "proposer", "executor", "executor"]

    # Executor calls: fresh single-grid sessions, one user message, no
    # instructions (byte-parity with native executor training rows).
    for index in (1, 2, 4, 5):
        _, params = agent.model_requests[index]
        assert "instructions" not in params
        assert len(params["input"]) == 1
        assert "<transformation>" in params["input"][0]["content"]
    # The rule is applied to exactly one grid per call.
    assert "2 0" in agent.model_requests[1][1]["input"][0]["content"]
    assert "3 0" not in agent.model_requests[1][1]["input"][0]["content"]
    # The sweep call re-answers test_0 with the FINAL (revised) rule.
    sweep_call = agent.model_requests[5][1]["input"][0]["content"]
    assert "2 0" in sweep_call
    assert "recolor" in sweep_call

    # The revision proposer turn carries the server-rendered evidence.
    second_proposer_input = json.dumps(agent.model_requests[3][1]["input"])
    assert evidence in second_proposer_input

    # The solved grid's hidden target never appears in any model request.
    all_model_requests = json.dumps([payload for _, payload in agent.model_requests])
    assert "[[0, 2]]" not in all_model_requests
    assert "0 2" not in json.dumps(agent.model_requests[0][1])  # not in the initial prompt

    # Every grid's last verification used the final rule.
    verify_ids = [payload["grid_id"] for path, payload in agent.resource_requests if path == "/verify_eval_grid"]
    assert verify_ids == ["test_0", "test_1", "test_1", "test_0"]

    finalize_payload = agent.resource_requests[-1][1]
    assert finalize_payload["protocol"] == "eval_sequence"
    assert finalize_payload["response"]["output"][0]["content"][0]["text"] == REVISED_RULE


async def test_eval_sequence_budget_exhaustion_sweeps_unreached_grids() -> None:
    evidence = "EVIDENCE test_0: expected differs, with enough bytes to overflow the tiny budget"
    agent = _eval_agent(
        [
            _response(CANONICAL_RULE, prompt_tokens=100, generation_tokens=20),
            _response("<answer>\n2 2\n</answer>", prompt_tokens=60, generation_tokens=6),
            # Final sweep reaches test_1 even though the sequence never did;
            # test_0 is skipped (already last-attempted with the final rule).
            _response("<answer>\n0 3\n</answer>", prompt_tokens=60, generation_tokens=6),
        ],
        [
            {
                "grid_id": "test_0",
                "format_valid": True,
                "exact": False,
                "feedback": "mismatch",
                "revision_feedback": evidence,
            },
            {"grid_id": "test_1", "format_valid": True, "exact": True, "feedback": "exact", "revision_feedback": None},
        ],
        config=_config(protocol="eval_sequence", model_context_limit=300),
    )
    request = MagicMock(spec=Request)
    request.cookies = {}

    result = await agent.run(request, _eval_body())

    assert result.termination_reason == "context_exhausted"
    assert not result.loss_masked
    roles = [name for name, _ in agent.model_requests]
    assert roles == ["proposer", "executor", "executor"]
    verify_ids = [payload["grid_id"] for path, payload in agent.resource_requests if path == "/verify_eval_grid"]
    assert verify_ids == ["test_0", "test_1"]


async def test_eval_sequence_masks_double_executor_format_failure() -> None:
    agent = _eval_agent(
        [
            _response(CANONICAL_RULE, prompt_tokens=100, generation_tokens=20),
            _response("no grid", prompt_tokens=60, generation_tokens=4),
            _response("still no grid", prompt_tokens=70, generation_tokens=4),
        ],
        [
            {
                "grid_id": "test_0",
                "format_valid": False,
                "exact": False,
                "feedback": "format",
                "parse_error": "no grid",
                "revision_feedback": None,
            },
            {
                "grid_id": "test_0",
                "format_valid": False,
                "exact": False,
                "feedback": "format",
                "parse_error": "no grid",
                "revision_feedback": None,
            },
        ],
    )
    request = MagicMock(spec=Request)
    request.cookies = {}

    result = await agent.run(request, _eval_body())

    assert result.termination_reason == "executor_format_failure"
    assert result.loss_masked
    retry_input = agent.model_requests[2][1]["input"]
    assert len(retry_input) == 3
    assert "Do not change or reinterpret the transformation" in retry_input[-1]["content"]


async def test_eval_sequence_floors_non_canonical_proposer_output_with_loss_on() -> None:
    agent = _eval_agent(
        [
            _response(
                "<transform_description>not canonical</transform_description>", prompt_tokens=90, generation_tokens=8
            )
        ],
        [],
    )
    request = MagicMock(spec=Request)
    request.cookies = {}

    result = await agent.run(request, _eval_body())

    # Proposer-caused failure: reward floor with loss ON, never masked.
    assert result.termination_reason == "agent_error"
    assert not result.loss_masked
    assert result.proposer_format_failure
    assert len(agent.model_requests) == 1


REVISED_RULE = (
    "<rules_summary>Reverse each row and recolor.</rules_summary>\n"
    "<solution_steps>Reverse the cell order of every row, keep colors.</solution_steps>\n"
    "<key_insight>Only the horizontal order changes.</key_insight>\n"
    "<puzzle_concepts>reversal</puzzle_concepts>"
)


def _hidden_agent(model_responses, eval_results, **config_overrides) -> MockEvalAgent:
    return _eval_agent(model_responses, eval_results, config=_config(**config_overrides))


def _hidden_body() -> ArcTransformRunRequest:
    return ArcTransformRunRequest(
        responses_create_params={"input": [], "temperature": 0.0},
        train=[{"input": [[1, 0]], "output": [[0, 1]]}],
        test=[{"input": [[3, 0]], "output": [[0, 3]]}],
        task_id="hidden-task",
    )


def _demo_miss(evidence: str) -> dict:
    return {
        "grid_id": "train_0",
        "format_valid": True,
        "exact": False,
        "feedback": "mismatch",
        "revision_feedback": evidence,
    }


def _exact(grid_id: str) -> dict:
    return {"grid_id": grid_id, "format_valid": True, "exact": True, "feedback": "exact", "revision_feedback": None}


async def test_hidden_test_refines_on_demos_then_answers_the_hidden_test() -> None:
    evidence = "EVIDENCE train_0: expected differs"
    agent = _hidden_agent(
        [
            _response(CANONICAL_RULE, prompt_tokens=100, generation_tokens=20),
            _response("<answer>\n1 0\n</answer>", prompt_tokens=60, generation_tokens=6),
            _response(REVISED_RULE, prompt_tokens=160, generation_tokens=20),
            _response("<answer>\n0 1\n</answer>", prompt_tokens=60, generation_tokens=6),
            _response("<answer>\n0 3\n</answer>", prompt_tokens=60, generation_tokens=6),
        ],
        [_demo_miss(evidence), _exact("train_0"), _exact("test_0")],
    )
    request = MagicMock(spec=Request)
    request.cookies = {}

    result = await agent.run(request, _hidden_body())

    assert result.termination_reason == "train_verified"
    assert not result.loss_masked
    roles = [name for name, _ in agent.model_requests]
    assert roles == ["proposer", "executor", "proposer", "executor", "executor"]

    # Executor calls: fresh single-grid sessions, one user message, no
    # instructions (byte-parity with native executor training rows).
    for index in (1, 3, 4):
        _, params = agent.model_requests[index]
        assert "instructions" not in params
        assert len(params["input"]) == 1
        assert "<transformation>" in params["input"][0]["content"]
    # The final test call applies the revised rule to the hidden test input.
    final_call = agent.model_requests[4][1]["input"][0]["content"]
    assert "3 0" in final_call
    assert "recolor" in final_call

    # The revision proposer turn carries the server-rendered evidence.
    assert evidence in json.dumps(agent.model_requests[2][1]["input"])

    # The hidden test target never appears in any model request.
    all_model_requests = json.dumps([payload for _, payload in agent.model_requests])
    assert "[[0, 3]]" not in all_model_requests
    assert "0 3" not in json.dumps(agent.model_requests[0][1])

    # Verified grid ids: the demo twice, then the hidden test once.
    verify_ids = [payload["grid_id"] for path, payload in agent.resource_requests if path == "/verify_eval_grid"]
    assert verify_ids == ["train_0", "train_0", "test_0"]

    finalize_payload = agent.resource_requests[-1][1]
    assert finalize_payload["protocol"] == "hidden_test"
    assert finalize_payload["response"]["output"][0]["content"][0]["text"] == REVISED_RULE
    assert finalize_payload["trace"]["policy_loss"]["round_index"] == 1


async def test_hidden_test_budget_exhaustion_still_answers_the_test() -> None:
    evidence = "EVIDENCE train_0: expected differs, with enough bytes to overflow the tiny budget"
    agent = _hidden_agent(
        [
            _response(CANONICAL_RULE, prompt_tokens=100, generation_tokens=20),
            _response("<answer>\n1 0\n</answer>", prompt_tokens=60, generation_tokens=6),
            _response("<answer>\n0 3\n</answer>", prompt_tokens=60, generation_tokens=6),
        ],
        [_demo_miss(evidence), _exact("test_0")],
        model_context_limit=300,
    )
    request = MagicMock(spec=Request)
    request.cookies = {}

    result = await agent.run(request, _hidden_body())

    assert result.termination_reason == "context_exhausted"
    assert not result.loss_masked
    # The demo loop ended on budget, but the hidden test was still answered
    # with the current rule.
    verify_ids = [payload["grid_id"] for path, payload in agent.resource_requests if path == "/verify_eval_grid"]
    assert verify_ids == ["train_0", "test_0"]


async def test_row_context_limit_override_unthrottles_the_budget() -> None:
    """A per-row model_context_limit lifts the config budget for that episode."""
    evidence = "EVIDENCE train_0: expected differs, with enough bytes to overflow the tiny budget"
    agent = _hidden_agent(
        [
            _response(CANONICAL_RULE, prompt_tokens=100, generation_tokens=20),
            _response("<answer>\n1 0\n</answer>", prompt_tokens=60, generation_tokens=6),
            _response(REVISED_RULE, prompt_tokens=200, generation_tokens=20),
            _response("<answer>\n0 1\n</answer>", prompt_tokens=60, generation_tokens=6),
            _response("<answer>\n0 3\n</answer>", prompt_tokens=60, generation_tokens=6),
        ],
        [_demo_miss(evidence), _exact("train_0"), _exact("test_0")],
        model_context_limit=300,  # exhausts after the first miss without the override
    )
    request = MagicMock(spec=Request)
    request.cookies = {}
    body = _hidden_body().model_copy(update={"model_context_limit": 2048})

    result = await agent.run(request, body)

    # The row-level limit permitted the revision the config limit would deny.
    assert result.termination_reason == "train_verified"
    assert not result.loss_masked
    roles = [name for name, _ in agent.model_requests]
    assert roles == ["proposer", "executor", "proposer", "executor", "executor"]


async def test_hidden_test_masks_double_executor_format_failure_without_answering() -> None:
    failure = {
        "grid_id": "train_0",
        "format_valid": False,
        "exact": False,
        "feedback": "format",
        "parse_error": "no grid",
        "revision_feedback": None,
    }
    agent = _hidden_agent(
        [
            _response(CANONICAL_RULE, prompt_tokens=100, generation_tokens=20),
            _response("no grid", prompt_tokens=60, generation_tokens=4),
            _response("still no grid", prompt_tokens=70, generation_tokens=4),
        ],
        [failure, dict(failure)],
    )
    request = MagicMock(spec=Request)
    request.cookies = {}

    result = await agent.run(request, _hidden_body())

    assert result.termination_reason == "executor_format_failure"
    assert result.loss_masked
    verify_ids = [payload["grid_id"] for path, payload in agent.resource_requests if path == "/verify_eval_grid"]
    assert verify_ids == ["train_0", "train_0"]  # the hidden test was never reached


async def test_hidden_test_unparseable_first_rule_is_floored_with_loss_on() -> None:
    agent = _hidden_agent(
        [_response("not a rule", prompt_tokens=90, generation_tokens=8)],
        [],
    )
    request = MagicMock(spec=Request)
    request.cookies = {}

    result = await agent.run(request, _hidden_body())

    assert result.termination_reason == "agent_error"
    assert not result.loss_masked
    assert result.proposer_format_failure
    assert len(agent.model_requests) == 1


async def test_hidden_test_parse_failure_falls_back_to_the_prior_rule() -> None:
    evidence = "EVIDENCE train_0: expected differs"
    agent = _hidden_agent(
        [
            _response(CANONICAL_RULE, prompt_tokens=100, generation_tokens=20),
            _response("<answer>\n1 0\n</answer>", prompt_tokens=60, generation_tokens=6),
            _response("not a rule anymore", prompt_tokens=160, generation_tokens=8),
            _response("<answer>\n0 3\n</answer>", prompt_tokens=60, generation_tokens=6),
        ],
        [_demo_miss(evidence), _exact("test_0")],
    )
    request = MagicMock(spec=Request)
    request.cookies = {}

    result = await agent.run(request, _hidden_body())

    # The trained final turn is the unparseable rule: floored, loss ON.
    assert result.termination_reason == "agent_error"
    assert not result.loss_masked
    assert result.proposer_format_failure
    # The hidden test was still answered, using the first (valid) rule.
    final_call = agent.model_requests[-1][1]["input"][0]["content"]
    assert "3 0" in final_call
    assert "Reverse each row." in final_call


async def test_row_protocol_overrides_the_config_default() -> None:
    """A hidden_test-configured agent runs eval_sequence when the row says so."""
    agent = _hidden_agent(
        [
            _response(CANONICAL_RULE, prompt_tokens=100, generation_tokens=20),
            _response("<answer>\n0 3\n</answer>", prompt_tokens=60, generation_tokens=6),
        ],
        [_exact("test_0")],
    )
    request = MagicMock(spec=Request)
    request.cookies = {}
    body = _hidden_body().model_copy(update={"protocol": "eval_sequence"})

    result = await agent.run(request, body)

    assert result.termination_reason == "all_solved"
    finalize_payload = agent.resource_requests[-1][1]
    assert finalize_payload["protocol"] == "eval_sequence"

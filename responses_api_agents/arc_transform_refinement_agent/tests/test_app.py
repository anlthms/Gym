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


class MockArcAgent(ArcTransformRefinementAgent):
    def set_model_responses(self, responses: list[NeMoGymResponse]) -> None:
        object.__setattr__(self, "model_responses", list(responses))
        object.__setattr__(self, "model_requests", [])

    async def _call_model(self, *, request, server, params, cookies, run_body):
        self.model_requests.append((server.name, params))
        return self.model_responses.pop(0), {}

    async def _call_resource(self, *, url_path, payload, cookies):
        self.resource_requests.append((url_path, payload))
        if url_path == "/verify_training":
            return self.training_results.pop(0), cookies
        if url_path == "/verify_test":
            return self.test_results.pop(0), cookies
        if url_path == "/finalize":
            response = {
                "responses_create_params": payload["responses_create_params"],
                "response": payload["response"],
                "reward": float(self.final_test_exact),
                "termination_reason": payload["termination_reason"],
                "loss_masked": payload["loss_masked"],
                "trace": payload["trace"],
                "test_exact": self.final_test_exact,
            }
            return response, cookies
        raise AssertionError(f"unexpected resource path {url_path}")


def _agent(
    model_responses,
    training_results,
    test_results=(),
    *,
    config: ArcTransformRefinementAgentConfig | None = None,
) -> MockArcAgent:
    agent = MockArcAgent(
        config=config or _config(),
        server_client=MagicMock(spec=ServerClient),
    )
    agent.server_client.post = AsyncMock(return_value=_SeedResponse())
    agent.set_model_responses(model_responses)
    object.__setattr__(agent, "training_results", list(training_results))
    object.__setattr__(agent, "test_results", list(test_results))
    object.__setattr__(agent, "resource_requests", [])
    object.__setattr__(
        agent,
        "final_test_exact",
        bool(test_results and test_results[-1].get("all_exact")),
    )
    return agent


def _body() -> ArcTransformRunRequest:
    return ArcTransformRunRequest(
        responses_create_params={"input": [], "temperature": 0.0},
        train=[{"input": [[1, 0]], "output": [[0, 1]]}],
        test=[{"input": [[3, 0]], "output": [[0, 8]]}],
        task_id="mock-task",
    )


async def test_two_round_agent_isolates_histories_and_returns_only_final_proposer() -> None:
    proposer_v0 = _response(
        "<transform_description>Keep the grid unchanged.</transform_description>",
        prompt_tokens=100,
        generation_tokens=12,
    )
    executor_v0 = _response(
        '<predictions>{"train_0": [[1, 0]]}</predictions>',
        prompt_tokens=80,
        generation_tokens=8,
    )
    proposer_v1 = _response(
        "<transform_description>Reverse every row.</transform_description>",
        prompt_tokens=180,
        generation_tokens=12,
    )
    executor_v1 = _response(
        '<predictions>{"train_0": [[0, 1]]}</predictions>',
        prompt_tokens=82,
        generation_tokens=8,
    )
    test_answer = _response(
        '<answers>{"test_0": [[0, 8]]}</answers>',
        prompt_tokens=100,
        generation_tokens=8,
    )
    agent = _agent(
        [proposer_v0, executor_v0, proposer_v1, executor_v1, test_answer],
        [
            {
                "format_valid": True,
                "all_exact": False,
                "feedback": "Example train_0: mismatch\nDiff (predicted-correct):\n1-0 0-1",
                "parse_error": None,
                "predictions": {"train_0": [[1, 0]]},
            },
            {
                "format_valid": True,
                "all_exact": True,
                "feedback": "Example train_0: exact.",
                "parse_error": None,
                "predictions": {"train_0": [[0, 1]]},
            },
        ],
        [
            {
                "format_valid": True,
                "all_exact": True,
                "feedback": "Example test_0: exact.",
                "parse_error": None,
            }
        ],
    )
    request = MagicMock(spec=Request)
    request.cookies = {}

    result = await agent.run(request, _body())

    assert result.reward == 1.0
    assert result.termination_reason == "train_verified"
    assert not result.loss_masked
    assert result.response.output[0].content[0].text == proposer_v1.output[0].content[0].text

    roles = [name for name, _ in agent.model_requests]
    assert roles == ["proposer", "executor", "proposer", "executor", "executor"]
    assert "rule proposer" in agent.model_requests[0][1]["instructions"]
    assert "literal ARC transformation executor" in agent.model_requests[1][1]["instructions"]
    assert agent.model_requests[0][1]["instructions"] != agent.model_requests[1][1]["instructions"]
    second_proposer_input = agent.model_requests[2][1]["input"]
    serialized_second_proposer = json.dumps(second_proposer_input)
    assert "Keep the grid unchanged" in serialized_second_proposer
    assert "predicted-correct" in serialized_second_proposer
    assert '<predictions>{"train_0": [[1, 0]]}' not in serialized_second_proposer

    first_executor_input = agent.model_requests[1][1]["input"]
    second_executor_input = agent.model_requests[3][1]["input"]
    assert len(first_executor_input) == 1
    assert len(second_executor_input) == 1
    assert "Keep the grid unchanged" in first_executor_input[0]["content"]
    assert "Reverse every row" in second_executor_input[0]["content"]

    all_model_requests = json.dumps([payload for _, payload in agent.model_requests])
    assert "[[0, 8]]" not in all_model_requests
    finalize_payload = agent.resource_requests[-1][1]
    assert finalize_payload["response"] == proposer_v1.model_dump()
    assert finalize_payload["trace"]["policy_loss"]["round_index"] == 1
    assert finalize_payload["trace"]["policy_loss"]["role"] == "proposer"


async def test_executor_gets_one_format_retry_then_masks_episode() -> None:
    proposer = _response(
        "<transform_description>Reverse every row.</transform_description>",
        prompt_tokens=100,
        generation_tokens=10,
    )
    invalid_first = _response("[[0, 1]]", prompt_tokens=80, generation_tokens=4)
    invalid_second = _response("still invalid", prompt_tokens=100, generation_tokens=4)
    agent = _agent(
        [proposer, invalid_first, invalid_second],
        [
            {
                "format_valid": False,
                "all_exact": False,
                "feedback": "Executor format failure",
                "parse_error": "missing predictions tags",
                "predictions": None,
            },
            {
                "format_valid": False,
                "all_exact": False,
                "feedback": "Executor format failure",
                "parse_error": "still invalid",
                "predictions": None,
            },
        ],
    )
    request = MagicMock(spec=Request)
    request.cookies = {}

    result = await agent.run(request, _body())

    assert result.termination_reason == "executor_format_failure"
    assert result.loss_masked
    assert len(agent.model_requests) == 3
    retry_input = agent.model_requests[2][1]["input"]
    assert len(retry_input) == 3
    assert "Do not change or reinterpret the transformation" in retry_input[-1]["content"]


async def test_context_budget_stops_before_another_proposer_turn_without_masking() -> None:
    proposer = _response(
        "<transform_description>Keep the grid unchanged.</transform_description>",
        prompt_tokens=100,
        generation_tokens=10,
    )
    executor = _response(
        '<predictions>{"train_0": [[1, 0]]}</predictions>',
        prompt_tokens=80,
        generation_tokens=8,
    )
    agent = _agent(
        [proposer, executor],
        [
            {
                "format_valid": True,
                "all_exact": False,
                "feedback": "Example train_0: mismatch\nDiff (predicted-correct):\n1-0 0-1",
                "parse_error": None,
                "predictions": {"train_0": [[1, 0]]},
            }
        ],
        config=_config(model_context_limit=300),
    )
    request = MagicMock(spec=Request)
    request.cookies = {}

    result = await agent.run(request, _body())

    assert result.termination_reason == "context_exhausted"
    assert not result.loss_masked
    assert len(agent.model_requests) == 2


def test_config_rejects_invalid_context_reservation() -> None:
    try:
        _config(model_context_limit=100, reserved_proposer_output_tokens=90, chat_template_margin=10)
    except ValueError as error:
        assert "must fit" in str(error)
    else:
        raise AssertionError("invalid context reservation was accepted")

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from unittest.mock import MagicMock

import pytest
from fastapi import Request

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.arc_agi_2.app import (
    ARCAGIFinalizeRequest,
    ARCAGIResourcesServer,
    ARCAGIResourcesServerConfig,
    ARCAGIRunRequest,
    ARCAGIVerifyRequest,
    EvalGridVerificationRequest,
    ExecutorVerificationRequest,
    _parse_grid,
)


class TestApp:
    def test_sanity(self) -> None:
        config = ARCAGIResourcesServerConfig(
            host="127.0.0.1",
            port=8080,
            entrypoint="app.py",
            name="test_arc_agi_2",
        )
        ARCAGIResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_parse_grid(self) -> None:
        grid_text = "[[1,2],[3,4]]"
        result = _parse_grid(grid_text)
        assert result == [[1, 2], [3, 4]]

        grid_text = "[[1,2,3],[4,5,6],[7,8,9]]"
        result = _parse_grid(grid_text)
        assert result == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        grid_text = "[1,2,3,4"
        result = _parse_grid(grid_text)
        assert result is None

        result = _parse_grid("")
        assert result is None

    async def test_session_verification_keeps_targets_server_side(self) -> None:
        config = ARCAGIResourcesServerConfig(
            host="127.0.0.1",
            port=8080,
            entrypoint="app.py",
            name="test_arc_agi_2",
        )
        server = ARCAGIResourcesServer(
            config=config,
            server_client=MagicMock(spec=ServerClient),
        )
        request = MagicMock(spec=Request)
        request.session = {"session_id": "arc-test"}
        run_request = ARCAGIRunRequest(
            responses_create_params={"input": []},
            train=[{"input": [[1, 0]], "output": [[0, 1]]}],
            test=[{"input": [[2, 0]], "output": [[0, 2]]}],
            task_id="task",
        )
        await server.seed_session(request, run_request)

        model_response = NeMoGymResponse.model_validate(
            {
                "id": "response",
                "created_at": 0.0,
                "model": "mock",
                "object": "response",
                "output": [
                    {
                        "id": "message",
                        "content": [
                            {
                                "annotations": [],
                                "text": '<predictions>{"train_0": [[0, 1]]}</predictions>',
                                "type": "output_text",
                            }
                        ],
                        "role": "assistant",
                        "status": "completed",
                        "type": "message",
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            }
        )
        result = await server.verify_training(
            request,
            ExecutorVerificationRequest(response=model_response),
        )
        assert result.format_valid
        assert result.all_exact
        assert server._sessions["arc-test"].test_targets == {"test_0": [[0, 2]]}

    async def test_training_parse_failure_is_not_misdiagnosed_as_rule_failure(self) -> None:
        config = ARCAGIResourcesServerConfig(
            host="127.0.0.1",
            port=8080,
            entrypoint="app.py",
            name="test_arc_agi_2",
        )
        server = ARCAGIResourcesServer(
            config=config,
            server_client=MagicMock(spec=ServerClient),
        )
        request = MagicMock(spec=Request)
        request.session = {"session_id": "arc-format"}
        await server.seed_session(
            request,
            ARCAGIRunRequest(
                responses_create_params={"input": []},
                train=[{"input": [[1]], "output": [[2]]}],
                test=[{"input": [[3]], "output": [[4]]}],
            ),
        )
        response = NeMoGymResponse.model_validate(
            {
                "id": "response",
                "created_at": 0.0,
                "model": "mock",
                "object": "response",
                "output": [
                    {
                        "id": "message",
                        "content": [{"annotations": [], "text": "[[2]]", "type": "output_text"}],
                        "role": "assistant",
                        "status": "completed",
                        "type": "message",
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            }
        )
        result = await server.verify_training(
            request,
            ExecutorVerificationRequest(response=response),
        )
        assert not result.format_valid
        assert result.parse_error is not None
        assert not server._sessions["arc-format"].train_results


def _server() -> ARCAGIResourcesServer:
    config = ARCAGIResourcesServerConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="app.py",
        name="test_arc_agi_2",
    )
    return ARCAGIResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _request(session_id: str) -> Request:
    request = MagicMock(spec=Request)
    request.session = {"session_id": session_id}
    return request


def _text_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse.model_validate(
        {
            "id": "response",
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
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
    )


def _finalize_request(*, protocol: str, loss_masked: bool = False, rounds: int = 1) -> ARCAGIFinalizeRequest:
    return ARCAGIFinalizeRequest(
        responses_create_params={"input": []},
        response=_text_response("<rules_summary>r</rules_summary>").model_dump(),
        termination_reason="all_solved",
        loss_masked=loss_masked,
        protocol=protocol,
        trace={"rounds": [{}] * rounds},
    )


class TestEvalSequence:
    async def _seeded(self, session_id: str) -> tuple[ARCAGIResourcesServer, Request]:
        server = _server()
        request = _request(session_id)
        await server.seed_session(
            request,
            ARCAGIRunRequest(
                responses_create_params={"input": []},
                train=[{"input": [[1, 0]], "output": [[0, 1]]}],
                test=[
                    {"input": [[2, 0]], "output": [[0, 2]]},
                    {"input": [[3, 0]], "output": [[0, 3]]},
                ],
                task_id="eval-task",
            ),
        )
        return server, request

    async def test_verify_eval_grid_scores_and_records(self) -> None:
        server, request = await self._seeded("eval-exact")
        result = await server.verify_eval_grid(
            request,
            EvalGridVerificationRequest(response=_text_response("<answer>\n0 2\n</answer>"), grid_id="test_0"),
        )
        assert result.format_valid and result.exact
        assert result.terms["grid_match"] == 1.0
        assert result.revision_feedback is None
        assert server._sessions["eval-exact"].eval_results["test_0"][0]["grid_match"] == 1.0

    async def test_verify_eval_grid_renders_behavioral_evidence_on_miss(self) -> None:
        server, request = await self._seeded("eval-miss")
        result = await server.verify_eval_grid(
            request,
            EvalGridVerificationRequest(response=_text_response("<answer>\n2 2\n</answer>"), grid_id="test_1"),
        )
        assert result.format_valid and not result.exact
        assert result.revision_feedback is not None
        assert "Input:\n3 0" in result.revision_feedback
        assert "Executor output:\n2 2" in result.revision_feedback
        assert "Expected output:\n0 3" in result.revision_feedback

    async def test_verify_eval_grid_flags_unparseable_responses(self) -> None:
        server, request = await self._seeded("eval-format")
        result = await server.verify_eval_grid(
            request,
            EvalGridVerificationRequest(response=_text_response("no grid here"), grid_id="test_0"),
        )
        assert not result.format_valid
        assert result.parse_error is not None
        # The floor attempt is still recorded: an unparseable attempt is a
        # real (bad) outcome for that grid, not a non-event.
        assert server._sessions["eval-format"].eval_results["test_0"][0]["format_valid"] == 0.0

    async def test_finalize_aggregates_over_the_grid_sequence(self) -> None:
        server, request = await self._seeded("eval-agg")
        await server.verify_eval_grid(
            request,
            EvalGridVerificationRequest(response=_text_response("<answer>\n0 2\n</answer>"), grid_id="test_0"),
        )
        # test_1 never attempted: contributes the reward floor.
        result = await server.finalize(request, _finalize_request(protocol="eval_sequence", rounds=2))
        assert result.eval_exact_fraction == 0.5
        assert not result.all_solved
        assert result.rounds_used == 2
        assert result.instance_config == {"mask_sample": False}
        floor = -(0.20 + 0.10 + 0.05 + 0.05)
        exact_reward = 1.0 + 0.20 + 0.10 + 0.05 + 0.05  # exact + full gains + color + format
        assert result.reward == pytest.approx((exact_reward + floor) / 2)

    async def test_finalize_pays_the_all_solved_bonus(self) -> None:
        server, request = await self._seeded("eval-bonus")
        for grid_id, answer in (("test_0", "0 2"), ("test_1", "0 3")):
            await server.verify_eval_grid(
                request,
                EvalGridVerificationRequest(
                    response=_text_response(f"<answer>\n{answer}\n</answer>"), grid_id=grid_id
                ),
            )
        result = await server.finalize(request, _finalize_request(protocol="eval_sequence"))
        assert result.all_solved
        assert result.eval_exact_fraction == 1.0
        exact_reward = 1.0 + 0.20 + 0.10 + 0.05 + 0.05
        assert result.reward == pytest.approx(exact_reward + 0.5)

    async def test_finalize_masks_flagged_episodes(self) -> None:
        server, request = await self._seeded("eval-mask")
        result = await server.finalize(request, _finalize_request(protocol="eval_sequence", loss_masked=True))
        assert result.loss_masked
        assert result.instance_config == {"mask_sample": True}


class TestSingleTurnVerify:
    async def test_answer_contract_rows_earn_scored_reward_and_metrics(self) -> None:
        server = _server()
        result = await server.verify(
            ARCAGIVerifyRequest(
                responses_create_params={"input": []},
                response=_text_response("<answer>\n0 1\n</answer>"),
                target=[[0, 1]],
                test_input=[[1, 0]],
                task_id="executor-row",
            )
        )
        assert result.grid_match == 1.0
        assert result.cell_match == 1.0
        assert result.test_exact
        assert result.reward > 1.0  # exact dominates and shaping is additive

    async def test_answer_contract_echo_sits_on_the_floor(self) -> None:
        server = _server()
        result = await server.verify(
            ARCAGIVerifyRequest(
                responses_create_params={"input": []},
                response=_text_response("<answer>\n1 0\n</answer>"),
                target=[[0, 1]],
                test_input=[[1, 0]],
            )
        )
        assert result.copied_input == 1.0
        assert result.reward == pytest.approx(-(0.20 + 0.10 + 0.05 + 0.05))

    async def test_legacy_rows_still_verify_boxed_answers(self) -> None:
        server = _server()
        result = await server.verify(
            ARCAGIVerifyRequest(
                responses_create_params={"input": []},
                response=_text_response("\\boxed{[[0,2]]}"),
                train=[{"input": [[1, 0]], "output": [[0, 1]]}],
                test=[{"input": [[2, 0]], "output": [[0, 2]]}],
                task_id="legacy-row",
            )
        )
        assert result.test_exact
        assert result.reward == 1.0

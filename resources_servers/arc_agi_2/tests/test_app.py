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

from fastapi import Request

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.arc_agi_2.app import (
    ARCAGIResourcesServer,
    ARCAGIResourcesServerConfig,
    ARCAGIRunRequest,
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

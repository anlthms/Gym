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
"""ARC-AGI resource server for single-turn and verifier-guided episodes."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.arc_agi_2.logic import (
    BatchVerification,
    Grid,
    PredictionParseError,
    parse_tagged_grids,
    validate_grid,
    verify_predictions,
)


class ARCAGIResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for the ARC-AGI resources server."""


class ARCGridPair(BaseModel):
    """One public training input/output pair."""

    input: Grid
    output: Grid

    @model_validator(mode="after")
    def validate_grids(self) -> ARCGridPair:
        self.input = validate_grid(self.input, grid_id="train input")
        self.output = validate_grid(self.output, grid_id="train output")
        return self


class ARCTestPair(BaseModel):
    """One test input and its verifier-only output."""

    input: Grid
    output: Grid

    @model_validator(mode="after")
    def validate_grids(self) -> ARCTestPair:
        self.input = validate_grid(self.input, grid_id="test input")
        self.output = validate_grid(self.output, grid_id="test output")
        return self


class ARCAGIRunRequest(BaseRunRequest):
    """A seeded ARC task, supporting legacy one-test and new multi-test rows."""

    train: list[ARCGridPair] = Field(default_factory=list)
    test: list[ARCTestPair] = Field(default_factory=list)
    test_input: Grid | None = None
    expected_output: Grid | None = None
    task_id: str | None = None

    @model_validator(mode="after")
    def validate_task(self) -> ARCAGIRunRequest:
        if not self.train:
            raise ValueError("ARC task must contain at least one training pair")
        if self.test:
            if self.test_input is not None or self.expected_output is not None:
                raise ValueError("use either test or legacy test_input/expected_output fields, not both")
        elif self.test_input is None or self.expected_output is None:
            raise ValueError("ARC task must contain test pairs or legacy test_input/expected_output")
        else:
            self.test_input = validate_grid(self.test_input, grid_id="test_0 input")
            self.expected_output = validate_grid(self.expected_output, grid_id="test_0 output")
        return self

    def normalized_tests(self) -> list[ARCTestPair]:
        if self.test:
            return self.test
        assert self.test_input is not None and self.expected_output is not None
        return [ARCTestPair(input=self.test_input, output=self.expected_output)]


class ARCAGIVerifyRequest(ARCAGIRunRequest, BaseVerifyRequest):
    """Legacy direct-answer verification request."""


class ExecutorVerificationRequest(BaseModel):
    """One executor response to parse against session-owned targets."""

    response: NeMoGymResponse
    record_result: bool = True


class GridVerificationRecord(BaseModel):
    """Serializable form of one deterministic grid comparison."""

    grid_id: str
    predicted: Grid
    correct: Grid
    exact: bool
    shape_match: bool
    cell_accuracy: float
    feedback: str


class ExecutorVerificationResponse(BaseModel):
    """Strict parse and deterministic comparison returned to the agent."""

    format_valid: bool
    parse_error: str | None = None
    all_exact: bool = False
    exact_fraction: float = 0.0
    shape_match_fraction: float = 0.0
    cell_accuracy: float = 0.0
    feedback: str
    predictions: dict[str, Grid] | None = None
    records: list[GridVerificationRecord] = Field(default_factory=list)


class ARCAGIFinalizeRequest(BaseVerifyRequest):
    """Final-proposer-only response plus audit metadata."""

    termination_reason: str
    loss_masked: bool
    trace: dict[str, Any]


class ARCAGIVerifyResponse(BaseVerifyResponse):
    """Episode result consumed by NeMo-RL and offline trace inspection."""

    model_config = ConfigDict(extra="allow")

    task_id: str | None = None
    termination_reason: str | None = None
    loss_masked: bool = False
    train_gate_pass: bool = False
    test_exact: bool = False
    test_cell_accuracy: float = 0.0
    train_exact_fraction: float = 0.0
    train_cell_accuracy: float = 0.0
    trace: dict[str, Any] = Field(default_factory=dict)
    expected_output: Grid | None = None
    predicted_output: Grid | None = None
    extraction_successful: bool = False


@dataclass
class ARCSessionState:
    """Verifier-owned task secrets and latest deterministic results."""

    task_id: str | None
    train_inputs: dict[str, Grid]
    train_targets: dict[str, Grid]
    test_inputs: dict[str, Grid]
    test_targets: dict[str, Grid]
    train_results: list[BatchVerification]
    test_result: BatchVerification | None


def _extract_assistant_text(response: NeMoGymResponse) -> str:
    texts: list[str] = []
    for output in response.output:
        if getattr(output, "type", None) != "message" or getattr(output, "role", None) != "assistant":
            continue
        content = getattr(output, "content", None)
        if isinstance(content, list):
            for part in content:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    texts.append(text)
        elif isinstance(content, str):
            texts.append(content)
    return "\n".join(texts).strip()


def _parse_grid(text: str) -> Grid | None:
    """Parse the legacy ``\\boxed{[[...]]}`` single-answer format."""
    boxed_pattern = r"\\boxed\{(\[\s*\[[\d\s,\[\]]+\]\s*\])\}"
    matches = re.findall(boxed_pattern, text, re.DOTALL)
    if not matches:
        matches = re.findall(r"\[\s*\[[\d\s,\[\]]+\]\s*\]", text, re.DOTALL)
    for match in matches:
        try:
            payload = json.loads(re.sub(r"\s+", "", match))
            return validate_grid(payload, grid_id="answer")
        except (json.JSONDecodeError, PredictionParseError, TypeError):
            continue
    return None


def _verification_response(result: BatchVerification, predictions: dict[str, Grid]) -> ExecutorVerificationResponse:
    return ExecutorVerificationResponse(
        format_valid=True,
        all_exact=result.all_exact,
        exact_fraction=result.exact_fraction,
        shape_match_fraction=result.shape_match_fraction,
        cell_accuracy=result.cell_accuracy,
        feedback=result.feedback,
        predictions=predictions,
        records=[
            {
                "grid_id": record.grid_id,
                "predicted": record.predicted,
                "correct": record.correct,
                "exact": record.exact,
                "shape_match": record.shape_match,
                "cell_accuracy": record.cell_accuracy,
                "feedback": record.feedback,
            }
            for record in result.records
        ],
    )


class ARCAGIResourcesServer(SimpleResourcesServer):
    """Own ARC targets and expose strict train/test verification endpoints."""

    config: ARCAGIResourcesServerConfig

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._sessions: dict[str, ARCSessionState] = {}

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/verify_training")(self.verify_training)
        app.post("/verify_test")(self.verify_test)
        app.post("/finalize")(self.finalize)
        return app

    async def seed_session(self, request: Request, body: ARCAGIRunRequest) -> BaseSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        tests = body.normalized_tests()
        self._sessions[session_id] = ARCSessionState(
            task_id=body.task_id,
            train_inputs={f"train_{index}": pair.input for index, pair in enumerate(body.train)},
            train_targets={f"train_{index}": pair.output for index, pair in enumerate(body.train)},
            test_inputs={f"test_{index}": pair.input for index, pair in enumerate(tests)},
            test_targets={f"test_{index}": pair.output for index, pair in enumerate(tests)},
            train_results=[],
            test_result=None,
        )
        return BaseSeedSessionResponse()

    def _session(self, request: Request) -> ARCSessionState:
        session_id = request.session[SESSION_ID_KEY]
        if session_id not in self._sessions:
            raise HTTPException(status_code=400, detail="ARC session is not initialized; call seed_session first")
        return self._sessions[session_id]

    async def verify_training(
        self,
        request: Request,
        body: ExecutorVerificationRequest,
    ) -> ExecutorVerificationResponse:
        session = self._session(request)
        return self._verify_executor_response(
            body.response,
            tag="predictions",
            targets=session.train_targets,
            store_on=session,
            is_test=False,
            record_result=body.record_result,
        )

    async def verify_test(
        self,
        request: Request,
        body: ExecutorVerificationRequest,
    ) -> ExecutorVerificationResponse:
        session = self._session(request)
        return self._verify_executor_response(
            body.response,
            tag="answers",
            targets=session.test_targets,
            store_on=session,
            is_test=True,
            record_result=body.record_result,
        )

    def _verify_executor_response(
        self,
        response: NeMoGymResponse,
        *,
        tag: str,
        targets: dict[str, Grid],
        store_on: ARCSessionState,
        is_test: bool,
        record_result: bool,
    ) -> ExecutorVerificationResponse:
        text = _extract_assistant_text(response)
        try:
            predictions = parse_tagged_grids(text, tag=tag, expected_ids=list(targets))
        except PredictionParseError as error:
            return ExecutorVerificationResponse(
                format_valid=False,
                parse_error=str(error),
                feedback=f"Executor format failure: {error}",
            )
        result = verify_predictions(predictions=predictions, correct=targets)
        if record_result:
            if is_test:
                store_on.test_result = result
            else:
                store_on.train_results.append(result)
        return _verification_response(result, predictions)

    async def finalize(self, request: Request, body: ARCAGIFinalizeRequest) -> ARCAGIVerifyResponse:
        session = self._session(request)
        train_result = (
            max(
                session.train_results,
                key=lambda result: (result.exact_fraction, result.cell_accuracy),
            )
            if session.train_results
            else None
        )
        test_result = session.test_result
        train_gate_pass = any(result.all_exact for result in session.train_results)
        test_exact = bool(test_result and test_result.all_exact)
        return ARCAGIVerifyResponse(
            **body.model_dump(exclude={"termination_reason", "loss_masked", "trace"}),
            reward=float(test_exact),
            task_id=session.task_id,
            termination_reason=body.termination_reason,
            loss_masked=body.loss_masked,
            train_gate_pass=train_gate_pass,
            test_exact=test_exact,
            test_cell_accuracy=test_result.cell_accuracy if test_result else 0.0,
            train_exact_fraction=train_result.exact_fraction if train_result else 0.0,
            train_cell_accuracy=train_result.cell_accuracy if train_result else 0.0,
            trace=body.trace,
            predicted_output=(
                test_result.records[0].predicted if test_result and len(test_result.records) == 1 else None
            ),
            expected_output=(
                test_result.records[0].correct if test_result and len(test_result.records) == 1 else None
            ),
            extraction_successful=test_result is not None,
        )

    async def verify(self, body: ARCAGIVerifyRequest) -> ARCAGIVerifyResponse:
        """Preserve the existing single-turn ARC verifier for legacy agents."""
        assistant_text = _extract_assistant_text(body.response)
        predicted_grid = _parse_grid(assistant_text)
        extraction_successful = predicted_grid is not None
        expected = body.normalized_tests()[0].output
        exact = extraction_successful and predicted_grid == expected
        return ARCAGIVerifyResponse(
            **body.model_dump(),
            reward=float(exact),
            task_id=body.task_id,
            test_exact=exact,
            expected_output=expected,
            predicted_output=predicted_grid,
            extraction_successful=extraction_successful,
        )


if __name__ == "__main__":
    ARCAGIResourcesServer.run_webserver()

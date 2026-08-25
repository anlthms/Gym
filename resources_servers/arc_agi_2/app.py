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
    Grid,
    PredictionParseError,
    build_eval_feedback_prompt,
    compare_grid,
    validate_grid,
)
from resources_servers.arc_agi_2.scoring import (
    RewardWeights,
    extract_answer_grid,
    reward_floor,
    score_grid,
)


class ARCAGIResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for the ARC-AGI resources server.

    The reward weights mirror NeMo-RL's ``ArcAgiEnvConfig`` defaults: exact
    match dominates the sum of all shaping terms, the similarity terms are
    paid as gain over echoing the input, and ``extraneous_color_weight``
    keeps ``color_weight`` from being maxed for free.
    """

    exact_weight: float = 1.0
    cell_weight: float = 0.20
    edit_weight: float = 0.10
    color_weight: float = 0.05
    extraneous_color_weight: float = 0.05
    shape_weight: float = 0.05
    format_weight: float = 0.05
    # Added once to the aggregated evaluation-sequence reward when every
    # held-out grid was solved exactly.
    all_solved_bonus: float = 0.5

    def reward_weights(self) -> RewardWeights:
        return RewardWeights(
            exact=self.exact_weight,
            cell=self.cell_weight,
            edit=self.edit_weight,
            color=self.color_weight,
            extraneous=self.extraneous_color_weight,
            shape=self.shape_weight,
            format=self.format_weight,
        )


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
    """A seeded ARC task: episode rows, legacy one-test rows, or single-turn rows.

    Single-turn answer-contract rows (executor tasks and real-ARC induction
    tasks) carry ``target`` and ``test_input`` and no training pairs; the
    session they seed holds the one hidden pair and ``verify`` reads the row
    directly.
    """

    train: list[ARCGridPair] = Field(default_factory=list)
    test: list[ARCTestPair] = Field(default_factory=list)
    test_input: Grid | None = None
    expected_output: Grid | None = None
    target: Grid | None = None
    task_id: str | None = None

    def is_single_turn(self) -> bool:
        return not self.train and self.target is not None and self.test_input is not None

    @model_validator(mode="after")
    def validate_task(self) -> ARCAGIRunRequest:
        if self.is_single_turn():
            self.test_input = validate_grid(self.test_input, grid_id="test_0 input")
            self.target = validate_grid(self.target, grid_id="test_0 target")
            return self
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
        assert self.test_input is not None
        if self.is_single_turn():
            assert self.target is not None
            return [ARCTestPair(input=self.test_input, output=self.target)]
        assert self.expected_output is not None
        return [ARCTestPair(input=self.test_input, output=self.expected_output)]


class ARCAGIVerifyRequest(BaseVerifyRequest):
    """Single-turn verification request.

    Two row shapes are accepted: answer-contract rows carrying ``target`` and
    ``test_input`` (single-grid executor tasks and real-ARC induction tasks,
    scored with the shared ``<answer>`` parser and gain-over-echo reward), and
    legacy induction rows carrying ``train``/``test`` grids (scored with the
    boxed-array parser and exact-only reward).
    """

    model_config = ConfigDict(extra="allow")

    train: list[ARCGridPair] = Field(default_factory=list)
    test: list[ARCTestPair] = Field(default_factory=list)
    test_input: Grid | None = None
    expected_output: Grid | None = None
    target: Grid | None = None
    task_id: str | None = None

    def legacy_expected_output(self) -> Grid:
        if self.test:
            return self.test[0].output
        if self.expected_output is None:
            raise ValueError("legacy verification requires test pairs or expected_output")
        return validate_grid(self.expected_output, grid_id="test_0 output")


class EvalGridVerificationRequest(BaseModel):
    """One fresh single-grid executor response for one evaluation grid."""

    response: NeMoGymResponse
    grid_id: str
    record_result: bool = True


class EvalGridVerificationResponse(BaseModel):
    """Strict single-grid parse, comparison, and gain-over-echo score terms.

    ``revision_feedback`` is the complete behavioral-evidence prompt (input,
    executor output, expected output, diff) rendered server-side, so the
    hidden target reaches the proposer only through this deliberate channel
    and the agent never handles raw targets.
    """

    grid_id: str
    format_valid: bool
    parse_error: str | None = None
    exact: bool = False
    predicted: Grid | None = None
    feedback: str
    revision_feedback: str | None = None
    terms: dict[str, float] = Field(default_factory=dict)


class ARCAGIFinalizeRequest(BaseVerifyRequest):
    """Final-proposer-only response plus audit metadata."""

    termination_reason: str
    loss_masked: bool
    protocol: str = "hidden_test"
    trace: dict[str, Any]


class ARCAGIVerifyResponse(BaseVerifyResponse):
    """Episode result consumed by NeMo-RL and offline trace inspection.

    Scalar fields are deliberately top-level: NeMo-RL's gym rollout surfaces
    them as per-agent metrics (``<agent_name>/<field>``), which is how
    ``cell_match`` reaches validation and checkpoint selection.
    """

    model_config = ConfigDict(extra="allow")

    task_id: str | None = None
    termination_reason: str | None = None
    loss_masked: bool = False
    # NeMo-RL reads instance_config.mask_sample to zero the episode's loss
    # multiplier (the sample still counts toward the group baseline).
    instance_config: dict[str, Any] = Field(default_factory=lambda: {"mask_sample": False})
    train_gate_pass: bool = False
    test_exact: bool = False
    test_cell_accuracy: float = 0.0
    train_exact_fraction: float = 0.0
    train_cell_accuracy: float = 0.0
    # Evaluation-sequence episode metrics (eval_sequence protocol only).
    eval_exact_fraction: float = 0.0
    eval_cell_match: float = 0.0
    all_solved: bool = False
    rounds_used: int = 0
    # Single-turn answer-contract metrics (verify endpoint).
    grid_match: float = 0.0
    cell_match: float = 0.0
    format_valid: float = 0.0
    copied_input: float = 0.0
    terms: dict[str, float] = Field(default_factory=dict)
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
    # Per-grid attempt score terms, keyed by grid id. eval_sequence episodes
    # record held-out test grids; hidden_test episodes record demo (train)
    # grids during refinement and test grids for the final answers.
    eval_results: dict[str, list[dict[str, float]]]


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


class ARCAGIResourcesServer(SimpleResourcesServer):
    """Own ARC targets and expose strict train/test verification endpoints."""

    config: ARCAGIResourcesServerConfig

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._sessions: dict[str, ARCSessionState] = {}

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/verify_eval_grid")(self.verify_eval_grid)
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
            eval_results={},
        )
        return BaseSeedSessionResponse()

    def _session(self, request: Request) -> ARCSessionState:
        session_id = request.session[SESSION_ID_KEY]
        if session_id not in self._sessions:
            raise HTTPException(status_code=400, detail="ARC session is not initialized; call seed_session first")
        return self._sessions[session_id]

    async def verify_eval_grid(
        self,
        request: Request,
        body: EvalGridVerificationRequest,
    ) -> EvalGridVerificationResponse:
        """Verify one fresh single-grid ``<answer>`` response for one grid.

        Resolves the grid id in the hidden test pool first, then the demo
        (train) pool -- hidden_test episodes verify demo grids during
        refinement and test grids for the final answers. Scores with the
        shared gain-over-echo terms (the grid's own input is the echo
        baseline) and records the attempt for finalize aggregation.
        """
        session = self._session(request)
        if body.grid_id in session.test_targets:
            target = session.test_targets[body.grid_id]
            echo_input = session.test_inputs[body.grid_id]
        elif body.grid_id in session.train_targets:
            target = session.train_targets[body.grid_id]
            echo_input = session.train_inputs[body.grid_id]
        else:
            raise HTTPException(status_code=400, detail=f"unknown evaluation grid {body.grid_id!r}")
        text = _extract_assistant_text(body.response)
        predicted = extract_answer_grid(text)
        terms = score_grid(predicted, target, echo_input, self.config.reward_weights())
        if body.record_result:
            session.eval_results.setdefault(body.grid_id, []).append(dict(terms))
        if predicted is None:
            return EvalGridVerificationResponse(
                grid_id=body.grid_id,
                format_valid=False,
                parse_error="no parseable grid inside an <answer> block",
                feedback="Executor format failure: no parseable grid inside an <answer> block",
                terms=terms,
            )
        comparison = compare_grid(grid_id=body.grid_id, predicted=predicted, correct=target)
        revision_feedback = None
        if not comparison.exact:
            revision_feedback = build_eval_feedback_prompt(
                grid_id=body.grid_id,
                input_grid=echo_input,
                predicted=predicted,
                expected=target,
                diff_feedback=comparison.feedback,
            )
        return EvalGridVerificationResponse(
            grid_id=body.grid_id,
            format_valid=True,
            exact=comparison.exact,
            predicted=predicted,
            feedback=comparison.feedback,
            revision_feedback=revision_feedback,
            terms=terms,
        )

    async def finalize(self, request: Request, body: ARCAGIFinalizeRequest) -> ARCAGIVerifyResponse:
        session = self._session(request)
        if body.protocol == "eval_sequence":
            return self._finalize_eval_sequence(session, body)
        return self._finalize_hidden_test(session, body)

    def _finalize_hidden_test(self, session: ARCSessionState, body: ARCAGIFinalizeRequest) -> ARCAGIVerifyResponse:
        """Score the final hidden-test answers of a real-ARC episode.

        Each test grid is scored on its LAST recorded attempt -- the answer
        the episode actually committed to (a format retry replaces the failed
        first attempt). Unanswered test grids sit at the reward floor. The
        demo-refinement loop contributes diagnostics only: ``train_gate_pass``
        (every demo grid verified exactly at some point) and
        ``train_exact_fraction``.
        """
        floor = reward_floor(self.config.reward_weights())
        per_grid_rewards: list[float] = []
        per_grid_exact: list[float] = []
        per_grid_cell: list[float] = []
        per_grid_format: list[float] = []
        for grid_id in session.test_targets:
            attempts = session.eval_results.get(grid_id, [])
            if attempts:
                final = attempts[-1]
                per_grid_rewards.append(final["reward"])
                per_grid_exact.append(final["grid_match"])
                per_grid_cell.append(final["cell_match"])
                per_grid_format.append(final["format_valid"])
            else:
                per_grid_rewards.append(floor)
                per_grid_exact.append(0.0)
                per_grid_cell.append(0.0)
                per_grid_format.append(0.0)
        count = len(per_grid_rewards)
        demo_solved = [
            float(any(attempt["grid_match"] for attempt in session.eval_results.get(grid_id, [])))
            for grid_id in session.train_targets
        ]
        grid_match = sum(per_grid_exact) / count if count else 0.0
        return ARCAGIVerifyResponse(
            **body.model_dump(exclude={"termination_reason", "loss_masked", "protocol", "trace"}),
            reward=sum(per_grid_rewards) / count if count else floor,
            task_id=session.task_id,
            termination_reason=body.termination_reason,
            loss_masked=body.loss_masked,
            instance_config={"mask_sample": body.loss_masked},
            train_gate_pass=bool(demo_solved) and all(demo_solved),
            test_exact=count > 0 and grid_match == 1.0,
            grid_match=grid_match,
            cell_match=sum(per_grid_cell) / count if count else 0.0,
            format_valid=sum(per_grid_format) / count if count else 0.0,
            train_exact_fraction=(sum(demo_solved) / len(demo_solved)) if demo_solved else 0.0,
            rounds_used=len(body.trace.get("rounds", [])),
            trace=body.trace,
        )

    def _finalize_eval_sequence(self, session: ARCSessionState, body: ARCAGIFinalizeRequest) -> ARCAGIVerifyResponse:
        """Aggregate the evaluation-grid sequence into one episode reward.

        Each grid contributes its best attempt's gain-over-echo grid score;
        a grid the episode never reached sits at the reward floor, so ending
        early is never better than attempting the remaining grids. A rule
        that generalizes across several grids therefore outscores one that
        happens to solve a single grid, and solving everything earns the
        configured bonus on top.
        """
        floor = reward_floor(self.config.reward_weights())
        per_grid_rewards: list[float] = []
        per_grid_cell: list[float] = []
        solved = 0
        for grid_id in session.test_targets:
            attempts = session.eval_results.get(grid_id, [])
            if attempts:
                per_grid_rewards.append(max(attempt["reward"] for attempt in attempts))
                per_grid_cell.append(max(attempt["cell_match"] for attempt in attempts))
                solved += int(any(attempt["grid_match"] for attempt in attempts))
            else:
                per_grid_rewards.append(floor)
                per_grid_cell.append(0.0)
        count = len(per_grid_rewards)
        all_solved = count > 0 and solved == count
        reward = sum(per_grid_rewards) / count if count else floor
        if all_solved:
            reward += self.config.all_solved_bonus
        return ARCAGIVerifyResponse(
            **body.model_dump(exclude={"termination_reason", "loss_masked", "protocol", "trace"}),
            reward=reward,
            task_id=session.task_id,
            termination_reason=body.termination_reason,
            loss_masked=body.loss_masked,
            instance_config={"mask_sample": body.loss_masked},
            eval_exact_fraction=solved / count if count else 0.0,
            eval_cell_match=sum(per_grid_cell) / count if count else 0.0,
            all_solved=all_solved,
            rounds_used=len(body.trace.get("rounds", [])),
            trace=body.trace,
        )

    async def verify(self, body: ARCAGIVerifyRequest) -> ARCAGIVerifyResponse:
        """Verify one single-turn response.

        Rows carrying ``target`` and ``test_input`` (executor tasks and
        real-ARC induction tasks) are scored with the shared ``<answer>``
        parser and gain-over-echo reward; ``grid_match``/``cell_match`` are
        top-level so NeMo-RL surfaces them as per-agent validation metrics.
        Rows without them fall back to the legacy boxed-array verifier.
        """
        assistant_text = _extract_assistant_text(body.response)
        if body.target is not None and body.test_input is not None:
            try:
                target = validate_grid(body.target, grid_id="target")
                test_input = validate_grid(body.test_input, grid_id="test_input")
            except PredictionParseError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
            predicted = extract_answer_grid(assistant_text)
            terms = score_grid(predicted, target, test_input, self.config.reward_weights())
            return ARCAGIVerifyResponse(
                **body.model_dump(exclude={"task_id", "expected_output"}),
                reward=terms["reward"],
                task_id=body.task_id,
                test_exact=bool(terms["grid_match"]),
                grid_match=terms["grid_match"],
                cell_match=terms["cell_match"],
                format_valid=terms["format_valid"],
                copied_input=terms["copied_input"],
                terms=terms,
                expected_output=target,
                predicted_output=predicted,
                extraction_successful=predicted is not None,
            )

        predicted_grid = _parse_grid(assistant_text)
        extraction_successful = predicted_grid is not None
        expected = body.legacy_expected_output()
        exact = extraction_successful and predicted_grid == expected
        return ARCAGIVerifyResponse(
            **body.model_dump(exclude={"task_id", "expected_output"}),
            reward=float(exact),
            task_id=body.task_id,
            test_exact=exact,
            grid_match=float(exact),
            expected_output=expected,
            predicted_output=predicted_grid,
            extraction_successful=extraction_successful,
        )


if __name__ == "__main__":
    ARCAGIResourcesServer.run_webserver()

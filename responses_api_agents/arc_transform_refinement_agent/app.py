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
"""Dual-history ARC transform refinement agent."""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

from fastapi import Request, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from resources_servers.arc_agi_2.logic import (
    ContextBudget,
    EpisodePhase,
    EpisodeState,
    Grid,
    TerminationReason,
    TransformDescriptionParseError,
    assert_model_request_safe,
    build_nvarc_proposer_prompt,
    build_single_executor_prompt,
    build_single_grid_format_retry_prompt,
    conservative_text_token_bound,
    parse_canonical_rule,
)


PROPOSER_INSTRUCTIONS = (
    "Act only as an ARC transformation-rule proposer. Return the requested "
    "transformation-description artifact; do not execute the test grid."
)
EXECUTOR_INSTRUCTIONS = (
    "Act only as a literal ARC transformation executor. Apply the supplied "
    "description without revising or replacing it, and return only the requested grids."
)


class ArcTransformRefinementAgentConfig(BaseResponsesAPIAgentConfig):
    """User-facing controls for the dual-history ARC protocol."""

    resources_server: ResourcesServerRef
    proposer_model_server: ModelServerRef
    executor_model_server: ModelServerRef
    # Both protocols share the same single-grid <answer> executor contract
    # (byte-parity with native executor training rows) and the same canonical
    # 4-section proposer rule.
    # hidden_test: the real-ARC protocol -- refine the rule against the
    # puzzle's own demo pairs (their outputs are public at inference), then
    # answer every hidden test grid once with the current rule.
    # eval_sequence: the NVARC co-training protocol -- demos shown, held-out
    # evaluation grids solved one at a time, advance-on-solve,
    # revise-on-fail, reward aggregated server-side over the grid sequence.
    # candidate_select: sample num_candidates independent rules from the same
    # demo prompt, score each on EVERY demo grid server-side, let the server
    # pick the winner (exact count, then cell match, then format validity,
    # then emission order), and only then answer each test grid once.
    # A task row may override this default per episode via its own
    # ``protocol`` field.
    protocol: Literal["hidden_test", "eval_sequence", "candidate_select"] = "hidden_test"
    model_context_limit: int = 32_768
    reserved_proposer_output_tokens: int = 4_096
    chat_template_margin: int = 512
    proposer_max_output_tokens: int = 4_096
    executor_max_output_tokens: int = 4_096
    max_rounds: int = 32
    # candidate_select only: independent proposer samples per episode.
    num_candidates: int = 8

    @model_validator(mode="after")
    def validate_limits(self) -> ArcTransformRefinementAgentConfig:
        ContextBudget(
            model_context_limit=self.model_context_limit,
            reserved_proposer_output_tokens=self.reserved_proposer_output_tokens,
            chat_template_margin=self.chat_template_margin,
        )
        if self.proposer_max_output_tokens <= 0 or self.executor_max_output_tokens <= 0:
            raise ValueError("proposer and executor max output tokens must be positive")
        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        if self.num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        return self


class ArcPair(BaseModel):
    """Public ARC training pair."""

    input: Grid
    output: Grid


class ArcTestPair(BaseModel):
    """ARC test pair; output is verifier-only and never read by prompt builders."""

    input: Grid
    output: Grid


class ArcTransformRunRequest(BaseRunRequest):
    """Task row consumed by the refinement agent."""

    model_config = ConfigDict(extra="allow")

    train: list[ArcPair]
    test: list[ArcTestPair] = Field(default_factory=list)
    test_input: Grid | None = None
    expected_output: Grid | None = None
    task_id: str | None = None
    # Per-row protocol override, so one agent instance can train on
    # eval_sequence episodes and validate on hidden_test episodes.
    protocol: Literal["hidden_test", "eval_sequence", "candidate_select"] | None = None
    # Per-row context-budget override: validation loop rows may use the full
    # inference-engine window, while training rows stay at the config default
    # sized to the trainable pack.
    model_context_limit: int | None = None
    # Per-row candidate-count override for candidate_select episodes.
    num_candidates: int | None = None

    @model_validator(mode="after")
    def validate_task(self) -> ArcTransformRunRequest:
        if self.model_context_limit is not None and self.model_context_limit <= 0:
            raise ValueError("model_context_limit override must be positive")
        if self.num_candidates is not None and self.num_candidates <= 0:
            raise ValueError("num_candidates override must be positive")
        if not self.train:
            raise ValueError("ARC task must contain at least one training pair")
        if self.test:
            if self.test_input is not None or self.expected_output is not None:
                raise ValueError("use either test or legacy test_input/expected_output fields, not both")
        elif self.test_input is None or self.expected_output is None:
            raise ValueError("ARC task must contain test pairs or legacy test_input/expected_output")
        return self

    def public_train_pairs(self) -> list[dict[str, Grid]]:
        return [{"input": pair.input, "output": pair.output} for pair in self.train]

    def public_test_inputs(self) -> dict[str, Grid]:
        if self.test:
            return {f"test_{index}": pair.input for index, pair in enumerate(self.test)}
        assert self.test_input is not None
        return {"test_0": self.test_input}


class ArcTransformVerifyResponse(BaseVerifyResponse):
    """Verifier response with episode fields passed through."""

    model_config = ConfigDict(extra="allow")


@dataclass
class RoleCall:
    """One fully inspectable model call."""

    role: str
    request: dict[str, Any]
    response: dict[str, Any]
    prompt_tokens: int
    output_tokens: int
    latency_seconds: float


@dataclass
class RoundTrace:
    """All proposer, executor, and verifier artifacts for one revision round."""

    round_index: int
    proposer: RoleCall
    transform_description: str | None
    executor_calls: list[RoleCall]
    training_verifications: list[dict[str, Any]]
    test_verification: dict[str, Any] | None
    feedback: str | None


def _message(content: str) -> dict[str, str]:
    return {"role": "user", "content": content, "type": "message"}


def _response_text(response: NeMoGymResponse) -> str:
    texts: list[str] = []
    for output in response.output:
        if getattr(output, "type", None) != "message" or getattr(output, "role", None) != "assistant":
            continue
        for part in getattr(output, "content", []):
            text = getattr(part, "text", None)
            if isinstance(text, str):
                texts.append(text)
    return "\n".join(texts).strip()


def _response_token_counts(response: NeMoGymResponse) -> tuple[int, int]:
    prompt_tokens = 0
    output_tokens = 0
    for item in response.output:
        item_prompt_ids = getattr(item, "prompt_token_ids", None)
        item_generation_ids = getattr(item, "generation_token_ids", None)
        if item_prompt_ids is not None:
            prompt_tokens = max(prompt_tokens, len(item_prompt_ids))
        if item_generation_ids is not None:
            output_tokens += len(item_generation_ids)
    if response.usage is not None:
        prompt_tokens = max(prompt_tokens, response.usage.input_tokens)
        output_tokens = max(output_tokens, response.usage.output_tokens)
    return prompt_tokens, output_tokens


def _is_context_window_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "context length",
            "context window",
            "maximum input length",
            "max_model_len",
            "too many tokens",
        )
    )


class ArcTransformRefinementAgent(SimpleResponsesAPIAgent):
    """Orchestrate persistent proposer and disposable executor histories."""

    config: ArcTransformRefinementAgentConfig

    async def _call_model(
        self,
        *,
        request: Request,
        server: ModelServerRef,
        params: dict[str, Any],
        cookies: dict[str, str] | None,
        run_body: ArcTransformRunRequest | None,
    ) -> tuple[NeMoGymResponse, dict[str, str]]:
        assert_model_request_safe(params)
        url_path = (
            self.url_path_for_run("/v1/responses", run_body)
            if run_body is not None
            else self.url_path_for_request("/v1/responses", request)
        )
        model_response = await self.server_client.post(
            server_name=server.name,
            url_path=url_path,
            json=params,
            cookies=cookies,
        )
        await raise_for_status(model_response)
        parsed = NeMoGymResponse.model_validate(await get_response_json(model_response))
        return parsed, dict(model_response.cookies)

    async def _call_resource(
        self,
        *,
        url_path: str,
        payload: dict[str, Any],
        cookies: dict[str, str],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        resource_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path=url_path,
            json=payload,
            cookies=cookies,
        )
        await raise_for_status(resource_response)
        return await get_response_json(resource_response), dict(resource_response.cookies)

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        """Forward direct Responses API calls to the proposer role."""
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]
        parsed, cookies = await self._call_model(
            request=request,
            server=self.config.proposer_model_server,
            params=body.model_dump(exclude_none=True),
            cookies=dict(request.cookies),
            run_body=None,
        )
        for key, value in cookies.items():
            response.set_cookie(key, value)
        return parsed

    def _context_budget(self, body: ArcTransformRunRequest) -> ContextBudget:
        """Build the proposer context budget, honoring a per-row limit override."""
        return ContextBudget(
            model_context_limit=body.model_context_limit or self.config.model_context_limit,
            reserved_proposer_output_tokens=self.config.reserved_proposer_output_tokens,
            chat_template_margin=self.config.chat_template_margin,
        )

    def _params(
        self,
        body: ArcTransformRunRequest,
        *,
        input_items: list[dict[str, Any]],
        max_tokens: int,
        instructions: str | None,
    ) -> dict[str, Any]:
        params = body.responses_create_params.model_dump(exclude_none=True)
        params["input"] = copy.deepcopy(input_items)
        # None omits the system message entirely: the single-grid executor
        # contract is one user message, matching executor training rows.
        if instructions is None:
            params.pop("instructions", None)
        else:
            params["instructions"] = instructions
        params["max_output_tokens"] = max_tokens
        params["tools"] = []
        return params

    async def _recorded_model_call(
        self,
        *,
        request: Request,
        role: str,
        server: ModelServerRef,
        params: dict[str, Any],
        cookies: dict[str, str] | None,
        run_body: ArcTransformRunRequest,
    ) -> tuple[NeMoGymResponse, dict[str, str], RoleCall]:
        started = time.perf_counter()
        response, response_cookies = await self._call_model(
            request=request,
            server=server,
            params=params,
            cookies=cookies,
            run_body=run_body,
        )
        prompt_tokens, output_tokens = _response_token_counts(response)
        record = RoleCall(
            role=role,
            request=params,
            response=response.model_dump(),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            latency_seconds=time.perf_counter() - started,
        )
        return response, response_cookies, record

    async def _finalize(
        self,
        *,
        body: ArcTransformRunRequest,
        cookies: dict[str, str],
        last_proposer_response: NeMoGymResponse,
        state: EpisodeState,
        rounds: list[RoundTrace],
        loss_masked: bool,
        protocol: str,
        proposer_format_failure: bool = False,
        selected_candidate_id: str | None = None,
        policy_round: RoundTrace | None = None,
        extra_trace: dict[str, Any] | None = None,
    ) -> ArcTransformVerifyResponse:
        assert state.termination_reason is not None
        trace: dict[str, Any] = {
            "task_id": body.task_id,
            "rounds": [asdict(round_trace) for round_trace in rounds],
            "termination_reason": state.termination_reason.value,
            "loss_masked": loss_masked,
            "proposer_format_failure": proposer_format_failure,
            "policy_loss": {
                "role": "proposer",
                "round_index": (policy_round or rounds[-1]).round_index,
                "output": last_proposer_response.model_dump(),
            },
        }
        if extra_trace:
            trace.update(extra_trace)
        payload = {
            "responses_create_params": body.responses_create_params.model_dump(),
            "response": last_proposer_response.model_dump(),
            "termination_reason": state.termination_reason.value,
            "loss_masked": loss_masked,
            "proposer_format_failure": proposer_format_failure,
            "protocol": protocol,
            "selected_candidate_id": selected_candidate_id,
            "trace": trace,
        }
        result, _ = await self._call_resource(url_path="/finalize", payload=payload, cookies=cookies)
        return ArcTransformVerifyResponse.model_validate(result)

    async def run(self, request: Request, body: ArcTransformRunRequest) -> ArcTransformVerifyResponse:
        """Run one episode under the row's protocol (falling back to config)."""
        protocol = body.protocol or self.config.protocol
        if protocol == "eval_sequence":
            return await self._run_eval_sequence(request, body)
        if protocol == "candidate_select":
            return await self._run_candidate_select(request, body)
        return await self._run_hidden_test(request, body)

    async def _run_hidden_test(self, request: Request, body: ArcTransformRunRequest) -> ArcTransformVerifyResponse:
        """Run the real-ARC protocol: demo-refinement loop, then the hidden test.

        The proposer sees the demonstration pairs and induces a canonical
        4-section rule; fresh single-grid executor sessions apply it to the
        demo inputs in order and the server verifies against the (public)
        demo outputs, returning behavioral evidence for a revision on a miss.
        Whatever ends the loop -- all demos verified, context budget, or the
        round cap -- every demo grid whose last attempt used an older rule is
        swept once with the current rule (so the recorded demo gate describes
        the rule that actually answers the test), then that rule is applied
        once to every hidden test grid and the server scores those final
        answers.
        """
        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=request.cookies,
        )
        await raise_for_status(seed_response)
        resource_cookies = dict(seed_response.cookies)

        demo_inputs = {f"train_{index}": pair.input for index, pair in enumerate(body.train)}
        demo_ids = list(demo_inputs)
        test_inputs = body.public_test_inputs()
        proposer_history: list[dict[str, Any]] = [
            _message(build_nvarc_proposer_prompt(demo_pairs=body.public_train_pairs()))
        ]
        budget = self._context_budget(body)
        state = EpisodeState.initial()
        rounds: list[RoundTrace] = []
        demo_index = 0
        description: str | None = None
        last_proposer_response: NeMoGymResponse | None = None
        loop_end: TerminationReason | None = None
        proposer_format_failure = False
        # Demo grids whose recorded last attempt used the current rule; reset
        # on every accepted revision so the pre-test sweep re-attempts the
        # rest with the final rule.
        attempted_with_current_rule: set[str] = set()

        while loop_end is None:
            proposer_params = self._params(
                body,
                input_items=proposer_history,
                max_tokens=self.config.proposer_max_output_tokens,
                instructions=PROPOSER_INSTRUCTIONS,
            )
            proposer_response, _, proposer_call = await self._recorded_model_call(
                request=request,
                role="proposer",
                server=self.config.proposer_model_server,
                params=proposer_params,
                cookies=None,
                run_body=body,
            )
            last_proposer_response = proposer_response
            round_trace = RoundTrace(
                round_index=state.round_index,
                proposer=proposer_call,
                transform_description=None,
                executor_calls=[],
                training_verifications=[],
                test_verification=None,
                feedback=None,
            )
            rounds.append(round_trace)

            try:
                description = parse_canonical_rule(_response_text(proposer_response))
            except TransformDescriptionParseError:
                # Proposer-caused failure: reward floor with loss ON (the
                # trained final turn IS the unparseable/runaway one), never
                # masked -- masking is reserved for executor-caused failures.
                proposer_format_failure = True
                if description is None:
                    state = state.terminate(TerminationReason.AGENT_ERROR)
                    return await self._finalize(
                        body=body,
                        cookies=resource_cookies,
                        last_proposer_response=last_proposer_response,
                        state=state,
                        rounds=rounds,
                        loss_masked=False,
                        protocol="hidden_test",
                        proposer_format_failure=True,
                    )
                # A prior valid rule exists: still answer the hidden test.
                loop_end = TerminationReason.AGENT_ERROR
                state = state.demo_loop_exhausted()
                break
            round_trace.transform_description = description
            state = state.description_generated()
            attempted_with_current_rule = set()

            failed_verification: dict[str, Any] | None = None
            while demo_index < len(demo_ids):
                grid_id = demo_ids[demo_index]
                verification, state, resource_cookies = await self._run_single_eval_executor(
                    request=request,
                    body=body,
                    description=description,
                    grid_id=grid_id,
                    input_grid=demo_inputs[grid_id],
                    state=state,
                    round_trace=round_trace,
                    resource_cookies=resource_cookies,
                )
                if verification is None:
                    return await self._finalize(
                        body=body,
                        cookies=resource_cookies,
                        last_proposer_response=last_proposer_response,
                        state=state,
                        rounds=rounds,
                        loss_masked=True,
                        protocol="hidden_test",
                    )
                attempted_with_current_rule.add(grid_id)
                if verification["exact"]:
                    demo_index += 1
                    if demo_index == len(demo_ids):
                        loop_end = TerminationReason.TRAIN_VERIFIED
                        state = state.demos_verified()
                        break
                    state = state.eval_grid_solved(all_solved=False)
                    continue
                failed_verification = verification
                break
            if loop_end is not None:
                break

            assert failed_verification is not None  # a revision follows only a miss
            state = state.eval_grid_failed()
            feedback = failed_verification["revision_feedback"]
            round_trace.feedback = feedback
            current_tokens = proposer_call.prompt_tokens + proposer_call.output_tokens
            if not budget.permits_revision(
                current_proposer_tokens=current_tokens,
                next_feedback_tokens=conservative_text_token_bound(feedback),
            ):
                loop_end = TerminationReason.CONTEXT_EXHAUSTED
                state = state.demo_loop_exhausted()
                break
            if state.round_index >= self.config.max_rounds:
                loop_end = TerminationReason.EMERGENCY_ROUND_CAP
                state = state.demo_loop_exhausted()
                break
            proposer_history.extend(item.model_dump() for item in proposer_response.output)
            proposer_history.append(_message(feedback))

        assert description is not None and last_proposer_response is not None
        # Final all-demo sweep: the demo gate must describe the rule that is
        # about to answer the test, so every demo grid last attempted with an
        # older rule is re-answered once with the final rule.
        for grid_id in demo_ids:
            if grid_id in attempted_with_current_rule:
                continue
            verification, state, resource_cookies = await self._run_single_eval_executor(
                request=request,
                body=body,
                description=description,
                grid_id=grid_id,
                input_grid=demo_inputs[grid_id],
                state=state,
                round_trace=rounds[-1],
                resource_cookies=resource_cookies,
                include_revision_feedback=False,
            )
            if verification is None:
                return await self._finalize(
                    body=body,
                    cookies=resource_cookies,
                    last_proposer_response=last_proposer_response,
                    state=state,
                    rounds=rounds,
                    loss_masked=True,
                    protocol="hidden_test",
                )
        for grid_id, input_grid in test_inputs.items():
            # No revision ever follows a hidden-test answer, so the server is
            # told to omit the target-bearing evidence prompt: hidden targets
            # must not reach the recorded trace.
            verification, state, resource_cookies = await self._run_single_eval_executor(
                request=request,
                body=body,
                description=description,
                grid_id=grid_id,
                input_grid=input_grid,
                state=state,
                round_trace=rounds[-1],
                resource_cookies=resource_cookies,
                include_revision_feedback=False,
            )
            if verification is None:
                return await self._finalize(
                    body=body,
                    cookies=resource_cookies,
                    last_proposer_response=last_proposer_response,
                    state=state,
                    rounds=rounds,
                    loss_masked=True,
                    protocol="hidden_test",
                )
        state = state.terminate(loop_end)
        return await self._finalize(
            body=body,
            cookies=resource_cookies,
            last_proposer_response=last_proposer_response,
            state=state,
            rounds=rounds,
            loss_masked=False,
            protocol="hidden_test",
            proposer_format_failure=proposer_format_failure,
        )

    async def _run_single_eval_executor(
        self,
        *,
        request: Request,
        body: ArcTransformRunRequest,
        description: str,
        grid_id: str,
        input_grid: Grid,
        state: EpisodeState,
        round_trace: RoundTrace,
        resource_cookies: dict[str, str],
        include_revision_feedback: bool = True,
        candidate_id: str | None = None,
    ) -> tuple[dict[str, Any] | None, EpisodeState, dict[str, str]]:
        """Apply one rule to one evaluation grid in a fresh executor session.

        One format-only retry; returns ``(verification, state, cookies)`` with
        ``verification is None`` when a terminal guard fired (the returned
        state is then TERMINATED and the episode must be loss-masked).
        ``include_revision_feedback=False`` marks a verification that no
        revision will ever consume (final test answers, candidate scoring):
        the server then omits the target-bearing evidence prompt, keeping
        targets out of the recorded trace.
        """
        executor_history = [_message(build_single_executor_prompt(description=description, input_grid=input_grid))]
        while True:
            executor_params = self._params(
                body,
                input_items=executor_history,
                max_tokens=self.config.executor_max_output_tokens,
                instructions=None,
            )
            try:
                executor_response, _, executor_call = await self._recorded_model_call(
                    request=request,
                    role="executor_eval",
                    server=self.config.executor_model_server,
                    params=executor_params,
                    cookies=None,
                    run_body=body,
                )
            except Exception as error:
                if not _is_context_window_error(error):
                    raise
                return None, state.terminate(TerminationReason.EXECUTOR_CONTEXT_EXHAUSTED), resource_cookies
            round_trace.executor_calls.append(executor_call)
            verification, resource_cookies = await self._call_resource(
                url_path="/verify_eval_grid",
                payload={
                    "response": executor_response.model_dump(),
                    "grid_id": grid_id,
                    "include_revision_feedback": include_revision_feedback,
                    "candidate_id": candidate_id,
                },
                cookies=resource_cookies,
            )
            round_trace.training_verifications.append(verification)
            if verification["format_valid"]:
                return verification, state, resource_cookies
            state = state.executor_format_failed()
            if state.phase is EpisodePhase.TERMINATED:
                return None, state, resource_cookies
            executor_history.extend(item.model_dump() for item in executor_response.output)
            executor_history.append(_message(build_single_grid_format_retry_prompt()))

    async def _run_eval_sequence(self, request: Request, body: ArcTransformRunRequest) -> ArcTransformVerifyResponse:
        """Run the NVARC evaluation-grid sequence: advance-on-solve, revise-on-fail.

        The proposer sees only the demonstration pairs and induces a canonical
        4-section rule; fresh single-grid executor sessions apply it to the
        held-out evaluation grids in order. An exact solve advances to the
        next grid with the same rule; a miss returns the server-rendered
        behavioral evidence (input, prediction, expected, diff) for a
        revision.

        Final-rule credit: only the FINAL proposer turn is trained, so before
        finalizing, the current rule is re-applied once to every grid whose
        recorded last attempt used an older rule (including grids the
        sequence never reached), and the server scores each grid's LAST
        attempt. Without this sweep, a degraded final revision inherits
        best-attempt rewards that earlier rules earned under advance-on-solve
        -- credit misassignment that can reinforce revision churn (measured
        as the declining loop metric of the first two co-training runs).
        """
        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=request.cookies,
        )
        await raise_for_status(seed_response)
        resource_cookies = dict(seed_response.cookies)

        eval_inputs = body.public_test_inputs()
        eval_ids = list(eval_inputs)
        proposer_history: list[dict[str, Any]] = [
            _message(build_nvarc_proposer_prompt(demo_pairs=body.public_train_pairs()))
        ]
        budget = self._context_budget(body)
        state = EpisodeState.initial()
        rounds: list[RoundTrace] = []
        grid_index = 0
        description: str | None = None
        # Grids whose recorded last attempt used the current rule; reset on
        # every accepted revision so the final sweep re-attempts the rest.
        attempted_with_current_rule: set[str] = set()
        loop_end: TerminationReason | None = None
        proposer_response: NeMoGymResponse | None = None

        while loop_end is None:
            proposer_params = self._params(
                body,
                input_items=proposer_history,
                max_tokens=self.config.proposer_max_output_tokens,
                instructions=PROPOSER_INSTRUCTIONS,
            )
            proposer_response, _, proposer_call = await self._recorded_model_call(
                request=request,
                role="proposer",
                server=self.config.proposer_model_server,
                params=proposer_params,
                cookies=None,
                run_body=body,
            )
            round_trace = RoundTrace(
                round_index=state.round_index,
                proposer=proposer_call,
                transform_description=None,
                executor_calls=[],
                training_verifications=[],
                test_verification=None,
                feedback=None,
            )
            rounds.append(round_trace)

            try:
                description = parse_canonical_rule(_response_text(proposer_response))
            except TransformDescriptionParseError:
                # Proposer-caused failure: reward floor with loss ON, so
                # runaway thinking finally receives negative gradient.
                state = state.terminate(TerminationReason.AGENT_ERROR)
                return await self._finalize(
                    body=body,
                    cookies=resource_cookies,
                    last_proposer_response=proposer_response,
                    state=state,
                    rounds=rounds,
                    loss_masked=False,
                    protocol="eval_sequence",
                    proposer_format_failure=True,
                )
            round_trace.transform_description = description
            state = state.description_generated()
            attempted_with_current_rule = set()

            failed_verification: dict[str, Any] | None = None
            while grid_index < len(eval_ids):
                grid_id = eval_ids[grid_index]
                verification, state, resource_cookies = await self._run_single_eval_executor(
                    request=request,
                    body=body,
                    description=description,
                    grid_id=grid_id,
                    input_grid=eval_inputs[grid_id],
                    state=state,
                    round_trace=round_trace,
                    resource_cookies=resource_cookies,
                )
                if verification is None:
                    return await self._finalize(
                        body=body,
                        cookies=resource_cookies,
                        last_proposer_response=proposer_response,
                        state=state,
                        rounds=rounds,
                        loss_masked=True,
                        protocol="eval_sequence",
                    )
                attempted_with_current_rule.add(grid_id)
                if verification["exact"]:
                    grid_index += 1
                    if grid_index == len(eval_ids):
                        loop_end = TerminationReason.ALL_SOLVED
                        # Enter the final-answer phase for the sweep.
                        state = state.demos_verified()
                        break
                    state = state.eval_grid_solved(all_solved=False)
                    continue
                failed_verification = verification
                break
            if loop_end is not None:
                break

            assert failed_verification is not None  # a revision follows only a miss
            state = state.eval_grid_failed()
            feedback = failed_verification["revision_feedback"]
            round_trace.feedback = feedback
            current_tokens = proposer_call.prompt_tokens + proposer_call.output_tokens
            if not budget.permits_revision(
                current_proposer_tokens=current_tokens,
                next_feedback_tokens=conservative_text_token_bound(feedback),
            ):
                loop_end = TerminationReason.CONTEXT_EXHAUSTED
                state = state.demo_loop_exhausted()
                break
            if state.round_index >= self.config.max_rounds:
                loop_end = TerminationReason.EMERGENCY_ROUND_CAP
                state = state.demo_loop_exhausted()
                break
            proposer_history.extend(item.model_dump() for item in proposer_response.output)
            proposer_history.append(_message(feedback))

        # Final sweep: the trained turn is scored on what ITS rule does, so
        # every grid last attempted with an older rule (or never reached) is
        # answered once more with the final rule before finalizing.
        assert description is not None and proposer_response is not None
        for grid_id in eval_ids:
            if grid_id in attempted_with_current_rule:
                continue
            verification, state, resource_cookies = await self._run_single_eval_executor(
                request=request,
                body=body,
                description=description,
                grid_id=grid_id,
                input_grid=eval_inputs[grid_id],
                state=state,
                round_trace=rounds[-1],
                resource_cookies=resource_cookies,
                # No revision follows the final sweep; keep the evidence
                # prompt (and its targets) out of the recorded trace.
                include_revision_feedback=False,
            )
            if verification is None:
                return await self._finalize(
                    body=body,
                    cookies=resource_cookies,
                    last_proposer_response=proposer_response,
                    state=state,
                    rounds=rounds,
                    loss_masked=True,
                    protocol="eval_sequence",
                )
        state = state.terminate(loop_end)
        return await self._finalize(
            body=body,
            cookies=resource_cookies,
            last_proposer_response=proposer_response,
            state=state,
            rounds=rounds,
            loss_masked=False,
            protocol="eval_sequence",
        )

    async def _score_candidate_demo(
        self,
        *,
        request: Request,
        body: ArcTransformRunRequest,
        candidate_id: str,
        description: str,
        grid_id: str,
        input_grid: Grid,
        round_trace: RoundTrace,
        resource_cookies: dict[str, str],
    ) -> dict[str, str]:
        """Apply one candidate rule to one demo grid, tolerating its failure.

        Unlike the episode executor, a candidate's double format failure or
        context overflow must not end the episode: the unrecorded (or
        format-failed) attempt simply scores as a miss in the candidate's
        server-side namespace, and selection moves on.
        """
        executor_history = [_message(build_single_executor_prompt(description=description, input_grid=input_grid))]
        for attempt in range(2):
            executor_params = self._params(
                body,
                input_items=executor_history,
                max_tokens=self.config.executor_max_output_tokens,
                instructions=None,
            )
            try:
                executor_response, _, executor_call = await self._recorded_model_call(
                    request=request,
                    role="executor_candidate",
                    server=self.config.executor_model_server,
                    params=executor_params,
                    cookies=None,
                    run_body=body,
                )
            except Exception as error:
                if not _is_context_window_error(error):
                    raise
                return resource_cookies
            round_trace.executor_calls.append(executor_call)
            verification, resource_cookies = await self._call_resource(
                url_path="/verify_eval_grid",
                payload={
                    "response": executor_response.model_dump(),
                    "grid_id": grid_id,
                    "candidate_id": candidate_id,
                    "include_revision_feedback": False,
                },
                cookies=resource_cookies,
            )
            round_trace.training_verifications.append(verification)
            if verification["format_valid"]:
                break
            if attempt == 0:
                executor_history.extend(item.model_dump() for item in executor_response.output)
                executor_history.append(_message(build_single_grid_format_retry_prompt()))
        return resource_cookies

    async def _run_candidate_select(
        self, request: Request, body: ArcTransformRunRequest
    ) -> ArcTransformVerifyResponse:
        """Sample K rules, select on demos server-side, touch the test once.

        Every candidate is an independent proposer sample from the same
        demo-only prompt. Each parseable rule is applied to EVERY demo grid
        and recorded under its own server-side namespace; ``/select_candidate``
        ranks those namespaces (demo exact count, then mean cell match, then
        mean format validity, then emission order) without ever seeing test
        information. Only the winning rule is then executed on each hidden
        test grid -- once -- and finalize re-verifies that the committed
        selection matches the demo-only ranking.
        """
        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=request.cookies,
        )
        await raise_for_status(seed_response)
        resource_cookies = dict(seed_response.cookies)

        num_candidates = body.num_candidates or self.config.num_candidates
        demo_inputs = {f"train_{index}": pair.input for index, pair in enumerate(body.train)}
        test_inputs = body.public_test_inputs()
        base_prompt = build_nvarc_proposer_prompt(demo_pairs=body.public_train_pairs())

        rounds: list[RoundTrace] = []
        candidates: dict[str, tuple[str, NeMoGymResponse, RoundTrace]] = {}
        last_proposer_response: NeMoGymResponse | None = None
        for index in range(num_candidates):
            proposer_params = self._params(
                body,
                input_items=[_message(base_prompt)],
                max_tokens=self.config.proposer_max_output_tokens,
                instructions=PROPOSER_INSTRUCTIONS,
            )
            proposer_response, _, proposer_call = await self._recorded_model_call(
                request=request,
                role="proposer",
                server=self.config.proposer_model_server,
                params=proposer_params,
                cookies=None,
                run_body=body,
            )
            last_proposer_response = proposer_response
            round_trace = RoundTrace(
                round_index=index,
                proposer=proposer_call,
                transform_description=None,
                executor_calls=[],
                training_verifications=[],
                test_verification=None,
                feedback=None,
            )
            rounds.append(round_trace)
            try:
                description = parse_canonical_rule(_response_text(proposer_response))
            except TransformDescriptionParseError:
                continue  # an unparseable candidate is skipped, not fatal
            round_trace.transform_description = description
            # Zero-padded ids so the server's lexicographic tie-break is
            # emission order.
            candidates[f"cand_{index:03d}"] = (description, proposer_response, round_trace)

        assert last_proposer_response is not None
        if not candidates:
            # Every sample was unparseable: the same floored, loss-on outcome
            # as a hidden_test first-rule parse failure.
            state = EpisodeState.initial().terminate(TerminationReason.AGENT_ERROR)
            return await self._finalize(
                body=body,
                cookies=resource_cookies,
                last_proposer_response=last_proposer_response,
                state=state,
                rounds=rounds,
                loss_masked=False,
                protocol="candidate_select",
                proposer_format_failure=True,
            )

        for candidate_id, (description, _response_obj, round_trace) in candidates.items():
            for grid_id, input_grid in demo_inputs.items():
                resource_cookies = await self._score_candidate_demo(
                    request=request,
                    body=body,
                    candidate_id=candidate_id,
                    description=description,
                    grid_id=grid_id,
                    input_grid=input_grid,
                    round_trace=round_trace,
                    resource_cookies=resource_cookies,
                )

        selection, resource_cookies = await self._call_resource(
            url_path="/select_candidate", payload={}, cookies=resource_cookies
        )
        selected_id = selection["selected_candidate_id"]
        selected_description, selected_response, selected_round = candidates[selected_id]

        state = EpisodeState.initial().description_generated().demos_verified()
        for grid_id, input_grid in test_inputs.items():
            verification, state, resource_cookies = await self._run_single_eval_executor(
                request=request,
                body=body,
                description=selected_description,
                grid_id=grid_id,
                input_grid=input_grid,
                state=state,
                round_trace=selected_round,
                resource_cookies=resource_cookies,
                include_revision_feedback=False,
            )
            if verification is None:
                return await self._finalize(
                    body=body,
                    cookies=resource_cookies,
                    last_proposer_response=selected_response,
                    state=state,
                    rounds=rounds,
                    loss_masked=True,
                    protocol="candidate_select",
                    selected_candidate_id=selected_id,
                    policy_round=selected_round,
                )
        state = state.terminate(TerminationReason.CANDIDATE_SELECTED)
        return await self._finalize(
            body=body,
            cookies=resource_cookies,
            last_proposer_response=selected_response,
            state=state,
            rounds=rounds,
            loss_masked=False,
            protocol="candidate_select",
            selected_candidate_id=selected_id,
            policy_round=selected_round,
            # Scores only -- task-level aggregates, never grids -- so the
            # persisted selection report stays target-free.
            extra_trace={
                "candidate_selection": {
                    "selected_candidate_id": selected_id,
                    "num_candidates": num_candidates,
                    "parseable_candidates": len(candidates),
                    "scores": selection["scores"],
                }
            },
        )


if __name__ == "__main__":
    ArcTransformRefinementAgent.run_webserver()

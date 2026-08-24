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
from typing import Any

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
    build_executor_prompt,
    build_format_retry_prompt,
    build_proposer_prompt,
    build_revision_prompt,
    build_test_followup_prompt,
    conservative_text_token_bound,
    parse_transform_description,
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
    model_context_limit: int = 32_768
    reserved_proposer_output_tokens: int = 4_096
    chat_template_margin: int = 512
    proposer_max_output_tokens: int = 4_096
    executor_max_output_tokens: int = 4_096
    max_rounds: int = 32
    confirm_mismatches: bool = False

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

    @model_validator(mode="after")
    def validate_task(self) -> ArcTransformRunRequest:
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

    def _params(
        self,
        body: ArcTransformRunRequest,
        *,
        input_items: list[dict[str, Any]],
        max_tokens: int,
        instructions: str,
    ) -> dict[str, Any]:
        params = body.responses_create_params.model_dump(exclude_none=True)
        params["input"] = copy.deepcopy(input_items)
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
    ) -> ArcTransformVerifyResponse:
        assert state.termination_reason is not None
        payload = {
            "responses_create_params": body.responses_create_params.model_dump(),
            "response": last_proposer_response.model_dump(),
            "termination_reason": state.termination_reason.value,
            "loss_masked": loss_masked,
            "trace": {
                "task_id": body.task_id,
                "rounds": [asdict(round_trace) for round_trace in rounds],
                "termination_reason": state.termination_reason.value,
                "loss_masked": loss_masked,
                "policy_loss": {
                    "role": "proposer",
                    "round_index": rounds[-1].round_index,
                    "output": last_proposer_response.model_dump(),
                },
            },
        }
        result, _ = await self._call_resource(url_path="/finalize", payload=payload, cookies=cookies)
        return ArcTransformVerifyResponse.model_validate(result)

    async def run(self, request: Request, body: ArcTransformRunRequest) -> ArcTransformVerifyResponse:
        """Run refinement until the train gate passes or a terminal guard fires."""
        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=request.cookies,
        )
        await raise_for_status(seed_response)
        resource_cookies = dict(seed_response.cookies)

        test_inputs = body.public_test_inputs()
        train_inputs = {f"train_{index}": pair.input for index, pair in enumerate(body.train)}
        proposer_history: list[dict[str, Any]] = [
            _message(
                build_proposer_prompt(
                    train_pairs=body.public_train_pairs(),
                    test_inputs=list(test_inputs.values()),
                )
            )
        ]
        budget = ContextBudget(
            model_context_limit=self.config.model_context_limit,
            reserved_proposer_output_tokens=self.config.reserved_proposer_output_tokens,
            chat_template_margin=self.config.chat_template_margin,
        )
        state = EpisodeState.initial()
        rounds: list[RoundTrace] = []
        last_proposer_response: NeMoGymResponse | None = None

        while state.phase is not EpisodePhase.TERMINATED:
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
                description = parse_transform_description(_response_text(proposer_response))
            except TransformDescriptionParseError:
                state = state.terminate(TerminationReason.AGENT_ERROR)
                return await self._finalize(
                    body=body,
                    cookies=resource_cookies,
                    last_proposer_response=last_proposer_response,
                    state=state,
                    rounds=rounds,
                    loss_masked=True,
                )
            round_trace.transform_description = description
            state = state.description_generated()

            executor_history = [
                _message(
                    build_executor_prompt(
                        description=description,
                        inputs=train_inputs,
                        tag="predictions",
                    )
                )
            ]
            while True:
                executor_params = self._params(
                    body,
                    input_items=executor_history,
                    max_tokens=self.config.executor_max_output_tokens,
                    instructions=EXECUTOR_INSTRUCTIONS,
                )
                try:
                    executor_response, _, executor_call = await self._recorded_model_call(
                        request=request,
                        role="executor_train",
                        server=self.config.executor_model_server,
                        params=executor_params,
                        cookies=None,
                        run_body=body,
                    )
                except Exception as error:
                    if not _is_context_window_error(error):
                        raise
                    state = state.terminate(TerminationReason.EXECUTOR_CONTEXT_EXHAUSTED)
                    return await self._finalize(
                        body=body,
                        cookies=resource_cookies,
                        last_proposer_response=last_proposer_response,
                        state=state,
                        rounds=rounds,
                        loss_masked=True,
                    )
                round_trace.executor_calls.append(executor_call)
                training_result, resource_cookies = await self._call_resource(
                    url_path="/verify_training",
                    payload={"response": executor_response.model_dump()},
                    cookies=resource_cookies,
                )
                round_trace.training_verifications.append(training_result)
                if training_result["format_valid"]:
                    break
                state = state.executor_format_failed()
                if state.phase is EpisodePhase.TERMINATED:
                    return await self._finalize(
                        body=body,
                        cookies=resource_cookies,
                        last_proposer_response=last_proposer_response,
                        state=state,
                        rounds=rounds,
                        loss_masked=True,
                    )
                executor_history.extend(item.model_dump() for item in executor_response.output)
                executor_history.append(
                    _message(
                        build_format_retry_prompt(
                            tag="predictions",
                            expected_ids=list(train_inputs),
                            error=training_result["parse_error"],
                        )
                    )
                )

            if self.config.confirm_mismatches and not training_result["all_exact"]:
                confirmation_params = self._params(
                    body,
                    input_items=[executor_history[0]],
                    max_tokens=self.config.executor_max_output_tokens,
                    instructions=EXECUTOR_INSTRUCTIONS,
                )
                confirmation_params["temperature"] = 0.0
                confirmation_response, _, confirmation_call = await self._recorded_model_call(
                    request=request,
                    role="executor_train_confirmation",
                    server=self.config.executor_model_server,
                    params=confirmation_params,
                    cookies=None,
                    run_body=body,
                )
                round_trace.executor_calls.append(confirmation_call)
                confirmation, resource_cookies = await self._call_resource(
                    url_path="/verify_training",
                    payload={
                        "response": confirmation_response.model_dump(),
                        "record_result": False,
                    },
                    cookies=resource_cookies,
                )
                round_trace.training_verifications.append(confirmation)
                if confirmation.get("predictions") != training_result.get("predictions"):
                    training_result["feedback"] += (
                        "\n\nExecutor instability: a deterministic confirmation produced different grids. "
                        "Make the replacement description more operational."
                    )

            state = state.training_verified(all_exact=training_result["all_exact"])
            if state.phase is EpisodePhase.EXECUTOR_TEST:
                executor_history.extend(item.model_dump() for item in executor_response.output)
                executor_history.append(_message(build_test_followup_prompt(test_inputs=test_inputs)))
                test_params = self._params(
                    body,
                    input_items=executor_history,
                    max_tokens=self.config.executor_max_output_tokens,
                    instructions=EXECUTOR_INSTRUCTIONS,
                )
                try:
                    test_response, _, test_call = await self._recorded_model_call(
                        request=request,
                        role="executor_test",
                        server=self.config.executor_model_server,
                        params=test_params,
                        cookies=None,
                        run_body=body,
                    )
                except Exception as error:
                    if not _is_context_window_error(error):
                        raise
                    state = state.terminate(TerminationReason.EXECUTOR_CONTEXT_EXHAUSTED)
                    return await self._finalize(
                        body=body,
                        cookies=resource_cookies,
                        last_proposer_response=last_proposer_response,
                        state=state,
                        rounds=rounds,
                        loss_masked=True,
                    )
                round_trace.executor_calls.append(test_call)
                test_result, resource_cookies = await self._call_resource(
                    url_path="/verify_test",
                    payload={"response": test_response.model_dump()},
                    cookies=resource_cookies,
                )
                round_trace.test_verification = test_result
                if not test_result["format_valid"]:
                    state = state.terminate(TerminationReason.EXECUTOR_FORMAT_FAILURE)
                    loss_masked = True
                else:
                    state = state.test_answered()
                    loss_masked = False
                return await self._finalize(
                    body=body,
                    cookies=resource_cookies,
                    last_proposer_response=last_proposer_response,
                    state=state,
                    rounds=rounds,
                    loss_masked=loss_masked,
                )

            feedback = build_revision_prompt(training_result["feedback"])
            round_trace.feedback = feedback
            current_tokens = proposer_call.prompt_tokens + proposer_call.output_tokens
            if not budget.permits_revision(
                current_proposer_tokens=current_tokens,
                next_feedback_tokens=conservative_text_token_bound(feedback),
            ):
                state = state.terminate(TerminationReason.CONTEXT_EXHAUSTED)
                return await self._finalize(
                    body=body,
                    cookies=resource_cookies,
                    last_proposer_response=last_proposer_response,
                    state=state,
                    rounds=rounds,
                    loss_masked=False,
                )
            if state.round_index >= self.config.max_rounds:
                state = state.terminate(TerminationReason.EMERGENCY_ROUND_CAP)
                return await self._finalize(
                    body=body,
                    cookies=resource_cookies,
                    last_proposer_response=last_proposer_response,
                    state=state,
                    rounds=rounds,
                    loss_masked=False,
                )
            proposer_history.extend(item.model_dump() for item in proposer_response.output)
            proposer_history.append(_message(feedback))

        raise RuntimeError("ARC episode terminated without returning a verifier response")


if __name__ == "__main__":
    ArcTransformRefinementAgent.run_webserver()

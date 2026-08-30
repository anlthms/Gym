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


def _finalize_request(
    *,
    protocol: str,
    loss_masked: bool = False,
    proposer_format_failure: bool = False,
    rounds: int = 1,
) -> ARCAGIFinalizeRequest:
    return ARCAGIFinalizeRequest(
        responses_create_params={"input": []},
        response=_text_response("<rules_summary>r</rules_summary>").model_dump(),
        termination_reason="all_solved",
        loss_masked=loss_masked,
        proposer_format_failure=proposer_format_failure,
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

    async def test_finalize_scores_each_grid_on_its_last_attempt(self) -> None:
        # Final-rule credit: an exact solve by an EARLIER rule does not pay
        # if the final rule's sweep attempt on the same grid misses -- the
        # trained final turn is scored on what its own rule achieves.
        server, request = await self._seeded("eval-last-attempt")
        for grid_id, answer in (("test_0", "0 2"), ("test_1", "0 3"), ("test_0", "9 9")):
            await server.verify_eval_grid(
                request,
                EvalGridVerificationRequest(
                    response=_text_response(f"<answer>\n{answer}\n</answer>"), grid_id=grid_id
                ),
            )
        result = await server.finalize(request, _finalize_request(protocol="eval_sequence"))
        assert result.eval_exact_fraction == 0.5  # test_0's LAST attempt missed
        assert not result.all_solved
        exact_reward = 1.0 + 0.20 + 0.10 + 0.05 + 0.05
        assert result.reward < (exact_reward + exact_reward) / 2  # no best-attempt inheritance

    async def test_finalize_masks_flagged_episodes(self) -> None:
        server, request = await self._seeded("eval-mask")
        result = await server.finalize(request, _finalize_request(protocol="eval_sequence", loss_masked=True))
        assert result.loss_masked
        assert result.instance_config == {"mask_sample": True}

    async def test_finalize_floors_proposer_format_failures_with_loss_on(self) -> None:
        # Even a sequence with solved grids is pinned to the floor when the
        # trained final turn was an unparseable/runaway rule; the sample
        # stays in loss so runaway thinking receives negative gradient.
        server, request = await self._seeded("eval-floor")
        for grid_id, answer in (("test_0", "0 2"), ("test_1", "0 3")):
            await server.verify_eval_grid(
                request,
                EvalGridVerificationRequest(
                    response=_text_response(f"<answer>\n{answer}\n</answer>"), grid_id=grid_id
                ),
            )
        result = await server.finalize(
            request,
            _finalize_request(protocol="eval_sequence", proposer_format_failure=True),
        )
        assert result.proposer_format_failure
        assert not result.loss_masked
        assert result.instance_config == {"mask_sample": False}
        assert result.reward == pytest.approx(-(0.20 + 0.10 + 0.05 + 0.05))
        # Metrics still report what actually happened on the grids.
        assert result.eval_exact_fraction == 1.0


class TestHiddenTest:
    """The real-ARC protocol: demo grids verified during refinement, hidden
    test grids scored on their final answers."""

    async def _seeded(self, session_id: str) -> tuple[ARCAGIResourcesServer, Request]:
        server = _server()
        request = _request(session_id)
        await server.seed_session(
            request,
            ARCAGIRunRequest(
                responses_create_params={"input": []},
                train=[
                    {"input": [[1, 0]], "output": [[0, 1]]},
                    {"input": [[4, 0]], "output": [[0, 4]]},
                ],
                test=[{"input": [[2, 0]], "output": [[0, 2]]}],
                task_id="hidden-task",
            ),
        )
        return server, request

    async def test_verify_eval_grid_resolves_demo_grids(self) -> None:
        server, request = await self._seeded("hidden-demo")
        result = await server.verify_eval_grid(
            request,
            EvalGridVerificationRequest(response=_text_response("<answer>\n0 1\n</answer>"), grid_id="train_0"),
        )
        assert result.format_valid and result.exact
        assert server._sessions["hidden-demo"].eval_results["train_0"][0]["grid_match"] == 1.0
        # Hidden test targets never leave the session.
        assert server._sessions["hidden-demo"].test_targets == {"test_0": [[0, 2]]}

    async def test_verify_eval_grid_demo_miss_renders_behavioral_evidence(self) -> None:
        server, request = await self._seeded("hidden-demo-miss")
        result = await server.verify_eval_grid(
            request,
            EvalGridVerificationRequest(response=_text_response("<answer>\n1 1\n</answer>"), grid_id="train_1"),
        )
        assert result.format_valid and not result.exact
        assert result.revision_feedback is not None
        assert "Input:\n4 0" in result.revision_feedback
        assert "Expected output:\n0 4" in result.revision_feedback

    async def test_finalize_scores_the_final_test_answer(self) -> None:
        server, request = await self._seeded("hidden-final")
        for grid_id, answer in (("train_0", "0 1"), ("train_1", "0 4"), ("test_0", "0 2")):
            await server.verify_eval_grid(
                request,
                EvalGridVerificationRequest(
                    response=_text_response(f"<answer>\n{answer}\n</answer>"), grid_id=grid_id
                ),
            )
        result = await server.finalize(request, _finalize_request(protocol="hidden_test", rounds=1))
        assert result.grid_match == 1.0
        assert result.cell_match == 1.0
        assert result.test_exact
        assert result.train_gate_pass
        assert result.train_exact_fraction == 1.0
        exact_reward = 1.0 + 0.20 + 0.10 + 0.05 + 0.05
        assert result.reward == pytest.approx(exact_reward)

    async def test_finalize_uses_the_last_test_attempt(self) -> None:
        """A format retry replaces the failed first attempt on the same grid."""
        server, request = await self._seeded("hidden-retry")
        for text in ("no grid here", "<answer>\n0 2\n</answer>"):
            await server.verify_eval_grid(
                request, EvalGridVerificationRequest(response=_text_response(text), grid_id="test_0")
            )
        result = await server.finalize(request, _finalize_request(protocol="hidden_test"))
        assert result.grid_match == 1.0
        assert result.format_valid == 1.0

    async def test_finalize_floors_unanswered_test_grids(self) -> None:
        server, request = await self._seeded("hidden-unanswered")
        await server.verify_eval_grid(
            request,
            EvalGridVerificationRequest(response=_text_response("<answer>\n0 1\n</answer>"), grid_id="train_0"),
        )
        result = await server.finalize(request, _finalize_request(protocol="hidden_test", loss_masked=True))
        assert result.grid_match == 0.0
        assert result.cell_match == 0.0
        assert not result.test_exact
        assert not result.train_gate_pass  # train_1 never solved
        assert result.train_exact_fraction == 0.5
        assert result.reward == pytest.approx(-(0.20 + 0.10 + 0.05 + 0.05))
        assert result.instance_config == {"mask_sample": True}

    async def test_finalize_floors_proposer_format_failures_with_loss_on(self) -> None:
        # A fallback rule may still have answered the hidden test exactly,
        # but the trained final turn was unparseable: reward floored, loss
        # ON, and the metrics keep reporting the real answers.
        server, request = await self._seeded("hidden-floor")
        await server.verify_eval_grid(
            request,
            EvalGridVerificationRequest(response=_text_response("<answer>\n0 2\n</answer>"), grid_id="test_0"),
        )
        result = await server.finalize(
            request,
            _finalize_request(protocol="hidden_test", proposer_format_failure=True),
        )
        assert result.proposer_format_failure
        assert not result.loss_masked
        assert result.instance_config == {"mask_sample": False}
        assert result.reward == pytest.approx(-(0.20 + 0.10 + 0.05 + 0.05))
        assert result.grid_match == 1.0
        assert result.cell_match == 1.0


class TestHiddenTestFinalRuleGate:
    """The demo gate must describe the FINAL rule, not any-attempt progress."""

    async def _seeded(self, session_id: str) -> tuple[ARCAGIResourcesServer, Request]:
        server = _server()
        request = _request(session_id)
        await server.seed_session(
            request,
            ARCAGIRunRequest(
                responses_create_params={"input": []},
                train=[
                    {"input": [[1, 0]], "output": [[0, 1]]},
                    {"input": [[4, 0]], "output": [[0, 4]]},
                ],
                test=[{"input": [[2, 0]], "output": [[0, 2]]}],
                task_id="gate-task",
            ),
        )
        return server, request

    async def test_final_rule_regressing_an_earlier_demo_fails_the_gate(self) -> None:
        # Rule 1 solved train_0; the revised (final) rule solves train_1 but
        # its sweep attempt on train_0 misses. Every demo was solved at SOME
        # point, but no single rule solved them all -- the gate must fail.
        server, request = await self._seeded("gate-regress")
        for grid_id, answer in (
            ("train_0", "0 1"),  # rule 1: exact
            ("train_1", "0 4"),  # final rule: exact
            ("train_0", "9 9"),  # final rule sweep: regression
            ("test_0", "0 2"),
        ):
            await server.verify_eval_grid(
                request,
                EvalGridVerificationRequest(
                    response=_text_response(f"<answer>\n{answer}\n</answer>"), grid_id=grid_id
                ),
            )
        result = await server.finalize(request, _finalize_request(protocol="hidden_test"))
        assert not result.train_gate_pass
        assert result.train_exact_fraction == 0.5  # final rule: train_1 only
        assert result.train_any_solved_fraction == 1.0  # loop progress diagnostic

    async def test_final_rule_solving_all_demos_passes_the_gate(self) -> None:
        server, request = await self._seeded("gate-pass")
        for grid_id, answer in (
            ("train_0", "9 9"),  # rule 1: miss
            ("train_0", "0 1"),  # final rule: exact
            ("train_1", "0 4"),  # final rule: exact
            ("test_0", "0 2"),
        ):
            await server.verify_eval_grid(
                request,
                EvalGridVerificationRequest(
                    response=_text_response(f"<answer>\n{answer}\n</answer>"), grid_id=grid_id
                ),
            )
        result = await server.finalize(request, _finalize_request(protocol="hidden_test"))
        assert result.train_gate_pass
        assert result.train_exact_fraction == 1.0


class TestFeedbackSuppression:
    async def test_no_feedback_verifications_never_carry_target_content(self) -> None:
        server = _server()
        request = _request("suppress")
        await server.seed_session(
            request,
            ARCAGIRunRequest(
                responses_create_params={"input": []},
                train=[{"input": [[1, 0]], "output": [[0, 1]]}],
                test=[{"input": [[2, 0]], "output": [[7, 8, 9]]}],
                task_id="suppress-task",
            ),
        )
        result = await server.verify_eval_grid(
            request,
            EvalGridVerificationRequest(
                response=_text_response("<answer>\n2 2\n</answer>"),
                grid_id="test_0",
                include_revision_feedback=False,
            ),
        )
        assert not result.exact
        assert result.revision_feedback is None
        serialized = result.model_dump_json()
        # Neither the evidence prompt nor the diff text leaks the target.
        assert "7 8 9" not in serialized
        assert "Expected output" not in serialized
        assert "mismatch" in result.feedback


class TestCandidateSelection:
    async def _seeded(self, session_id: str) -> tuple[ARCAGIResourcesServer, Request]:
        server = _server()
        request = _request(session_id)
        await server.seed_session(
            request,
            ARCAGIRunRequest(
                responses_create_params={"input": []},
                train=[
                    {"input": [[1, 0]], "output": [[0, 1]]},
                    {"input": [[4, 0]], "output": [[0, 4]]},
                ],
                test=[{"input": [[2, 0]], "output": [[0, 2]]}],
                task_id="cand-task",
            ),
        )
        return server, request

    async def _record(self, server, request, candidate_id: str, grid_id: str, answer: str) -> None:
        await server.verify_eval_grid(
            request,
            EvalGridVerificationRequest(
                response=_text_response(f"<answer>\n{answer}\n</answer>"),
                grid_id=grid_id,
                candidate_id=candidate_id,
                include_revision_feedback=False,
            ),
        )

    async def test_candidate_attempts_stay_out_of_episode_results(self) -> None:
        server, request = await self._seeded("cand-namespace")
        await self._record(server, request, "cand_000", "train_0", "0 1")
        session = server._sessions["cand-namespace"]
        assert session.eval_results == {}
        assert session.candidate_results["cand_000"]["train_0"][0]["grid_match"] == 1.0

    async def test_selection_ranks_by_exact_count_then_cell_then_order(self) -> None:
        from resources_servers.arc_agi_2.app import CandidateSelectionRequest

        server, request = await self._seeded("cand-rank")
        # cand_000: one exact, one miss. cand_001: both exact (must win).
        # cand_002: both exact too, but later in emission order.
        await self._record(server, request, "cand_000", "train_0", "0 1")
        await self._record(server, request, "cand_000", "train_1", "9 9")
        for candidate_id in ("cand_001", "cand_002"):
            await self._record(server, request, candidate_id, "train_0", "0 1")
            await self._record(server, request, candidate_id, "train_1", "0 4")
        selection = await server.select_candidate(request, CandidateSelectionRequest())
        assert selection.selected_candidate_id == "cand_001"
        assert selection.scores["cand_001"]["demo_exact_count"] == 2.0
        assert selection.scores["cand_000"]["demo_exact_count"] == 1.0
        # Scores are task-level aggregates only: no grids anywhere.
        serialized = selection.model_dump_json()
        assert "0 1" not in serialized and "[[" not in serialized

    async def test_selection_tie_breaks_on_cell_match_before_order(self) -> None:
        from resources_servers.arc_agi_2.app import CandidateSelectionRequest

        server, request = await self._seeded("cand-cell")
        # Equal exact counts (zero); cand_001 is closer on cells and must win
        # despite its later emission order.
        await self._record(server, request, "cand_000", "train_0", "9 9")
        await self._record(server, request, "cand_000", "train_1", "9 9")
        await self._record(server, request, "cand_001", "train_0", "0 9")
        await self._record(server, request, "cand_001", "train_1", "0 9")
        selection = await server.select_candidate(request, CandidateSelectionRequest())
        assert selection.selected_candidate_id == "cand_001"

    async def test_selection_without_recorded_candidates_is_an_error(self) -> None:
        from fastapi import HTTPException

        from resources_servers.arc_agi_2.app import CandidateSelectionRequest

        server, request = await self._seeded("cand-empty")
        with pytest.raises(HTTPException):
            await server.select_candidate(request, CandidateSelectionRequest())

    async def test_finalize_verifies_the_selection_and_scores_one_test_touch(self) -> None:
        server, request = await self._seeded("cand-final")
        await self._record(server, request, "cand_000", "train_0", "9 9")
        await self._record(server, request, "cand_000", "train_1", "9 9")
        await self._record(server, request, "cand_001", "train_0", "0 1")
        await self._record(server, request, "cand_001", "train_1", "0 4")
        # Only the selected rule touches the test, recorded in the main pool.
        await server.verify_eval_grid(
            request,
            EvalGridVerificationRequest(
                response=_text_response("<answer>\n0 2\n</answer>"),
                grid_id="test_0",
                include_revision_feedback=False,
            ),
        )
        finalize = _finalize_request(protocol="candidate_select")
        finalize = finalize.model_copy(update={"selected_candidate_id": "cand_001"})
        result = await server.finalize(request, finalize)
        assert result.selection_verified
        assert result.selected_candidate_id == "cand_001"
        assert result.num_candidates == 2
        assert result.train_gate_pass  # the SELECTED candidate solved every demo
        assert result.test_exact and result.grid_match == 1.0

    async def test_finalize_flags_a_selection_that_disagrees_with_demo_ranking(self) -> None:
        server, request = await self._seeded("cand-mismatch")
        await self._record(server, request, "cand_000", "train_0", "0 1")
        await self._record(server, request, "cand_000", "train_1", "0 4")
        await self._record(server, request, "cand_001", "train_0", "9 9")
        await self._record(server, request, "cand_001", "train_1", "9 9")
        finalize = _finalize_request(protocol="candidate_select")
        finalize = finalize.model_copy(update={"selected_candidate_id": "cand_001"})
        result = await server.finalize(request, finalize)
        assert not result.selection_verified
        assert not result.train_gate_pass  # gate follows the COMMITTED candidate


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


class TestSingleTurnSeeding:
    async def test_seed_session_accepts_single_turn_rows(self) -> None:
        """Executor and induction rows carry target/test_input and no train pairs."""
        server = _server()
        request = _request("single-turn")
        await server.seed_session(
            request,
            ARCAGIRunRequest(
                responses_create_params={"input": []},
                target=[[0, 1]],
                test_input=[[1, 0]],
                task_id="executor-row",
            ),
        )
        session = server._sessions["single-turn"]
        assert session.train_targets == {}
        assert session.test_targets == {"test_0": [[0, 1]]}

    def test_episode_rows_still_require_training_pairs(self) -> None:
        with pytest.raises(ValueError, match="at least one training pair"):
            ARCAGIRunRequest(
                responses_create_params={"input": []},
                test=[{"input": [[1]], "output": [[2]]}],
            )

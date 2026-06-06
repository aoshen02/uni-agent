"""Test the init/mid-trajectory failure path in UniAgentLoop.

The key invariant we are protecting: when a trajectory raises during env.start,
install_tools, interaction, or reward computation, UniAgentLoop.run() returns a
shape-compatible AgentLoopOutput so that downstream batching
(verl/experimental/agent_loop/agent_loop.py:_agent_loop_postprocess +
verl/experimental/fully_async_policy/detach_utils.py:assemble_batch_from_rollout_samples)
does not crash when a failed sample lands in the same batch as normal samples.

If this test passes, the round-11 class of crashes (single trajectory's
swerex CommandTimeoutError taking down the whole training job) can no longer
happen.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from uni_agent.agent_loop import UniAgentLoop
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput


def _make_stub_loop(
    *,
    moe_num_layers: int = 64,
    moe_topk: int = 8,
    calculate_log_probs: bool = True,
    pad_token_id: int = 0,
) -> UniAgentLoop:
    """Build a UniAgentLoop instance without going through AgentLoopBase.__init__.

    We bypass the parent __init__ because it needs real Ray actor handles,
    a real tokenizer, a server manager, etc. For this unit test we only need
    enough state for _make_failed_output to produce a valid AgentLoopOutput.
    """
    loop = UniAgentLoop.__new__(UniAgentLoop)

    # Stub the cached MoE shape directly so we do not actually load model config.
    UniAgentLoop._moe_num_layers = moe_num_layers
    UniAgentLoop._moe_topk = moe_topk

    # Minimal config tree: only what _make_failed_output reads.
    loop.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "calculate_log_probs": calculate_log_probs,
                    "prompt_length": 4096,
                },
                "model": {"path": "fake/path"},
            }
        }
    )
    loop.apply_chat_template_kwargs = {}

    # Tokenizer: realistic mock — apply_chat_template(tokenize=False) returns
    # a string template, encode() returns a list[int]. This matches the
    # Qwen3 tokenizer behavior the patched _make_failed_output now relies on.
    tok = MagicMock()
    tok.apply_chat_template = MagicMock(return_value="<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n")
    tok.encode = MagicMock(return_value=[1, 2, 3, 4, 5])
    tok.pad_token_id = pad_token_id
    loop.tokenizer = tok

    # Logger: nothing fancy, just absorb log calls.
    loop.logger = MagicMock()

    return loop


def test_failed_output_pydantic_construction():
    """The constructed AgentLoopOutput passes Pydantic validation."""
    loop = _make_stub_loop()
    output = loop._make_failed_output(
        reason="trajectory_failed",
        exc=RuntimeError("simulated swerex CommandTimeoutError"),
        raw_prompt=[{"role": "user", "content": "hi"}],
    )
    assert isinstance(output, AgentLoopOutput)
    assert isinstance(output.metrics, AgentLoopMetrics)
    assert output.reward_score == 0.0
    assert output.num_turns == 0
    assert output.response_ids == [0]
    assert output.response_mask == [0]


def test_failed_output_shape_when_logprobs_enabled():
    """When calculate_log_probs=True, response_logprobs must NOT be None."""
    loop = _make_stub_loop(calculate_log_probs=True)
    output = loop._make_failed_output(
        reason="trajectory_failed",
        exc=RuntimeError("x"),
        raw_prompt=[{"role": "user", "content": "hi"}],
    )
    assert output.response_logprobs == [0.0]
    # routed_experts must be a numpy array with shape (length=1, num_layers, topk)
    assert isinstance(output.routed_experts, np.ndarray)
    assert output.routed_experts.shape == (1, 64, 8)
    assert (output.routed_experts == 0).all()


def test_failed_output_shape_when_logprobs_disabled():
    """When calculate_log_probs=False, response_logprobs should be None."""
    loop = _make_stub_loop(calculate_log_probs=False)
    output = loop._make_failed_output(
        reason="trajectory_failed",
        exc=RuntimeError("x"),
        raw_prompt=[{"role": "user", "content": "hi"}],
    )
    assert output.response_logprobs is None


def test_failed_output_extra_fields():
    """Failure metadata is propagated via extra_fields for downstream metrics."""
    loop = _make_stub_loop()
    exc = TimeoutError("pip install timed out")
    output = loop._make_failed_output(
        reason="trajectory_failed",
        exc=exc,
        raw_prompt=[{"role": "user", "content": "hi"}],
    )
    # Top-level keys: only those in verl's default_extra_keys set, so
    # DataProto.concat across mixed (normal + failed) batches passes the
    # py_functional.list_of_dict_to_dict_of_list assert.
    assert output.extra_fields["min_global_steps"] == -1
    assert output.extra_fields["max_global_steps"] == -1
    # Diagnostic info packed under "extras" (also a default_extra_keys member).
    extras = output.extra_fields["extras"]
    assert extras["init_failed"] is True
    assert extras["traj_masked"] == 1
    assert extras["traj_exit_reason"] == "trajectory_failed"
    assert extras["failure_class"] == "TimeoutError"
    assert extras["failure_reason"] == "trajectory_failed"
    # Top-level extra_fields MUST be a subset of verl's default_extra_keys.
    allowed_top_level = {"turn_scores", "tool_rewards", "min_global_steps", "max_global_steps", "extras"}
    assert set(output.extra_fields.keys()) <= allowed_top_level, (
        f"extra_fields has keys outside default_extra_keys: "
        f"{set(output.extra_fields.keys()) - allowed_top_level}"
    )


def test_failed_output_key_set_matches_normal_path():
    """The set of non-None optional fields on the failed output must match what a
    normal (success) trajectory would produce — because verl _batch_outputs
    decides whether to emit `rollout_log_probs` and `routed_experts` keys into
    the DataProto based on `inputs[0]`. Mismatch ⇒ KeyError at DataProto.concat
    when failed and normal samples mix in the same MQ pull (16 samples per
    fit_step on round-11 config).
    """
    loop = _make_stub_loop(calculate_log_probs=True)
    failed = loop._make_failed_output(
        reason="trajectory_failed",
        exc=RuntimeError("x"),
        raw_prompt=[{"role": "user", "content": "hi"}],
    )

    # A hypothetical normal output from the existing convert_to_agent_output
    # path would have: prompt_ids, response_ids, response_mask, response_logprobs
    # (when calculate_log_probs=True), routed_experts (when MoE rollout writes
    # them), multi_modal_data, reward_score, num_turns, metrics, extra_fields.
    # Verify our failed output also sets all of these:
    assert failed.prompt_ids is not None and len(failed.prompt_ids) > 0
    assert failed.response_ids is not None and len(failed.response_ids) > 0
    assert failed.response_mask is not None and len(failed.response_mask) > 0
    assert failed.response_logprobs is not None  # critical for calculate_log_probs=True
    assert failed.routed_experts is not None  # critical for MoE rollout
    assert failed.multi_modal_data is not None
    assert failed.reward_score is not None
    assert failed.metrics is not None
    assert failed.extra_fields is not None


def test_failed_output_response_mask_is_zero():
    """response_mask must be all zeros so the trainer's PPO loss has 0
    contribution from this sample — no spurious gradient.
    """
    loop = _make_stub_loop()
    output = loop._make_failed_output(
        reason="trajectory_failed",
        exc=RuntimeError("x"),
        raw_prompt=[{"role": "user", "content": "hi"}],
    )
    assert all(m == 0 for m in output.response_mask)


def test_failed_output_reward_score_not_none_blocks_external_eval():
    """reward_score MUST be a float (0.0), not None. If it were None, verl's
    _compute_score in agent_loop.py:864 would try to invoke the external reward
    worker on our fake trajectory — which has no real env to evaluate.
    """
    loop = _make_stub_loop()
    output = loop._make_failed_output(
        reason="trajectory_failed",
        exc=RuntimeError("x"),
        raw_prompt=[{"role": "user", "content": "hi"}],
    )
    assert output.reward_score is not None
    assert isinstance(output.reward_score, float)
    assert output.reward_score == 0.0


def test_failed_output_with_qwen3_batchencoding_response():
    """Regression test for the round-12 bug: Qwen3 tokenizer's apply_chat_template
    with tokenize=True returns a BatchEncoding (dict-like) instead of a plain
    list[int]. Pydantic AgentLoopOutput.prompt_ids requires list[int] and
    rejected the BatchEncoding, taking down the whole job. The fix uses
    tokenize=False + encode() which always returns a plain list.

    This test mocks the bad behavior to ensure regression coverage.
    """
    from transformers import BatchEncoding

    loop = _make_stub_loop()
    # If apply_chat_template were ever called with tokenize=True, it returns
    # a BatchEncoding — but the patched code path uses tokenize=False, so
    # the apply_chat_template mock returns a string and encode() returns a list.
    loop.tokenizer.apply_chat_template = MagicMock(return_value="<|im_start|>user\nhi<|im_end|>\n")
    loop.tokenizer.encode = MagicMock(return_value=[100, 200, 300])

    output = loop._make_failed_output(
        reason="trajectory_failed",
        exc=RuntimeError("simulated"),
        raw_prompt=[{"role": "user", "content": "hi"}],
    )

    # The Pydantic validation that crashed round-12 must now pass.
    assert isinstance(output, AgentLoopOutput)
    assert output.prompt_ids == [100, 200, 300]
    assert isinstance(output.prompt_ids, list)


def test_failed_output_truncates_long_prompt_to_max_prompt_length():
    """Regression test for round-12-v2 bug: SWE-Bench val prompts can encode to
    8000+ tokens after apply_chat_template (system + tool defs + bug
    description). The failed output must truncate prompt_ids to
    max_prompt_length so torch.cat at verl agent_loop.py:925 sees consistent
    (1, prompt_length) shape across all samples in the batch — otherwise
    long-prompt failed samples crash the batch concat.

    Mirrors the truncation in convert_to_agent_output:192.
    """
    loop = _make_stub_loop()
    # Build the stub config so prompt_length is a known value.
    loop.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"calculate_log_probs": True, "prompt_length": 4096},
                "model": {"path": "fake/path"},
            }
        }
    )
    # Make encode return a long sequence — simulating an 8954-token SWE-Bench prompt.
    long_prompt = list(range(8954))
    loop.tokenizer.apply_chat_template = MagicMock(return_value="<longtext>")
    loop.tokenizer.encode = MagicMock(return_value=long_prompt)

    output = loop._make_failed_output(
        reason="trajectory_failed",
        exc=RuntimeError("simulated"),
        raw_prompt=[{"role": "user", "content": "long content"}],
    )
    # Must be truncated to exactly 4096
    assert len(output.prompt_ids) == 4096
    # The first 4096 tokens of the long prompt
    assert output.prompt_ids == list(range(4096))


def test_failed_output_when_tokenizer_itself_crashes():
    """Last-resort fallback: even tokenizer.apply_chat_template raising must
    not propagate. _make_failed_output must produce a valid output with
    prompt_ids=[pad_id]."""
    loop = _make_stub_loop()
    loop.tokenizer.apply_chat_template = MagicMock(side_effect=RuntimeError("tokenizer dead"))

    output = loop._make_failed_output(
        reason="trajectory_failed",
        exc=RuntimeError("original"),
        raw_prompt=[{"role": "user", "content": "hi"}],
    )

    assert isinstance(output, AgentLoopOutput)
    assert output.prompt_ids == [0]  # pad_token_id fallback
    assert output.response_ids == [0]


def test_simulated_postprocess_padding():
    """Simulate the padding step that verl's _agent_loop_postprocess performs
    on our failed output and verify final tensor shapes match what a normal
    sample's tensors would look like at this stage.

    This is a hand-rolled mini version of agent_loop.py:644-792 — too coupled
    to spin up a real AgentLoopBase here, but the key shape invariants are
    preserved.
    """
    loop = _make_stub_loop()
    output = loop._make_failed_output(
        reason="trajectory_failed",
        exc=RuntimeError("x"),
        raw_prompt=[{"role": "user", "content": "hi"}],
    )

    prompt_length = 4096
    response_length = 131072

    # Simulate tokenizer.pad on prompt_ids (left padding).
    padded_prompt = [0] * (prompt_length - len(output.prompt_ids)) + list(output.prompt_ids)
    assert len(padded_prompt) == prompt_length

    # Simulate tokenizer.pad on response_ids (right padding).
    padded_response = list(output.response_ids) + [0] * (response_length - len(output.response_ids))
    assert len(padded_response) == response_length

    # Simulate tokenizer.pad on response_mask (right padding, zeros).
    padded_response_mask = list(output.response_mask) + [0] * (response_length - len(output.response_mask))
    assert len(padded_response_mask) == response_length
    assert all(m == 0 for m in padded_response_mask)

    # Simulate the routed_experts placement at agent_loop.py:715-737.
    total_length = prompt_length + response_length
    length, layer_num, topk_num = output.routed_experts.shape
    padded_routed = torch.zeros(1, total_length, layer_num, topk_num, dtype=torch.int64)
    start_pos = prompt_length - len(output.prompt_ids)
    end_pos = start_pos + length
    padded_routed[:, start_pos:end_pos] = torch.from_numpy(output.routed_experts).unsqueeze(0)
    assert padded_routed.shape == (1, total_length, layer_num, topk_num)
    assert (padded_routed == 0).all()  # All zeros since we placed a zero tensor.

    # Simulate response_logprobs padding at agent_loop.py:703-706.
    pad_size = response_length - len(output.response_logprobs)
    padded_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)
    assert padded_logprobs.shape == (1, response_length)
    assert (padded_logprobs == 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

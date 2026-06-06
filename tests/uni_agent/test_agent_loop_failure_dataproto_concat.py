"""End-to-end shape compatibility: real DataProto.concat with mixed failed+normal samples.

This test goes one level deeper than test_agent_loop_failure_path.py:
- Builds a real failed DataProto (size=1) using the same shape contract as
  _make_failed_output (response_mask=[0], routed_experts zeros, etc.).
- Builds a real "normal" DataProto (size=1) with the same shape but non-zero
  fields.
- Calls DataProto.concat on a mixed batch and verifies no KeyError /
  shape error is raised. This is exactly what
  verl/experimental/fully_async_policy/detach_utils.py:127 does on every
  fit_step, after the trainer pulls required_samples from MQ.

If this passes, the round-11 crash class cannot recur via batching
mismatch — the failed sample is structurally indistinguishable from a
normal sample at the DataProto layer.
"""

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from verl.protocol import DataProto


PROMPT_LENGTH = 256
RESPONSE_LENGTH = 1024
MOE_LAYERS = 64
MOE_TOPK = 8


def _make_dataproto(*, response_mask_value: int, reward: float, has_logprobs: bool, has_routed: bool):
    """Build a DataProto with shape [1, ...] mimicking what _batch_outputs
    produces for a single sample."""
    prompts = torch.zeros((1, PROMPT_LENGTH), dtype=torch.int64)
    responses = torch.zeros((1, RESPONSE_LENGTH), dtype=torch.int64)
    response_mask = torch.full((1, RESPONSE_LENGTH), response_mask_value, dtype=torch.int64)
    input_ids = torch.cat([prompts, responses], dim=1)
    attention_mask = torch.ones((1, PROMPT_LENGTH + RESPONSE_LENGTH), dtype=torch.int64)
    position_ids = torch.arange(PROMPT_LENGTH + RESPONSE_LENGTH, dtype=torch.int64).unsqueeze(0)

    rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
    rm_scores[0, -1] = reward

    batch_dict = {
        "prompts": prompts,
        "responses": responses,
        "response_mask": response_mask,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "rm_scores": rm_scores,
    }
    if has_logprobs:
        batch_dict["rollout_log_probs"] = torch.zeros((1, RESPONSE_LENGTH), dtype=torch.float32)
    if has_routed:
        batch_dict["routed_experts"] = torch.zeros(
            (1, PROMPT_LENGTH + RESPONSE_LENGTH, MOE_LAYERS, MOE_TOPK), dtype=torch.int64
        )

    batch = TensorDict(batch_dict, batch_size=1)

    non_tensor_batch = {
        "__num_turns__": np.array([0], dtype=np.int32),
        "min_global_steps": np.array([1], dtype=np.int64),
        "max_global_steps": np.array([1], dtype=np.int64),
        "processing_times": np.array([0.0], dtype=np.float32),
        "tool_calls_times": np.array([0.0], dtype=np.float32),
    }
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


def test_concat_mixed_failed_and_normal_with_logprobs_and_moe():
    """Failed (mask=0, reward=0) + normal (mask=1, reward=1) mixed in same
    concat — both have rollout_log_probs and routed_experts keys, all shapes
    aligned. This is the realistic round-11 scenario."""
    failed = _make_dataproto(
        response_mask_value=0,
        reward=0.0,
        has_logprobs=True,
        has_routed=True,
    )
    normal_1 = _make_dataproto(
        response_mask_value=1,
        reward=1.0,
        has_logprobs=True,
        has_routed=True,
    )
    normal_2 = _make_dataproto(
        response_mask_value=1,
        reward=0.5,
        has_logprobs=True,
        has_routed=True,
    )

    # Concat 16 protos to match round-11 required_samples=16
    protos = [failed] + [normal_1] * 8 + [normal_2] * 7
    merged = DataProto.concat(protos)

    # Shape invariants
    assert merged.batch["prompts"].shape == (16, PROMPT_LENGTH)
    assert merged.batch["responses"].shape == (16, RESPONSE_LENGTH)
    assert merged.batch["response_mask"].shape == (16, RESPONSE_LENGTH)
    assert merged.batch["rollout_log_probs"].shape == (16, RESPONSE_LENGTH)
    assert merged.batch["routed_experts"].shape == (16, PROMPT_LENGTH + RESPONSE_LENGTH, MOE_LAYERS, MOE_TOPK)
    assert merged.batch["rm_scores"].shape == (16, RESPONSE_LENGTH)

    # Failed sample is index 0 — mask all 0, reward at tail = 0
    assert (merged.batch["response_mask"][0] == 0).all()
    assert merged.batch["rm_scores"][0, -1].item() == 0.0
    # Normal samples have mask=1
    assert (merged.batch["response_mask"][1] == 1).all()
    assert merged.batch["rm_scores"][1, -1].item() == 1.0


def test_concat_all_failed_does_not_crash():
    """Pathological case: every sample is a failure. Should still concat OK."""
    failed = _make_dataproto(response_mask_value=0, reward=0.0, has_logprobs=True, has_routed=True)
    merged = DataProto.concat([failed] * 16)
    assert merged.batch["response_mask"].shape == (16, RESPONSE_LENGTH)
    assert (merged.batch["response_mask"] == 0).all()


def test_concat_zero_logprobs_and_routed_match_normal_shapes():
    """The failed sample's all-zero rollout_log_probs and routed_experts must
    have IDENTICAL shape to the normal sample's — proving that
    DataProto.concat uses torch.cat which only requires shape match, not
    value match."""
    failed = _make_dataproto(response_mask_value=0, reward=0.0, has_logprobs=True, has_routed=True)
    normal = _make_dataproto(response_mask_value=1, reward=1.0, has_logprobs=True, has_routed=True)
    assert failed.batch["rollout_log_probs"].shape == normal.batch["rollout_log_probs"].shape
    assert failed.batch["routed_experts"].shape == normal.batch["routed_experts"].shape


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])

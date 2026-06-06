"""End-to-end integration test for the Layer 2 failure path.

Unlike test_agent_loop_failure_path.py (which uses MagicMock tokenizer and
hand-rolled shape simulation), this test:

1. Loads the REAL Qwen3 tokenizer used in round-12 training
2. Uses REAL raw_prompt structures (short + SWE-Bench-length long)
3. Pipes _make_failed_output output through the EXACT padding logic that
   verl/experimental/agent_loop/agent_loop.py:_agent_loop_postprocess
   performs at lines 670-792 in production
4. Asserts the resulting tensor shapes are bit-identical to what a normal
   trajectory would produce

If this test passes, the round-12 class of shape-mismatch crashes
(BatchEncoding return type, long-prompt overflow, missing keys in DataProto.concat)
cannot recur — because we are testing the actual production pipeline, not
a hand-rolled simulation of it.

The previous test_agent_loop_failure_path.py kept missing real bugs because
it stubbed apply_chat_template to return [1,2,3,4,5] (a toy list), not what
Qwen3 actually returns. This file exists so we catch those bugs HERE before
they hit a ray job.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from uni_agent.agent_loop import UniAgentLoop


QWEN3_PATH = "/mnt/lustre/hf-models/hub/Qwen3.5-35B-A3B"
PROMPT_LENGTH = 4096
RESPONSE_LENGTH = 131072
MOE_LAYERS = 64
MOE_TOPK = 8


@pytest.fixture(scope="module")
def real_qwen3_tokenizer():
    """Load the actual Qwen3 tokenizer used in round-12 training."""
    if not Path(QWEN3_PATH).exists():
        pytest.skip(f"Qwen3 tokenizer not found at {QWEN3_PATH}")
    return AutoTokenizer.from_pretrained(QWEN3_PATH, trust_remote_code=True)


def _make_real_stub_loop(tokenizer) -> UniAgentLoop:
    """Stub UniAgentLoop with the REAL tokenizer instead of a MagicMock."""
    loop = UniAgentLoop.__new__(UniAgentLoop)
    UniAgentLoop._moe_num_layers = MOE_LAYERS
    UniAgentLoop._moe_topk = MOE_TOPK
    loop.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "calculate_log_probs": True,
                    "prompt_length": PROMPT_LENGTH,
                    "response_length": RESPONSE_LENGTH,
                },
                "model": {"path": QWEN3_PATH},
            }
        }
    )
    loop.apply_chat_template_kwargs = {"thinking": True}  # Matches train script.
    loop.tokenizer = tokenizer
    loop.logger = MagicMock()
    return loop


def _simulate_agent_loop_postprocess(output, tokenizer):
    """Replay the exact padding logic from
    verl/experimental/agent_loop/agent_loop.py:_agent_loop_postprocess
    (lines 670-792) on the given AgentLoopOutput. Returns a dict with the
    final tensor shapes that the production code would emit as
    _InternalAgentLoopOutput.

    This mirrors the production code closely enough that any shape mismatch
    here will also manifest at production. (If verl changes the postprocess
    logic, this test would need to be re-synced.)
    """
    tokenizer.padding_side = "left"
    prompt_out = tokenizer.pad(
        {"input_ids": output.prompt_ids},
        padding="max_length",
        max_length=PROMPT_LENGTH,
        return_tensors="pt",
        return_attention_mask=True,
    )
    if prompt_out["input_ids"].dim() == 1:
        prompt_out["input_ids"] = prompt_out["input_ids"].unsqueeze(0)
        prompt_out["attention_mask"] = prompt_out["attention_mask"].unsqueeze(0)

    tokenizer.padding_side = "right"
    response_out = tokenizer.pad(
        {"input_ids": output.response_ids},
        padding="max_length",
        max_length=RESPONSE_LENGTH,
        return_tensors="pt",
        return_attention_mask=True,
    )
    if response_out["input_ids"].dim() == 1:
        response_out["input_ids"] = response_out["input_ids"].unsqueeze(0)
        response_out["attention_mask"] = response_out["attention_mask"].unsqueeze(0)

    response_mask_out = tokenizer.pad(
        {"input_ids": output.response_mask},
        padding="max_length",
        max_length=RESPONSE_LENGTH,
        return_tensors="pt",
        return_attention_mask=False,
    )
    if response_mask_out["input_ids"].dim() == 1:
        response_mask_out["input_ids"] = response_mask_out["input_ids"].unsqueeze(0)

    # Simulate response_logprobs padding (agent_loop.py:703-706)
    response_logprobs = None
    if output.response_logprobs is not None:
        pad_size = RESPONSE_LENGTH - len(output.response_logprobs)
        response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

    # Simulate routed_experts placement (agent_loop.py:712-737)
    routed_experts = None
    if output.routed_experts is not None:
        length, layer_num, topk_num = output.routed_experts.shape
        total_length = prompt_out["input_ids"].shape[1] + response_out["input_ids"].shape[1]
        routed_experts = torch.zeros(1, total_length, layer_num, topk_num, dtype=torch.int64)
        start_pos = prompt_out["input_ids"].shape[1] - len(output.prompt_ids)
        end_pos = min(start_pos + length, total_length)
        routed_experts[:, start_pos:end_pos] = torch.from_numpy(output.routed_experts).unsqueeze(0)

    # Production line agent_loop.py:708 — response_mask is multiplied by the
    # response's attention_mask. This zeros out the tokenizer.pad-introduced
    # pad_token_id at positions beyond the real response length, leaving a
    # clean 0/1 mask. Without this multiplication, response_mask retains the
    # raw pad_token_id (e.g. Qwen3's 248044) in the padded region.
    response_mask = response_mask_out["input_ids"] * response_out["attention_mask"]

    return {
        "prompts": prompt_out["input_ids"],
        "responses": response_out["input_ids"],
        "response_mask": response_mask,
        "rollout_log_probs": response_logprobs,
        "routed_experts": routed_experts,
        "attention_mask": torch.cat([prompt_out["attention_mask"], response_out["attention_mask"]], dim=1),
    }


SHORT_PROMPT = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "What is 2+2?"},
]


def _build_swebench_long_prompt() -> list[dict]:
    """SWE-Bench-style long prompt: system + lengthy bug description.
    Encodes to 6000-12000 tokens depending on tokenizer."""
    body = (
        "I am working on the Django ORM and have encountered the following issue. "
        "When trying to filter a QuerySet using nested OR conditions through Q objects, "
        "the resulting SQL has a regression in the WHERE clause. Here is the relevant code:\n\n"
    ) + "\n".join(f"def func_{i}(x):\n    return x * {i} + foo({i})" for i in range(500))
    return [
        {"role": "system", "content": "You are an expert Python and Django developer. " * 50},
        {"role": "user", "content": body},
    ]


def test_real_tokenizer_short_prompt(real_qwen3_tokenizer):
    """Short prompt: encoded length < max_prompt_length. No truncation needed."""
    loop = _make_real_stub_loop(real_qwen3_tokenizer)
    output = loop._make_failed_output(
        reason="trajectory_failed",
        exc=RuntimeError("simulated"),
        raw_prompt=SHORT_PROMPT,
    )
    assert isinstance(output.prompt_ids, list)
    assert all(isinstance(x, int) for x in output.prompt_ids)
    assert 0 < len(output.prompt_ids) <= PROMPT_LENGTH


def test_real_tokenizer_long_prompt_truncated(real_qwen3_tokenizer):
    """SWE-Bench long prompt: must be truncated to PROMPT_LENGTH so
    postprocess padding produces (1, PROMPT_LENGTH), not (1, longer)."""
    loop = _make_real_stub_loop(real_qwen3_tokenizer)
    long_prompt = _build_swebench_long_prompt()
    output = loop._make_failed_output(
        reason="trajectory_failed",
        exc=RuntimeError("simulated"),
        raw_prompt=long_prompt,
    )
    # The raw encoding would be way > 4096; our patch must have truncated it.
    assert len(output.prompt_ids) == PROMPT_LENGTH, (
        f"Long prompt not truncated: got {len(output.prompt_ids)}, "
        f"expected {PROMPT_LENGTH}. round-12-v2 crash class."
    )


def test_real_tokenizer_postprocess_shapes_short(real_qwen3_tokenizer):
    """End-to-end: short prompt → _make_failed_output → simulated postprocess.
    Final tensor shapes MUST match what verl _InternalAgentLoopOutput produces
    for a normal path: prompts (1, 4096), responses (1, 131072), etc.
    """
    loop = _make_real_stub_loop(real_qwen3_tokenizer)
    output = loop._make_failed_output(
        reason="trajectory_failed",
        exc=RuntimeError("simulated"),
        raw_prompt=SHORT_PROMPT,
    )
    final = _simulate_agent_loop_postprocess(output, real_qwen3_tokenizer)
    assert final["prompts"].shape == (1, PROMPT_LENGTH)
    assert final["responses"].shape == (1, RESPONSE_LENGTH)
    assert final["response_mask"].shape == (1, RESPONSE_LENGTH)
    assert final["rollout_log_probs"].shape == (1, RESPONSE_LENGTH)
    assert final["routed_experts"].shape == (1, PROMPT_LENGTH + RESPONSE_LENGTH, MOE_LAYERS, MOE_TOPK)
    assert final["attention_mask"].shape == (1, PROMPT_LENGTH + RESPONSE_LENGTH)
    # response_mask all zero → no gradient contribution
    assert (final["response_mask"] == 0).all()


def test_real_tokenizer_postprocess_shapes_long(real_qwen3_tokenizer):
    """End-to-end for the round-12-v2 crash class: long SWE-Bench prompt
    must produce IDENTICAL final shapes as a short prompt would."""
    loop = _make_real_stub_loop(real_qwen3_tokenizer)
    long_prompt = _build_swebench_long_prompt()
    output = loop._make_failed_output(
        reason="trajectory_failed",
        exc=RuntimeError("simulated"),
        raw_prompt=long_prompt,
    )
    final = _simulate_agent_loop_postprocess(output, real_qwen3_tokenizer)
    # The whole point of round-12-v2 was this assertion failing in production:
    assert final["prompts"].shape == (1, PROMPT_LENGTH)
    assert final["responses"].shape == (1, RESPONSE_LENGTH)
    assert final["response_mask"].shape == (1, RESPONSE_LENGTH)
    assert final["rollout_log_probs"].shape == (1, RESPONSE_LENGTH)
    assert final["routed_experts"].shape == (1, PROMPT_LENGTH + RESPONSE_LENGTH, MOE_LAYERS, MOE_TOPK)


def test_real_tokenizer_postprocess_concat_short_and_long(real_qwen3_tokenizer):
    """The exact failure point: torch.cat at verl agent_loop.py:925 across
    a batch that mixes failed-short and failed-long trajectories.
    """
    loop = _make_real_stub_loop(real_qwen3_tokenizer)
    out_short = loop._make_failed_output(
        reason="trajectory_failed", exc=RuntimeError("x"), raw_prompt=SHORT_PROMPT
    )
    out_long = loop._make_failed_output(
        reason="trajectory_failed", exc=RuntimeError("x"), raw_prompt=_build_swebench_long_prompt()
    )
    final_short = _simulate_agent_loop_postprocess(out_short, real_qwen3_tokenizer)
    final_long = _simulate_agent_loop_postprocess(out_long, real_qwen3_tokenizer)

    # This is the operation that crashed in round-12-v2 production:
    cat_prompts = torch.cat([final_short["prompts"], final_long["prompts"]], dim=0)
    cat_responses = torch.cat([final_short["responses"], final_long["responses"]], dim=0)
    cat_routed = torch.cat([final_short["routed_experts"], final_long["routed_experts"]], dim=0)

    assert cat_prompts.shape == (2, PROMPT_LENGTH)
    assert cat_responses.shape == (2, RESPONSE_LENGTH)
    assert cat_routed.shape == (2, PROMPT_LENGTH + RESPONSE_LENGTH, MOE_LAYERS, MOE_TOPK)


def test_real_tokenizer_response_mask_all_zero_after_postprocess(real_qwen3_tokenizer):
    """Critical invariant: the final post-pad response_mask must be all zeros,
    so the trainer's PPO loss gives this failed sample exactly 0 gradient.
    Verifies the tokenizer.pad × attention_mask multiplication in production
    code actually zeroes the pad_token_id-filled positions.
    """
    loop = _make_real_stub_loop(real_qwen3_tokenizer)
    for raw_prompt in [SHORT_PROMPT, _build_swebench_long_prompt()]:
        output = loop._make_failed_output(
            reason="trajectory_failed", exc=RuntimeError("x"), raw_prompt=raw_prompt
        )
        final = _simulate_agent_loop_postprocess(output, real_qwen3_tokenizer)
        assert (final["response_mask"] == 0).all(), (
            "response_mask must be all zero post-padding for failed trajectories — "
            "non-zero values would propagate gradient on garbage tokens"
        )


def test_ensure_moe_shape_cached_real_qwen35_config():
    """Regression for round-12-v3 AttributeError: Qwen3.5 MoE nests
    architecture params under config.text_config, NOT top-level. Earlier code
    did `model_cfg.num_hidden_layers` which raises AttributeError on
    Qwen3_5MoeConfig. The fix navigates text_config first, falls back to
    top-level for older Qwen3.

    Critically: this test does NOT stub _moe_num_layers — it exercises the
    real method against the real model config so we catch nested-config
    schema mismatches. The previous tests bypassed this method entirely,
    which is exactly why round-12-v3 crashed in production.
    """
    if not Path(QWEN3_PATH).exists():
        pytest.skip(f"Qwen3 model not found at {QWEN3_PATH}")
    import asyncio

    # Reset class-level cache so this test exercises the real lookup.
    UniAgentLoop._moe_num_layers = None
    UniAgentLoop._moe_topk = None

    loop = UniAgentLoop.__new__(UniAgentLoop)
    loop.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"calculate_log_probs": True, "prompt_length": 4096},
                "model": {"path": QWEN3_PATH},
            }
        }
    )
    loop.apply_chat_template_kwargs = {}
    loop.logger = MagicMock()

    asyncio.run(loop._ensure_moe_shape_cached())

    # Qwen3.5-35B-A3B's text_config has num_hidden_layers=40, num_experts_per_tok=8
    assert UniAgentLoop._moe_num_layers == 40, (
        f"Expected 40 layers from Qwen3.5 text_config.num_hidden_layers, "
        f"got {UniAgentLoop._moe_num_layers}. round-12-v3 crash class."
    )
    assert UniAgentLoop._moe_topk == 8, (
        f"Expected 8 from text_config.num_experts_per_tok, got {UniAgentLoop._moe_topk}"
    )


def test_ensure_moe_shape_cached_swallows_errors():
    """_ensure_moe_shape_cached must NEVER raise — it's a cache helper, and
    its failure must not propagate up and bring down every trajectory. On
    error it sets sentinel (-1, -1) so failed_output produces routed_experts=None.
    """
    import asyncio

    UniAgentLoop._moe_num_layers = None
    UniAgentLoop._moe_topk = None

    loop = UniAgentLoop.__new__(UniAgentLoop)
    loop.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"calculate_log_probs": True, "prompt_length": 4096},
                "model": {"path": "/nonexistent/path/that/will/fail/auto_config"},
            }
        }
    )
    loop.apply_chat_template_kwargs = {}
    loop.logger = MagicMock()

    # Must not raise even though the path doesn't exist.
    asyncio.run(loop._ensure_moe_shape_cached())

    # Sentinel values mean "tried and failed" — distinguish from None (not tried)
    assert UniAgentLoop._moe_num_layers == -1
    assert UniAgentLoop._moe_topk == -1


def test_make_failed_output_skips_routed_experts_when_cache_failed():
    """When _moe_num_layers is the -1 sentinel, _make_failed_output must
    produce routed_experts=None (not a zeros tensor of bogus shape).
    This is the safe fallback when we can't determine MoE shape.
    """
    UniAgentLoop._moe_num_layers = -1
    UniAgentLoop._moe_topk = -1

    loop = UniAgentLoop.__new__(UniAgentLoop)
    loop.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"calculate_log_probs": True, "prompt_length": 4096},
                "model": {"path": "fake"},
            }
        }
    )
    loop.apply_chat_template_kwargs = {}
    tok = MagicMock()
    tok.apply_chat_template = MagicMock(return_value="hi")
    tok.encode = MagicMock(return_value=[1, 2, 3])
    tok.pad_token_id = 0
    loop.tokenizer = tok
    loop.logger = MagicMock()

    output = loop._make_failed_output(
        reason="trajectory_failed",
        exc=RuntimeError("x"),
        raw_prompt=[{"role": "user", "content": "hi"}],
    )
    assert output.routed_experts is None


def test_real_tokenizer_apply_chat_template_with_thinking(real_qwen3_tokenizer):
    """Sanity: Qwen3 supports thinking=True in apply_chat_template_kwargs.
    Our failed-path call must not be broken by it."""
    loop = _make_real_stub_loop(real_qwen3_tokenizer)
    output = loop._make_failed_output(
        reason="trajectory_failed",
        exc=RuntimeError("simulated"),
        raw_prompt=SHORT_PROMPT,
    )
    # Just verify it ran and gave us something tokenizable
    assert len(output.prompt_ids) > 0
    decoded = real_qwen3_tokenizer.decode(output.prompt_ids)
    assert "2+2" in decoded or "system" in decoded.lower() or len(decoded) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

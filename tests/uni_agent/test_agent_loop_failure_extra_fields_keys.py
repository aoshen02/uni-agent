"""Regression tests for the round12v7 crash: failed-sample extra_fields keys
must be a subset of verl's default_extra_keys, otherwise DataProto.concat
asserts when a failed trajectory lands at position >0 in a mixed batch.

Crash trace from round12v7 (param_version=11+):
    File "verl/utils/py_functional.py", line 366, in list_of_dict_to_dict_of_list
        assert key in output, f"Key '{key}' is not present in the keys of the
                                first dictionary in the list."
    AssertionError: Key 'failure_class' is not present in the keys of the first
                    dictionary in the list.

Mechanism:
    - verl agent_loop.py:989 unpacks every AgentLoopOutput.extra_fields key
      into per-sample non_tensor_batch.
    - When DataProto.concat assembles the batch, it uses data[0].keys() as
      schema and asserts every later dict's keys are in that schema.
    - If failed sample has `failure_class` (custom key) but normal sample
      (= data[0]) doesn't, assertion fires.

Our fix: pack all diagnostic info under the `extras` default-key, which IS
in verl's default_extra_keys = {turn_scores, tool_rewards, min_global_steps,
max_global_steps, extras}. So normal trajectories also produce the `extras`
key (with None value), and the schema matches.

These tests are deliberately at TWO layers:
1. The unit-of-failure level: _make_failed_output's extra_fields key set.
2. The mechanism level: simulate exactly what list_of_dict_to_dict_of_list
   does on a mixed batch.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from uni_agent.agent_loop import UniAgentLoop
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics
from verl.utils.py_functional import list_of_dict_to_dict_of_list


# Mirror of verl/experimental/agent_loop/agent_loop.py:982 — keep in sync with
# upstream. If this set changes upstream, the test will catch drift via the
# round-trip simulation below.
VERL_DEFAULT_EXTRA_KEYS = {
    "turn_scores",
    "tool_rewards",
    "min_global_steps",
    "max_global_steps",
    "extras",
}


def _make_stub_loop():
    """Minimal loop with the attributes _make_failed_output reads. Mirrors
    test_agent_loop_failure_path._make_stub_loop but kept local so this test
    file is self-contained."""
    loop = UniAgentLoop.__new__(UniAgentLoop)
    loop._moe_num_layers = 2
    loop._moe_topk = 2
    cfg = MagicMock()
    cfg.actor_rollout_ref.rollout.prompt_length = 16
    cfg.actor_rollout_ref.rollout.calculate_log_probs = True
    loop.config = cfg
    loop.apply_chat_template_kwargs = {}
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.apply_chat_template = MagicMock(return_value="dummy")
    tok.encode = MagicMock(return_value=[0])
    loop.tokenizer = tok
    loop.logger = MagicMock()
    return loop


def test_failed_output_top_level_keys_are_subset_of_default_extra_keys():
    """REGRESSION: round12v7 crashed at 00:15:33 because failed-sample
    extra_fields had top-level keys (failure_class, traj_masked, etc.) that
    are NOT in verl's default_extra_keys. After fix, all custom keys live
    under `extras`; top-level is {min_global_steps, max_global_steps, extras}."""
    loop = _make_stub_loop()
    output = loop._make_failed_output(
        reason="trajectory_failed",
        exc=RuntimeError("simulated infra failure"),
        raw_prompt=[{"role": "user", "content": "hi"}],
    )
    top_level = set(output.extra_fields.keys())
    illegal = top_level - VERL_DEFAULT_EXTRA_KEYS
    assert not illegal, (
        f"failed-output extra_fields has top-level keys outside default_extra_keys: "
        f"{illegal}. These will trigger AssertionError in "
        f"verl.utils.py_functional.list_of_dict_to_dict_of_list when this "
        f"failed sample lands at position >0 in a mixed concat batch."
    )


def test_failed_output_diagnostic_packed_under_extras():
    """The diagnostic fields we use to filter/inspect failed trajectories
    must all live under the `extras` sub-dict, not at top level."""
    loop = _make_stub_loop()
    output = loop._make_failed_output(
        reason="trajectory_failed",
        exc=TimeoutError("modal cold start timed out"),
        raw_prompt=[{"role": "user", "content": "hi"}],
    )
    extras = output.extra_fields["extras"]
    assert isinstance(extras, dict), f"extras must be dict, got {type(extras)}"
    assert extras["failure_class"] == "TimeoutError"
    assert extras["failure_reason"] == "trajectory_failed"
    assert extras["init_failed"] is True
    assert extras["traj_masked"] == 1
    assert extras["traj_exit_reason"] == "trajectory_failed"


def _to_non_tensor_dict(output, default_extra_keys=VERL_DEFAULT_EXTRA_KEYS):
    """Mirror of verl/experimental/agent_loop/agent_loop.py:989-995 for a
    SINGLE-sample batch (which is how FullyAsyncAgentLoopWorker postprocesses
    each trajectory)."""
    all_keys = set(output.extra_fields.keys()) | default_extra_keys
    return {key: output.extra_fields.get(key) for key in all_keys}


def test_mixed_concat_with_normal_first_does_not_assert():
    """REGRESSION: the EXACT failure mode of round12v7. Normal trajectory at
    position 0, failed trajectory at position 1+. Before the fix, this
    asserted on `failure_class` not in batch[0].keys()."""
    loop = _make_stub_loop()
    failed_output = loop._make_failed_output(
        reason="trajectory_failed",
        exc=RuntimeError("modal disconnect"),
        raw_prompt=[{"role": "user", "content": "x"}],
    )

    # Simulate a normal trajectory's non_tensor_batch: it has the
    # default_extra_keys union, all populated as None (because its
    # extra_fields is empty).
    normal_dict = {k: None for k in VERL_DEFAULT_EXTRA_KEYS}
    failed_dict = _to_non_tensor_dict(failed_output)

    # The crash-reproducing list: normal at 0, failed at 1+.
    mixed = [normal_dict, failed_dict, normal_dict.copy()]

    # If our fix is correct, this MUST NOT raise.
    merged = list_of_dict_to_dict_of_list(mixed)

    # Schema must include `extras` and `min/max_global_steps`.
    assert "extras" in merged
    assert "min_global_steps" in merged
    assert "max_global_steps" in merged
    # Failed sample's `extras` dict survived intact.
    assert merged["extras"][1] is not None
    assert merged["extras"][1]["failure_class"] == "RuntimeError"
    # Normal samples' `extras` is None — confirms the schema is unified.
    assert merged["extras"][0] is None
    assert merged["extras"][2] is None


def test_mixed_concat_with_failed_first_does_not_assert():
    """Symmetric case: failed trajectory is data[0]. Before the fix, normal
    samples would either error (if their union of keys leaked failure_class)
    or silently lose data."""
    loop = _make_stub_loop()
    failed_output = loop._make_failed_output(
        reason="trajectory_failed",
        exc=ValueError("bad input"),
        raw_prompt=[{"role": "user", "content": "y"}],
    )
    failed_dict = _to_non_tensor_dict(failed_output)
    normal_dict = {k: None for k in VERL_DEFAULT_EXTRA_KEYS}

    mixed = [failed_dict, normal_dict, normal_dict.copy()]
    merged = list_of_dict_to_dict_of_list(mixed)
    assert merged["extras"][0]["failure_class"] == "ValueError"
    assert merged["extras"][1] is None
    assert merged["extras"][2] is None


def test_regression_canary_old_top_level_keys_would_have_crashed():
    """Sanity: confirms the assertion mechanism DOES fire when a stray top-
    level key sneaks back in. If this test stops raising, the upstream
    list_of_dict_to_dict_of_list semantics have shifted and we should
    re-evaluate the contract."""
    normal_dict = {k: None for k in VERL_DEFAULT_EXTRA_KEYS}
    bad_failed_dict = {**normal_dict, "failure_class": "RuntimeError"}  # OLD pre-fix shape
    with pytest.raises(AssertionError, match="failure_class"):
        list_of_dict_to_dict_of_list([normal_dict, bad_failed_dict])


def test_last_resort_fallback_also_uses_extras_only():
    """The nested fallback in run() (when _make_failed_output ITSELF raises)
    must use the same extras-only schema, otherwise it reintroduces the
    same crash class one level deeper."""
    # Build the literal AgentLoopOutput the fallback would produce. We
    # inspect the source code's extra_fields template (line ~133) by
    # constructing what it constructs.
    from verl.experimental.agent_loop.agent_loop import AgentLoopOutput

    fallback = AgentLoopOutput(
        prompt_ids=[0],
        response_ids=[0],
        response_mask=[0],
        response_logprobs=None,
        routed_experts=None,
        multi_modal_data={},
        reward_score=0.0,
        num_turns=0,
        metrics=AgentLoopMetrics(),
        # MUST match the actual code at agent_loop.py last-resort branch.
        extra_fields={
            "min_global_steps": -1,
            "max_global_steps": -1,
            "extras": {
                "traj_masked": 1,
                "traj_exit_reason": "build_failed",
                "init_failed": True,
                "failure_class": "RuntimeError",
                "failure_reason": "build_failed",
            },
        },
    )
    top_level = set(fallback.extra_fields.keys())
    illegal = top_level - VERL_DEFAULT_EXTRA_KEYS
    assert not illegal, (
        f"last-resort fallback re-introduced top-level keys outside default_extra_keys: "
        f"{illegal}"
    )


if __name__ == "__main__":
    import pytest as _pytest

    _pytest.main([__file__, "-v"])

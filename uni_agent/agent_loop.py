import asyncio
import json
import pickle
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from uni_agent.async_logging import add_file_handler, get_logger
from uni_agent.interaction import (
    AgentChatModel,
    AgentEnv,
    AgentEnvConfig,
    AgentInteraction,
    ToolsManager,
    ToolsManagerConfig,
)
from uni_agent.reward import load_reward_spec
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput
from verl.experimental.agent_loop.utils import resolve_config_path


class UniAgentLoop(AgentLoopBase):
    _semaphore: asyncio.Semaphore | None = None

    # Cached MoE shape metadata, populated lazily on first run() so init-failure
    # path can produce a routed_experts tensor whose shape matches the vLLM rollout
    # output, keeping DataProto.concat happy when failed and normal samples mix.
    _moe_num_layers: int | None = None
    _moe_topk: int | None = None

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        config_dict = self._init_config(sampling_params, **kwargs)
        self.mask_abnormal_exit_traj = config_dict.get("mask_abnormal_exit_traj", False)
        global_concurrent = config_dict.get("concurrency", 512)
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers
        worker_concurrent = max(global_concurrent // num_workers, 1)
        if UniAgentLoop._semaphore is None:
            UniAgentLoop._semaphore = asyncio.Semaphore(worker_concurrent)

        self.run_id = str(uuid.uuid4())
        self.logger = get_logger("agent-loop", run_id=self.run_id)
        # init chat model, tools manager and environment
        self.chat_model = self._init_chat_model(config_dict["model"])
        self.tools_manager = self._init_tools_manager(
            tools_config_list=config_dict["tools"],
            parser=config_dict.get("tool_parser", "qwen3_coder"),
        )
        self.env = self._init_env(config_dict["env"])
        self.output_dir = Path(config_dict["log_dir"]) / self.run_id
        self.interaction = AgentInteraction(
            run_id=self.run_id,
            env=self.env,
            model=self.chat_model,
            tools_manager=self.tools_manager,
            messages=list(kwargs["raw_prompt"]),
            **config_dict["interaction"],
        )
        if config_dict["reward"] is not None:
            reward_config = {
                **config_dict["reward"],
                "run_id": self.run_id,
                "env": self.env,
            }
            self.reward_spec = load_reward_spec(reward_config)
        else:
            self.reward_spec = None

        add_file_handler(self.output_dir / "run.log", self.run_id)

        self.logger.info(f"model name: {self.config.actor_rollout_ref.model.path}")
        self.logger.info(f"sampling_params: {sampling_params}")
        self.logger.info(f"environment config: {config_dict['env']}")
        self.logger.info(f"tools config: {config_dict['tools']}")
        self.logger.info(f"interaction config: {config_dict['interaction']}")
        self.logger.info(f"mask_abnormal_exit_traj: {self.mask_abnormal_exit_traj}")
        self.logger.info(f"output_dir: {self.output_dir}")

        async with self._semaphore:
            try:
                await self._ensure_moe_shape_cached()
                await self.env.start()
                interaction_result = await self._run_interaction()
                # interaction environment should be visible to the reward spec
                if self.reward_spec is not None:
                    reward_score, _ = await self.reward_spec.compute_reward(
                        interaction_result=interaction_result,
                    )
                    interaction_result["reward_score"] = reward_score
                else:
                    self.logger.warning("No reward spec is provided, reward score will be set to -100")
                    interaction_result["reward_score"] = -100

                self._save_interaction_result(interaction_result)
                return self.convert_to_agent_output(interaction_result)
            except Exception as exc:
                # Bare Exception is intentional: this is the outer boundary that must not let
                # any stochastic infra failure (Modal cold-start, swerex pexpect timeout, PyPI
                # mirror hiccup, reward eval crash) bubble up into verl _fit_validate and kill
                # the whole training job. Failed trajectory becomes a reward=0, fully-masked
                # sample so it contributes no gradient and only mildly tilts the GRPO baseline.
                try:
                    return self._make_failed_output(
                        reason="trajectory_failed",
                        exc=exc,
                        raw_prompt=kwargs["raw_prompt"],
                    )
                except Exception as build_exc:
                    # Last-resort: even building a failed AgentLoopOutput failed (e.g. tokenizer
                    # corruption, model config unreadable, Pydantic schema change). Return the
                    # absolutely minimal valid output: 1 pad token, no MoE, no logprobs.
                    # Trainer downstream will skip the rollout_log_probs / routed_experts keys
                    # when inputs[0] sees None — this risks a KeyError at DataProto.concat if
                    # this minimal output is the FIRST in a batch, but that is strictly safer
                    # than killing the whole training job.
                    # LOGURU GOTCHA: brace-in-exc-repr can crash loguru's
                    # internal .format(). Precompute the message and pass
                    # via "{}" template (safe positional substitution).
                    _msg = (
                        f"[traj-fail-buildfail] {type(build_exc).__name__}: {build_exc} "
                        f"(original exc: {type(exc).__name__}: {exc})"
                    )
                    self.logger.opt(exception=True).error("{}", _msg)
                    return AgentLoopOutput(
                        prompt_ids=[0],
                        response_ids=[0],
                        response_mask=[0],
                        response_logprobs=None,
                        routed_experts=None,
                        multi_modal_data={},
                        reward_score=0.0,
                        num_turns=0,
                        metrics=AgentLoopMetrics(),
                        # extra_fields keys MUST be a subset of upstream verl's
                        # default_extra_keys = {turn_scores, tool_rewards,
                        # min_global_steps, max_global_steps, extras}, otherwise
                        # DataProto.concat in py_functional.list_of_dict_to_dict_of_list
                        # asserts when this failed sample lands at position >0 in
                        # a batch whose data[0] is a normal trajectory (KeyError
                        # crash observed in round12v7 at param_version=11+).
                        # Diagnostic info goes under "extras".
                        extra_fields={
                            "min_global_steps": -1,
                            "max_global_steps": -1,
                            "extras": {
                                "traj_masked": 1,
                                "traj_exit_reason": "build_failed",
                                "init_failed": True,
                                "failure_class": type(build_exc).__name__,
                                "failure_reason": "build_failed",
                            },
                        },
                    )
            finally:
                try:
                    await self.env.close()
                except Exception as close_exc:
                    # finally-block: if this logger crashes (loguru brace bug),
                    # the exception replaces the function's return value and
                    # kills the rollouter actor. Use safe "{}" template.
                    self.logger.warning(
                        "{}",
                        f"env.close swallowed: {type(close_exc).__name__}: {close_exc}",
                    )

    async def _ensure_moe_shape_cached(self) -> None:
        """Read num_hidden_layers / num_experts_per_tok from model config once per worker.

        Needed so _make_failed_output can produce a routed_experts tensor with the same
        shape as a real vLLM rollout — otherwise DataProto.concat in
        verl/experimental/fully_async_policy/detach_utils.py:127 crashes the whole batch
        as soon as one failed sample lands in the same MQ pull as normal samples.

        Qwen3.5 MoE puts architecture params under `text_config` (nested), older Qwen3
        keeps them at top-level. We probe both. This method MUST NOT raise — if
        config navigation fails, leave the cache as sentinel (-1) and let
        _make_failed_output fall back to routed_experts=None. That fallback is
        only safe when ALL trajectories in a batch fail homogeneously, but is
        strictly better than killing every trajectory on a config schema mismatch.
        """
        cls = type(self)
        if cls._moe_num_layers is not None:
            return
        try:
            from transformers import AutoConfig

            model_path = self.config.actor_rollout_ref.model.path
            # Block in a thread so transformers' file I/O doesn't stall the event loop.
            model_cfg = await asyncio.to_thread(
                AutoConfig.from_pretrained, model_path, trust_remote_code=True
            )
            # Probe nested text_config first (Qwen3.5 MoE), fall back to top-level (Qwen3).
            text_cfg = getattr(model_cfg, "text_config", None) or model_cfg
            num_layers = (
                int(getattr(text_cfg, "num_hidden_layers", 0))
                or int(getattr(model_cfg, "num_hidden_layers", 0))
            )
            topk = (
                int(getattr(text_cfg, "num_experts_per_tok", 0))
                or int(getattr(model_cfg, "num_experts_per_tok", 0))
            )
            if num_layers <= 0 or topk <= 0:
                raise RuntimeError(
                    f"MoE shape extraction failed: num_layers={num_layers} topk={topk} "
                    f"(checked text_config + top-level; config type={type(model_cfg).__name__})"
                )
            cls._moe_num_layers = num_layers
            cls._moe_topk = topk
            self.logger.info(f"cached MoE shape: num_layers={num_layers} topk={topk}")
        except Exception as exc:
            self.logger.warning(
                "{}",
                (
                    f"_ensure_moe_shape_cached fallback (failed_output.routed_experts will be None): "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
            # Sentinel: tried-and-failed. Distinguish from None (not-tried-yet)
            # so we don't keep retrying AutoConfig.from_pretrained per trajectory.
            cls._moe_num_layers = -1
            cls._moe_topk = -1

    def _make_failed_output(self, reason: str, exc: BaseException, raw_prompt) -> AgentLoopOutput:
        """Build a shape-compatible zero-trajectory output for the init/mid-traj failure path.

        Shape constraints (see verl/experimental/agent_loop/agent_loop.py:932-938 and
        verl/experimental/fully_async_policy/detach_utils.py:127):

        - prompt_ids: real tokenized prompt (so _agent_loop_postprocess left-pads correctly)
        - response_ids: [pad_id] length=1 (will right-pad to response_length)
        - response_mask: [0] → after pad all-zero → trainer treats as 0 gradient
        - response_logprobs: [0.0] when calculate_log_probs=True, else None.
            Must match the normal-path key presence or DataProto.concat KeyErrors.
        - routed_experts: zeros (1, num_layers, topk) when MoE rollout writes them,
            else None. Same key-presence rule as response_logprobs.
        - reward_score: 0.0 (not None) so verl _compute_score does not try to invoke
            the external reward worker on a sample that has no real trajectory.
        - metrics: AgentLoopMetrics() — required field, must be the dataclass type.
        """
        rollout_cfg = self.config.actor_rollout_ref.rollout
        calculate_log_probs = bool(getattr(rollout_cfg, "calculate_log_probs", False))
        max_prompt_length = int(rollout_cfg.prompt_length)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        # Tokenize the prompt for downstream padding. tokenize=True can return
        # BatchEncoding on some tokenizers (Qwen3), so go through tokenize=False
        # + encode() to guarantee a plain list[int]. Defensive try/except
        # ensures even a tokenizer crash does not escape this method — the
        # whole point of Layer 2 is that failure path itself never fails.
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                list(raw_prompt),
                add_generation_prompt=True,
                tokenize=False,
                **self.apply_chat_template_kwargs,
            )
            prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            if not isinstance(prompt_ids, list):
                prompt_ids = list(prompt_ids)
            if not prompt_ids:
                prompt_ids = [pad_id]
        except Exception as tok_exc:
            self.logger.warning(
                "{}",
                f"_make_failed_output tokenizer fallback: {type(tok_exc).__name__}: {tok_exc}",
            )
            prompt_ids = [pad_id]

        # CRITICAL: truncate to max_prompt_length so downstream
        # _agent_loop_postprocess tokenizer.pad(max_length=prompt_length) produces
        # the SAME shape (1, max_prompt_length) as the normal path. tokenizer.pad
        # only pads up; it does not truncate down. Without this, a long prompt
        # (e.g. SWE-Bench's 8954-token problem statement) makes the failed
        # sample's tensor shape (1, 8954) and the normal samples' (1, 4096),
        # crashing torch.cat at agent_loop.py:925. Matches the truncation in
        # convert_to_agent_output:192 for consistency.
        if len(prompt_ids) > max_prompt_length:
            prompt_ids = prompt_ids[:max_prompt_length]

        response_logprobs = [0.0] if calculate_log_probs else None
        routed_experts = None
        # MUST match the normal-path schema in convert_to_agent_output:
        #   rollout_cache.get("routed_experts") is None whenever
        #   enable_rollout_routing_replay=False (vLLM never returns routed_experts
        #   via vllm_async_server.py:347-348, sglang via async_sglang_server.py:483).
        # If we filled zeros here while the normal path returns None, the batch
        # becomes heterogeneous and verl agent_loop.py:934's `inputs[0]` sentinel
        # gates on whichever sample lands at index 0 → torch.cat sees mixed
        # None/Tensor and crashes with "expected Tensor as element N, got NoneType".
        # Only emit zeros when (a) routing replay is actually on AND (b) we have
        # cached MoE shape from _ensure_moe_shape_cached.
        routing_replay_on = bool(getattr(rollout_cfg, "enable_rollout_routing_replay", False))
        if (
            routing_replay_on
            and self._moe_num_layers is not None
            and self._moe_num_layers > 0
            and self._moe_topk is not None
            and self._moe_topk > 0
        ):
            routed_experts = np.zeros(
                (1, self._moe_num_layers, self._moe_topk), dtype=np.int64
            )

        # LOGURU GOTCHA: see interaction.py MaxTokenExceededError handler comment.
        # f"{exc}" can produce literal '{'/'}' (e.g. swerex config repr) which
        # makes loguru's internal .format() crash. Use "{}" template + positional
        # arg so the message string is never re-parsed.
        _msg = f"[traj-fail] {reason}: {type(exc).__name__}: {exc}"
        self.logger.opt(exception=True).error("{}", _msg)

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=[pad_id],
            response_mask=[0],
            response_logprobs=response_logprobs,
            routed_experts=routed_experts,
            multi_modal_data={},
            reward_score=0.0,
            num_turns=0,
            metrics=AgentLoopMetrics(),
            # extra_fields keys MUST be a subset of upstream verl's
            # default_extra_keys = {turn_scores, tool_rewards, min_global_steps,
            # max_global_steps, extras}, otherwise DataProto.concat in
            # py_functional.list_of_dict_to_dict_of_list asserts when this
            # failed sample lands at position >0 in a batch whose data[0] is a
            # normal trajectory (KeyError crash observed in round12v7 at
            # param_version=11+). Diagnostic info goes under "extras".
            extra_fields={
                "min_global_steps": -1,
                "max_global_steps": -1,
                "extras": {
                    "traj_masked": 1,
                    "traj_exit_reason": "trajectory_failed",
                    "init_failed": True,
                    "failure_class": type(exc).__name__,
                    "failure_reason": reason,
                },
            },
        )

    async def _run_interaction(self) -> dict:
        # tools schemas should be visible to the model
        # to generate correct tool call format in response
        self.chat_model.set_tools_schemas(self.tools_manager.tools_schemas)
        # tool should be runnable in the environment
        await self.env.install_tools(self.tools_manager.tools)

        interaction_result = await self.interaction.run()
        interaction_result["metrics"] = dict(interaction_result.get("rollout_cache", {}).get("metrics", {}))
        return interaction_result

    def _save_interaction_result(self, interaction_result: dict):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # rollout_cache: binary pickle for fast I/O (no readability needed)
        with (self.output_dir / "rollout_cache.pkl").open("wb") as f:
            pickle.dump(interaction_result["rollout_cache"], f, protocol=pickle.HIGHEST_PROTOCOL)
        # rest: readable JSON
        save_content = {
            "trajectory": [s.model_dump() for s in interaction_result["trajectory"]],
            "execution_time": interaction_result["execution_time"],
            "messages": interaction_result["messages"],
            "metrics": interaction_result.get("metrics", {}),
        }
        (self.output_dir / "interaction_result.json").write_text(
            json.dumps(save_content, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def _init_config(self, sampling_params: dict[str, Any], **kwargs):
        # load config from file
        agent_loop_config_path = self.config.actor_rollout_ref.rollout.agent.agent_loop_config_path
        assert agent_loop_config_path is not None, "agent_loop_config_path is None"
        resolved_path = resolve_config_path(agent_loop_config_path)
        config_dict = yaml.safe_load(Path(resolved_path).read_text())[0]
        # model config
        rollout_config = self.config.actor_rollout_ref.rollout
        max_model_len = (
            rollout_config.max_model_len
            if rollout_config.max_model_len is not None
            else rollout_config.prompt_length + rollout_config.response_length
        )
        model_config = {
            "client": self.server_manager,
            "tokenizer": self.tokenizer,
            "max_model_len": max_model_len,
            "sampling_params": sampling_params,
        }
        config_dict["model"] = model_config
        # env config (optionally override sample-wise image / post_setup_cmd)
        env_kwargs = kwargs.get("tools_kwargs", {}).get("env") or {}
        if "image" in env_kwargs:
            config_dict["env"]["deployment"]["image"] = env_kwargs["image"]
        if "post_setup_cmd" in env_kwargs:
            config_dict["env"]["post_setup_cmd"] = env_kwargs["post_setup_cmd"]
        # reward module
        reward_config = config_dict.get("reward", {})
        reward_config.update(kwargs["tools_kwargs"].get("reward", {}))
        config_dict["reward"] = reward_config if reward_config else None
        return config_dict

    def _init_chat_model(self, config_dict: dict) -> AgentChatModel:
        chat_model = AgentChatModel(**config_dict)
        return chat_model

    def _init_tools_manager(self, tools_config_list: list[dict], parser: str = "qwen3_coder") -> ToolsManager:
        tools_manager_config = ToolsManagerConfig(tools=tools_config_list, parser=parser)
        return ToolsManager(tools_manager_config=tools_manager_config)

    def _init_env(self, config_dict: dict) -> AgentEnv:
        env_config = AgentEnvConfig(**config_dict)
        return AgentEnv(run_id=self.run_id, env_config=env_config)

    def convert_to_agent_output(self, interaction_result: dict) -> AgentLoopOutput:
        rollout_cache = interaction_result["rollout_cache"]
        reward_score = interaction_result.get("reward_score", None)

        num_turns = len(interaction_result["trajectory"])
        self.logger.info(f"num_turns: {num_turns}")

        prompt_ids = rollout_cache["prompt_ids"]
        traj_exit_reason = interaction_result["trajectory"][-1].exit_reason if num_turns > 0 else "unknown"
        should_mask_traj = self.mask_abnormal_exit_traj and traj_exit_reason != "finished"
        traj_masked = int(should_mask_traj)

        if should_mask_traj:
            response_mask = [0] * len(rollout_cache["response_mask"])
        else:
            response_mask = rollout_cache["response_mask"]
        # Guard: empty response_mask means model.query() on step 1 raised before
        # model.py:119 could append the response tokens. Most common trigger is
        # MaxTokenExceededError at model.py:95-99 (initial prompt+tool_schemas
        # already >= max_model_len). Without this guard the slicing below does
        # `prompt_ids[:-0]` which is `prompt_ids[:0]` = [] and the empty list
        # flows into verl _agent_loop_postprocess → tokenizer.pad → HF
        # early-return path returns a plain list (not Tensor) → verl's `.dim()`
        # call crashes the FullyAsyncRollouter with:
        #   AttributeError: 'list' object has no attribute 'dim'
        # Raising here routes to the outer try/except in run() →
        # _make_failed_output, which produces a shape-safe zero-trajectory
        # output (real tokenized prompt truncated to prompt_length,
        # response_ids=[pad_id], response_mask=[0]).
        if not response_mask:
            # Per-step exit-reason chain lets us tell at a glance whether step 1
            # broke (the only way response_mask can stay []) and which sub-cause:
            #   trajectory=[step1=token_limit]   → MaxTokenExceededError at model.py:95-99
            #                                      (prompt+tools >= max_model_len before
            #                                      generate() ran). Inspect run-start
            #                                      `headroom` value in run.log.
            #   trajectory=[step1=unknown_error] → other exception inside step() before
            #                                      model.py:119 could grow the mask.
            #                                      Inspect `[step1] unknown_error:
            #                                      <type>` log + traceback right above.
            #                                      Common: vLLM RPC/socket error,
            #                                      asyncio timeout, env start race.
            #   any other final exit_reason     → shouldn't be reachable (would imply
            #                                      response_mask grew then got truncated
            #                                      to [] somewhere — bug).
            traj_summary = ",".join(
                f"step{s.step_idx}={s.exit_reason}"
                for s in interaction_result["trajectory"]
            )
            # chat_model is always set by self._init_chat_model() in run() before
            # this method runs; max_model_len is always set in AgentChatModel.__init__
            # by agent_loop.py:359-364. Direct access so any future regression
            # surfaces as a loud AttributeError instead of a silent "?".
            max_model_len = self.chat_model.max_model_len
            self.logger.error(
                f"[convert_to_agent_output] empty response_mask — "
                f"exit_reason={traj_exit_reason} num_turns={num_turns} "
                f"trajectory=[{traj_summary}] "
                f"prompt_ids_len={len(prompt_ids)} max_model_len={max_model_len} "
                f"raw_response_mask_len={len(rollout_cache['response_mask'])} "
                f"should_mask_traj={should_mask_traj} "
                f"→ raising to route to _make_failed_output"
            )
            raise RuntimeError(
                f"empty response_mask (exit_reason={traj_exit_reason}, "
                f"num_turns={num_turns}, trajectory=[{traj_summary}]); "
                f"routing to _make_failed_output"
            )
        response_logprobs = rollout_cache.get("response_logprobs") or []
        routed_experts = rollout_cache.get("routed_experts")
        metrics = interaction_result.get("metrics", rollout_cache.get("metrics", {}))
        # extra_fields keys MUST be a subset of upstream verl's
        # default_extra_keys = {turn_scores, tool_rewards, min_global_steps,
        # max_global_steps, extras}. Diagnostic keys (traj_masked,
        # traj_exit_reason) go under "extras" to match the failure-path schema
        # at _make_failed_output / build_failed (lines 144-154, 330-340).
        # Otherwise DataProto.concat → list_of_dict_to_dict_of_list asserts
        # when a normal sample (top-level traj_masked) and a failed sample
        # (traj_masked nested under extras) land in the same batch. Same
        # regression as round12v7 (2026-05-18), just from the success side.
        extra_fields = dict(rollout_cache.get("extra_fields") or {})
        extras = dict(extra_fields.get("extras") or {})
        extras["traj_masked"] = traj_masked
        extras["traj_exit_reason"] = traj_exit_reason
        extra_fields["extras"] = extras
        response_ids = prompt_ids[-len(response_mask) :]
        prompt_ids = prompt_ids[: -len(response_mask)]

        max_prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        max_response_length = self.config.actor_rollout_ref.rollout.response_length

        if len(prompt_ids) > max_prompt_length:
            prompt_ids = prompt_ids[:max_prompt_length]
            self.logger.warning(
                f"prompt_ids length {len(prompt_ids)} exceeds max_prompt_length {max_prompt_length} "
                "truncate prompt_ids length"
            )
        if len(response_ids) > max_response_length:
            response_ids = response_ids[:max_response_length]
            response_mask = response_mask[:max_response_length]
            response_logprobs = response_logprobs[:max_response_length]
            self.logger.warning(
                f"response_ids length {len(response_ids)} exceeds max_response_length {max_response_length} "
                "truncate response_ids length"
            )

        self.logger.info(f"prompt_ids length: {len(prompt_ids)}")
        self.logger.info(f"response_ids length: {len(response_ids)}")
        self.logger.info(f"reward_score: {reward_score}")
        response_logprobs = response_logprobs if response_logprobs else None
        if routed_experts is not None:
            routed_experts = routed_experts[: len(prompt_ids) + len(response_ids)]

        multi_modal_data = {}
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            routed_experts=routed_experts,
            multi_modal_data=multi_modal_data,
            reward_score=reward_score,
            num_turns=num_turns,
            metrics=metrics,
            extra_fields=extra_fields,
        )

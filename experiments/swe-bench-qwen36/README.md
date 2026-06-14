# SWE-bench Verified 复现实验 — uni-agent 侧改动

## 实验概览

| 项目 | 值 |
|------|-----|
| **模型** | Qwen3.6-35B-A3B (vLLM serving, TP=4) |
| **数据集** | SWE-bench Verified 500 |
| **结果** | uniagent 模式 71.3% (355/498), 对齐官方 71.6% |
| **采样** | temperature=1.0, top_p=0.95 (官方设定) |
| **基础设施** | 4 节点 × 4 GPU, Modal sandbox, Cloudflare quick tunnel |

## 本分支改了什么

基于 `verl-project/uni-agent` main (`a563cc6`), 共 10 个文件:

### 核心代码 (4 文件)

| 文件 | 改动说明 |
|------|----------|
| `uni_agent/agent_loop.py` | failure path 安全化: DataProto.concat 的 extra_fields key-set 必须一致 (success/failure 都输出相同 key); trajectory 异常时降级为 ABORTED 而非 crash 整批 |
| `uni_agent/interaction/interaction.py` | loguru `{}` brace-safe logging: f-string 里含 JSON/dict 的 `{}` 会被 loguru 当占位符解析 → 改用 `logger.opt(depth=...)` 或 escape |
| `uni_agent/interaction/model.py` | `_get_new_message_ids()` 前缀 assert → LCP 兜底 (见下方"已知问题") |
| `uni_agent/deployment/modal/deployment.py` | Modal sandbox 配置调整 |

### 测试 (5 文件)

| 文件 | 说明 |
|------|------|
| `tests/uni_agent/test_agent_loop_failure_path.py` | failure path 降级逻辑 |
| `tests/uni_agent/test_agent_loop_failure_extra_fields_keys.py` | extra_fields key-set 一致性 |
| `tests/uni_agent/test_agent_loop_failure_dataproto_concat.py` | DataProto.concat 边界情况 |
| `tests/uni_agent/test_agent_loop_failure_real_tokenizer.py` | 真实 tokenizer 下的 failure path |
| `tests/deployment/test_modal_starting_limiter.py` | Modal 冷启动限流 |

### 其他

| 文件 | 说明 |
|------|------|
| `uni_agent/__init__.py` | 版本号 bump |

---

## 已知问题: model.py 的 `<think>` 块 LCP 兜底

### 问题

`_get_new_message_ids()` 用"减法"提取新消息的 token:

```
new_tokens = tokenize(历史 + 新消息) - tokenize(历史)
```

这要求 `tokenize(历史)` 是 `tokenize(历史 + 新消息)` 的**精确前缀**。
但 Qwen3.5/3.6 的 chat template 会给**最后一句 assistant** 注入空的 `<think>\n\n</think>\n\n>` 推理块,
该 turn 变为历史句(后面接了新消息)后模板把 `<think>` 块删掉, 前缀不再相等:

```
base (mock assistant 是最后一句):
  ... <|im_start|>assistant
      <think>\n\n</think>\n\n     ← 4 个 token (id: 248068, 271, 248069, 271)
      mock assistant<|im_end|>

full (mock assistant 变历史句):
  ... <|im_start|>assistant
      mock assistant<|im_end|>    ← 没有 <think> 块了
      \n<|im_start|>user\n...
```

原版 `assert` 会在这里崩。当前分支改成了 LCP (最长公共前缀) 兜底:
对不上时找到公共前缀截断点, 从那里开始切。

### 兜底的副作用

LCP 切出来的"新 token"会包含上一句 assistant turn 的残渣:

```
期望: [新 user 消息的 token]
实际: [旧 assistant 残渣 "mock assistant<|im_end|>"] + [新 user 消息的 token]
```

- **Smoke test**: 无影响, 能跑通 ✅
- **训练**: 旧 turn token 会污染新 turn 的 loss mask / response 抽取 ⚠️

### 复现

```bash
# CPU 离线即可, 用 Qwen3.5 同家族小模型的 tokenizer
pip install transformers
MODEL=Qwen/Qwen3.5-0.8B python experiments/swe-bench-qwen36/probe_prefix.py
```

输出会显示 Case A (user 消息) `exact_prefix_holds = False`,
Case B (tool 消息) `exact_prefix_holds = True`。

### 正确修法方向 (未实现)

1. 让历史 assistant turn 也补一个空 `<think>` 对齐
2. 工具结果改走 `tool` role (Case B 不会触发此问题)
3. 直接在 template 层面修复 (上游 Qwen tokenizer)

当前兜底仅用于 smoke test, 训练正确性需要上述正确修法之一。

---

## 复现步骤

### 前置条件

- 4 节点, 每节点 4 GPU
- 共享存储 (NFS/Lustre)
- Docker + CUDA
- Modal 账号 (SWE-bench sandbox)
- Cloudflare (tunnel)

### 1. 准备代码

```bash
# uni-agent (本分支)
git clone git@github.com:aoshen02/uni-agent.git -b benchmark/semianalysis-verl
cd uni-agent

# vime (实验分支 — 包含 adapter/rollout/sandbox 改动)
git clone git@github.com:aoshen02/vime.git -b experiment/swe-bench-qwen36-eval
```

### 2. 安装依赖

```bash
# uni-agent
cd uni-agent
pip install -e .

# verl (训练框架, uni-agent 依赖)
pip install verl==0.3.0  # 或 git+https://github.com/volcengine/verl@460ccf3c
```

### 3. 准备模型

```bash
huggingface-cli download Qwen/Qwen3.6-35B-A3B --local-dir /shared/models/Qwen3.6-35B-A3B
```

### 4. 运行

详见 [vime 侧 README](https://github.com/aoshen02/vime/tree/experiment/swe-bench-qwen36-eval/experiments/swe-bench-qwen36) 的容器启动和运行步骤。

**简要流程:**

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  vime train │────▶│ adapter shim │────▶│  vLLM (TP=4)    │
│  (ray job)  │     │ (:18001)     │     │  Qwen3.6-35B    │
│             │     │ Anthropic →  │     │                 │
│             │     │ vLLM generate│     │                 │
└──────┬──────┘     └──────────────┘     └─────────────────┘
       │
       │  per trajectory
       ▼
┌─────────────────┐
│  Modal sandbox  │
│  (SWE-bench     │
│   test env)     │
│                 │
│  uni-agent loop │
│  str_replace +  │
│  bash + submit  │
└─────────────────┘
```

1. vime 的 `train.py --debug-rollout-only` 起 ray job, 分发 500 个 SWE-bench instance
2. 每个 instance 启动一个 Modal sandbox + uni-agent loop
3. uni-agent 通过 adapter shim 调 vLLM 生成, 在 sandbox 里执行工具
4. 提交 patch 后 SWE-bench harness 跑测试, 返回 reward

### 5. 查看结果

```bash
ray job logs <job_id> | grep "reward=" | \
  awk -F'reward=' '{r=int($2); if(r>0) s++; t++} END {printf "%d/%d = %.1f%%\n", s, t, s/t*100}'
```

## 实验结果

| 配置 | 模式 | solved% | graded | 备注 |
|------|------|---------|--------|------|
| uniagent/9step, t=1.0 p=0.95 | uniagent | **72.2%** | 496 | sglang parser |
| uniagent/9step, t=1.0 p=0.95 | uniagent | **71.3%** | 498 | vLLM parser, 验证通过 |
| claude_code/9step | claude_code | 56.2% | 496 | 基线 |

**结论**: uniagent harness (+13pt) 和官方采样参数 t=1.0/p=0.95 (+2pt) 是决定性杠杆。

## 配套仓库

- **vime 实验分支**: [`aoshen02/vime@experiment/swe-bench-qwen36-eval`](https://github.com/aoshen02/vime/tree/experiment/swe-bench-qwen36-eval)
  - 包含 adapter shim, rollout 降级, sandbox uniagent 模式, system-merge 等

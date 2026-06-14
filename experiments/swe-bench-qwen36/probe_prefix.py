"""
probe_prefix.py — 验证 Qwen3.5 chat template 的 <think> 块
对 _get_new_message_ids() 前缀假设的影响。

用法 (CPU 离线即可):
    MODEL=/path/to/Qwen3.5-0.8B python probe_prefix.py

结论: Qwen3.5 模板给"最后一句 assistant"注入空 <think>\\n\\n</think>\\n\\n 块,
该 turn 变为历史句后删除, 导致 base 不再是 full 的精确前缀。
详见 experiments/swe-bench-qwen36/README.md "已知问题" 章节。
"""

import os
from transformers import AutoTokenizer

MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-0.8B")
tok = AutoTokenizer.from_pretrained(MODEL)


def norm(x):
    """normalize_token_ids 的简化版 (同 verl.utils.tokenizer)"""
    if isinstance(x, dict) and "input_ids" in x:
        x = x["input_ids"]
    elif hasattr(x, "input_ids"):
        x = x.input_ids
    if x and isinstance(x[0], list):
        x = x[0]
    return list(x)


tools = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}]


def run(label, new_messages, tools_schemas):
    # 固定的 3 句 mock 脚手架 (和 model.py _get_new_message_ids 一致)
    messages = [
        {"role": "system", "content": "mock system"},
        {"role": "user", "content": "mock user"},
        {"role": "assistant", "content": "mock assistant"},
    ]
    base = norm(tok.apply_chat_template(
        messages, tokenize=True, tools=tools_schemas))
    full = norm(tok.apply_chat_template(
        messages + new_messages, add_generation_prompt=True,
        tokenize=True, tools=tools_schemas))

    # 截断 base 到最后一个 eos (和 model.py 一致)
    eos = tok.eos_token_id
    cut = 0
    for i in range(len(base) - 1, -1, -1):
        if base[i] == eos:
            cut = i + 1
            break
    base = base[:cut]

    exact_prefix = full[:len(base)] == base
    common = 0
    for i in range(min(len(base), len(full))):
        if base[i] != full[i]:
            break
        common = i + 1

    print(f"\n===== {label} =====")
    print(f"eos_id={eos}  len(base)={len(base)}  len(full)={len(full)}")
    print(f"exact_prefix_holds = {exact_prefix}   common_prefix_len = {common}")
    if not exact_prefix:
        print(f"FIRST DIVERGENCE at index {common}:")
        b = base[common:common + 8]
        f = full[common:common + 8]
        print(f"  base[{common}:]= {b}  -> {[tok.decode([t]) for t in b]}")
        print(f"  full[{common}:]= {f}  -> {[tok.decode([t]) for t in f]}")
        print(f"  base tail around eos: "
              f"{base[-6:]} -> {[tok.decode([t]) for t in base[-6:]]}")
        ret = full[common:]
        print(f"  LCP-return first 12: {ret[:12]} -> {tok.decode(ret[:12])!r}")
    else:
        ret = full[len(base):]
        print(f"  clean-return first 12: {ret[:12]} -> "
              f"{tok.decode(ret[:12])!r}")


# Case A: assistant 后面接 user 消息 → 崩 (think 块)
run("A: new=[user], tools=tools",
    [{"role": "user", "content": "tool output: ok"}], tools)

# Case B: assistant 后面接 tool 消息 → 正常
run("B: new=[tool], tools=tools",
    [{"role": "tool", "content": "exit 0"}], tools)

# Case C: 不传 tools → 也崩 (同样是 think 块)
run("C: new=[user], tools=None",
    [{"role": "user", "content": "tool output: ok"}], None)

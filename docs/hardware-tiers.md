# Hardware tiers — which models run on which box

> The §5 spec defaults (DeepSeek R1 70B, Qwen 2.5 72B, Llama 3.3 70B) assume
> a workstation with ≥48GB VRAM. Most operators have less. This doc spells
> out the five supported tiers and which models map to each agent on each.
>
> The mapping lives in code at
> [`packages/agents/model_profiles.py`](../packages/agents/model_profiles.py)
> — this doc is the human-readable reflection.

## Pick your tier

Set `HARDWARE_PROFILE` in your `.env` (or Doppler) to one of:

| Profile        | Min VRAM | Min RAM | Typical hardware                                          |
|----------------|----------|---------|-----------------------------------------------------------|
| `cpu_only`     | 0 GB     | 16 GB   | NUC, laptop without dGPU, old Mac mini                    |
| `balanced`     | 8 GB     | 16 GB   | RTX 4060 Ti, 3060 12GB, 4070, M2/M3 base                  |
| `rx_7900_xt`   | 20 GB    | 32 GB   | RX 7900 XT/XTX, RTX 4080 16GB+, M3 Max 36GB               |
| `high_end`     | 24 GB    | 64 GB   | RTX 4090, RTX 6000 Ada, M3 Max 48GB+, M3 Ultra            |
| `workstation`  | 48 GB    | 128 GB  | Dual 4090, H100 80GB, A6000, M2/M3 Ultra 192GB            |

If `HARDWARE_PROFILE` is unset, the router checks `DETECTED_VRAM_GB` next,
then defaults to `balanced` (which works on essentially anything with 16GB
of system RAM — slow on pure CPU, but it runs).

## Model chains per tier

Each agent has a **chain** of three models: primary → backup → quantized
fallback. The router walks the chain on timeout / OOM / non-JSON output
(see §18 mitigations).

### `cpu_only` — everything 3-8B

| Agent     | Primary               | Backup                      | Quantized                |
|-----------|-----------------------|-----------------------------|--------------------------|
| Research  | qwen2.5:7b-q4_K_M     | llama3.1:8b-q4_K_M          | llama3.2:3b-q4_K_M       |
| Strategy  | qwen2.5:7b-q4_K_M     | llama3.1:8b-q4_K_M          | llama3.2:3b-q4_K_M       |
| Risk      | qwen2.5:7b-q4_K_M     | llama3.1:8b-q4_K_M          | llama3.2:3b-q4_K_M       |
| Execution | llama3.2:3b-q4_K_M    | qwen2.5:7b-q4_K_M           | llama3.2:3b-q4_K_M       |

Expect **5-15 tok/s** on a modern Ryzen/Intel CPU. Daily-briefing
generation takes ~30-60 seconds. Functional but slow.

### `balanced` — 7-8B with quick fallback

Same chain as `cpu_only` but inference runs on the GPU. **30-60 tok/s**.
Daily briefing ~5-10 seconds. This is the recommended starting point if
you're not sure.

### `rx_7900_xt` — 14B reasoning + 7B fast path

| Agent     | Primary                       | Backup                          | Quantized                |
|-----------|-------------------------------|---------------------------------|--------------------------|
| Research  | deepseek-r1:14b               | qwen2.5:14b-instruct-q4_K_M     | qwen2.5:7b-instruct-q4_K_M |
| Strategy  | qwen2.5:14b-instruct-q4_K_M   | llama3.1:8b-instruct-q4_K_M     | qwen2.5:7b-instruct-q4_K_M |
| Risk      | deepseek-r1:14b               | qwen2.5:14b-instruct-q4_K_M     | qwen2.5:7b-instruct-q4_K_M |
| Execution | llama3.1:8b-instruct-q4_K_M   | qwen2.5:7b-instruct-q4_K_M      | llama3.2:3b-instruct-q4_K_M |

VRAM math: each 14B at q4_K_M is ~11 GB. Your 20 GB card holds one hot +
swaps the second on demand. Two cold-starts per cycle, but acceptable.

**40-70 tok/s** on a 7900 XT once ROCm is enabled (WSL2 path). On Windows
native CPU-fallback path, expect **10-20 tok/s** — still fine for the
minute-scale trading loop.

### `high_end` — 32B reasoning

| Agent     | Primary                       | Backup                          | Quantized                |
|-----------|-------------------------------|---------------------------------|--------------------------|
| Research  | deepseek-r1:32b               | qwen2.5:32b-instruct-q4_K_M     | qwen2.5:14b-instruct-q4_K_M |
| Strategy  | qwen2.5:32b-instruct-q4_K_M   | llama3.3:70b-q4_K_M             | qwen2.5:14b-instruct-q4_K_M |
| Risk      | deepseek-r1:32b               | qwen2.5:32b-instruct-q4_K_M     | qwen2.5:14b-instruct-q4_K_M |
| Execution | llama3.3:70b-q4_K_M           | qwen2.5:14b-instruct-q4_K_M     | llama3.2:3b-instruct-q4_K_M |

Single 4090 or M3 Max 48GB territory. Quantized 70B sometimes runs (tight
fit, swap-heavy).

### `workstation` — spec defaults

Matches the §5 table verbatim: DeepSeek R1 70B, Qwen 2.5 72B, Llama 3.3
70B, Mistral Large. Needs 48GB+ VRAM and 128GB+ RAM. This is the highest
quality tier and where the published acceptance metrics were validated.

## Quick chooser

```bash
# Pull only the models your tier needs:
make pull-models                    # auto-detects HARDWARE_PROFILE
make pull-models PROFILE=rx_7900_xt # explicit
```

Disk required per tier: ~10 GB (cpu_only / balanced), ~30 GB (rx_7900_xt),
~70 GB (high_end), ~150 GB (workstation).

## Adding a new tier

1. Edit
   [`packages/agents/model_profiles.py`](../packages/agents/model_profiles.py)
   and add a `HardwareProfile` instance.
2. Append it to the `PROFILES` registry tuple.
3. Add the table row above and the chain table.
4. Add an assertion in
   `packages/agents/tests/test_model_profiles.py::test_known_profiles_are_registered`.
5. Open a PR. The chains are config, not code — reviewers should focus on
   the model sizes, not the syntax.

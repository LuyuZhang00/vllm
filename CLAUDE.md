# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Contribution Policy (Mandatory)

Before proposing a PR, check for duplicates:
```bash
gh issue view <issue_number> --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search "<issue_number> in:body"
gh pr list --repo vllm-project/vllm --state open --search "<short area keywords>"
```

- No low-value busywork PRs (single typo, isolated style change, etc.)
- Pure code-agent PRs are **not allowed** — a human must understand and defend every change
- PR descriptions for AI-assisted work must include: why not a duplicate, test commands run, and a statement that AI was used

## Development Commands

**Never use system `python3` or bare `pip`.** All Python commands go through `uv` and `.venv/bin/python`.

### Environment Setup
```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements/lint.txt
pre-commit install
```

### Install Dependencies
```bash
# Python-only changes (uses precompiled C/C++ extensions):
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# If also making C/C++ changes (builds from source):
uv pip install -e . --torch-backend=auto
```

### Run Tests
```bash
uv pip install -r requirements/test/cuda.in  # or requirements/test/cuda.txt on x86_64
.venv/bin/python -m pytest tests/path/to/test_file.py -v
```

### Linting
```bash
pre-commit run                    # staged files only
pre-commit run --all-files        # all files
pre-commit run ruff-check --all-files  # specific hook
pre-commit run mypy-3.10 --all-files --hook-stage manual  # mypy as in CI
```

Line length limit: 88 characters.

### Commit Messages
End with attribution trailers:
```
Co-authored-by: Claude
Signed-off-by: Your Name <your.email@example.com>
```

## Architecture Overview

vLLM is a high-throughput LLM inference and serving engine. It has a Python core with C++/CUDA kernels and a Rust frontend.

### Two Engine Generations

- **v1 engine** (`vllm/v1/`): The current default engine. Multi-process architecture with separate scheduler, model runner, and executor processes communicating via ZMQ IPC.
- **Legacy engine** (`vllm/engine/`): Older single-process engine, kept for backward compatibility.

### Key Components

**Entry Points** (`vllm/entrypoints/`):
- `openai/` — OpenAI-compatible API server (chat completions, completions, embeddings)
- `llm.py` — Offline synchronous LLM class for programmatic use
- `cli/` — Command-line interface (`vllm serve`, `vllm bench`)
- `anthropic/` — Anthropic Messages API compatibility

**Engine Core** (`vllm/v1/engine/`):
- `core.py` — EngineCore: the main engine loop that owns the scheduler and executor
- `async_llm.py` — AsyncLLM: async frontend that manages EngineCoreClient
- `llm_engine.py` — LLMEngine: legacy sync wrapper
- `core_client.py` — IPC client for communicating with EngineCore process
- `input_processor.py` — Tokenizes and preprocesses requests
- `output_processor.py` — Detokenizes and formats outputs

**Scheduler** (`vllm/v1/core/sched/`):
- `scheduler.py` — Main scheduler that decides which requests to run each iteration
- `interface.py` — SchedulerInterface and PauseState
- `output.py` — SchedulerOutput consumed by the model runner

**KV Cache Management** (`vllm/v1/core/`):
- `kv_cache_manager.py` — Manages KV cache block allocation/deallocation
- `block_pool.py` — Block pool with free/evict logic
- `kv_cache_utils.py` — Hashing, prefix caching, block utilities
- `kv_cache_coordinator.py` — Coordinates cache across attention backends

**Executor** (`vllm/v1/executor/`):
- `abstract.py` — Base Executor class
- `uniproc_executor.py` — Single-process executor
- `multiproc_executor.py` — Multi-process executor (one worker per GPU)
- `ray_executor.py` — Ray-based distributed executor

**Workers** (`vllm/v1/worker/`):
- `gpu_worker.py` — GPU worker process entry point
- `gpu_model_runner.py` — Runs the model forward pass, manages CUDA graphs, KV cache ops
- `worker_base.py` — Base worker interface

**Model Implementations** (`vllm/model_executor/models/`):
- One file per model architecture (e.g., `llama.py`, `qwen2.py`, `gpt2.py`)
- Registered via `_MODELS` dict mapping HF config types to implementation classes
- Use layers from `vllm/model_executor/layers/` (attention, linear, MoE, etc.)

**Attention** (`vllm/model_executor/layers/attention/` and `vllm/v1/attention/`):
- Backend-agnostic attention interface with pluggable backends (FlashAttention, FlashInfer, Triton, etc.)
- `v1/attention/backends/` contains v1-specific backend implementations

**Distributed** (`vllm/distributed/`):
- `parallel_state.py` — Manages tensor parallel, pipeline parallel, data parallel groups
- `kv_transfer/` — KV cache transfer between disaggregated prefill/decode nodes
- `ec_transfer/` — Expert cache transfer for MoE models

**Configuration** (`vllm/config/`):
- `model.py`, `cache.py`, `parallel.py`, `scheduler.py`, etc. — Typed config dataclasses
- All configs unified under `VllmConfig` in `vllm/config/vllm.py`

**Platform Abstraction** (`vllm/platforms/`):
- `interface.py` — Base platform interface
- `cuda.py`, `rocm.py`, `cpu.py`, `tpu.py`, `xpu.py` — Platform-specific implementations

**Rust Frontend** (`rust/`):
- `server/` — HTTP server (replaces Python server for production)
- `engine-core-client/` — Rust client for EngineCore IPC
- `tokenizer/` — Fast tokenization
- `tool-parser/` — Structured output / tool call parsing
- Built via `setuptools-rust` in `setup.py`

### Request Flow (v1)

1. API request arrives at `AsyncLLM` (or `LLMEngine`)
2. `InputProcessor` tokenizes and creates `EngineCoreRequest`
3. Request sent to `EngineCore` via ZMQ IPC
4. `Scheduler` adds request to waiting queue, schedules when resources available
5. `SchedulerOutput` sent to `Executor` → `Worker` → `GPUModelRunner`
6. `GPUModelRunner` runs model forward pass (with CUDA graphs if enabled)
7. Output tokens sent back through IPC chain to `OutputProcessor`
8. `OutputProcessor` detokenizes and yields to caller

### Build System

- Python packaging via `setuptools` with `setuptools-scm` for versioning
- C++/CUDA extensions built via CMake (`CMakeLists.txt`) and `torch.utils.cpp_extension`
- Rust extensions via `setuptools-rust`
- `VLLM_TARGET_DEVICE` env var controls target: `cuda`, `rocm`, `cpu`, `tpu`, `xpu`
- `VLLM_USE_PRECOMPILED=1` skips building C++/CUDA extensions (uses wheels)

### Test Structure

- `tests/` mirrors `vllm/` structure (e.g., `tests/v1/engine/`, `tests/models/`)
- `conftest.py` at root provides shared fixtures (model fixtures, cleanup, etc.)
- Markers: `@pytest.mark.slow_test`, `@pytest.mark.core_model`, `@pytest.mark.distributed`
- Model tests are in `tests/models/` organized by modality (language, vision, audio, etc.)

### Environment Variables

vLLM uses environment variables extensively (see `vllm/envs.py`). Key ones:
- `VLLM_USE_PRECOMPILED` — Skip C++ build
- `VLLM_TARGET_DEVICE` — Build target device
- `VLLM_WORKER_MULTIPROC_METHOD` — `fork` or `spawn` for multiprocessing
- `VLLM_LOGGING_LEVEL` — Log level
- `CUDA_VISIBLE_DEVICES` — GPU selection

## Domain-Specific Guides

Do not modify code in these areas without first reading the linked guide:
- **Editing these instructions**: `docs/contributing/editing-agent-instructions.md`

# Experimentations

Ad-hoc scripts for testing SimWorld benchmark hypotheses without changing the
library API.

## Humanoid Wrapper Comparison

Start the backend:

```bash
/home/ssingh/simworld_ue/Linux/SimWorld.sh /Game/Maps/demo_1.umap -nullrhi -nosplash -unattended -NoSound
```

Run stock vs non-blocking action wrappers:

```bash
.venv/bin/python experimentations/compare_humanoid_wrappers.py \
  --num-agents 1,2,4 \
  --steps 100 \
  --warmup-steps 10 \
  --state-mode individual \
  --output runs/profiles/wrapper_comparison.csv
```

Variants:

- `stock_blocking`: current `Communicator.humanoid_step_forward`, including
  Python `time.sleep(duration)`.
- `raw_sequential`: sends each `StepForward` command without sleeping, then
  ticks and observes.
- `raw_threaded`: attempts threaded Python dispatch, but still uses the shared
  UnrealCV socket lock, so it tests whether Python threading helps with the
  current transport.
- `raw_batch`: sends all `StepForward` commands through UnrealCV
  `request_batch` without sleeping, then ticks and observes.
- `state_tick_only`: no action; tick and observe structured state.

Use `--state-mode none` to isolate action/tick overhead, `individual` to avoid
the current batch-state latency artifact, or `batch` to reproduce batch state
behavior.

## Local LLM Structured Action Test

Start a local OpenAI-compatible server, for example Ollama:

```bash
ollama serve
```

Run constrained low-level action generation against one or more local models:

```bash
.venv/bin/python experimentations/test_local_llm_actions.py \
  --base-url http://127.0.0.1:11434/v1 \
  --models phi3:mini,llama3:8b \
  --output runs/profiles/local_llm_actions.json
```

The script uses `BaseLLM(provider="local")`, asks for JSON mode by default, and
validates responses with SimWorld's `LowLevelActionSpace` parser.

For vLLM, start an OpenAI-compatible server and point the same test at it:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --served-model-name qwen2.5-3b \
  --host 127.0.0.1 \
  --port 8000

.venv/bin/python experimentations/test_local_llm_actions.py \
  --base-url http://127.0.0.1:8000/v1 \
  --models qwen2.5-3b
```

The SimWorld planner can use the same endpoint with
`A2ALLM(model_name="qwen2.5-3b", url="http://127.0.0.1:8000/v1", provider="local")`.

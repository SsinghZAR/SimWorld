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
The default `--max-tokens` is intentionally 2048 because some Ollama Qwen
models emit reasoning before the final JSON answer.

For Ollama vision models, run the same preflight before connecting to Unreal:

```bash
ollama pull qwen3-vl:8b

.venv/bin/python experimentations/test_local_llm_actions.py \
  --base-url http://127.0.0.1:11434/v1 \
  --skip-text \
  --vision-models qwen3-vl:8b \
  --output runs/profiles/local_vlm_actions.json
```

Use `--vision-image path/to/frame.png` to test with a real SimWorld camera
frame. Keep `--vision-max-width` at 640 or lower on 16GB GPUs that also run the
UE5 server.

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

## Local Model UE Render Eval

Use `run_local_render_eval.py` after the local model preflight passes and the
external UE5 SimWorld server is running. On 16GB GPUs, start the Windows server
with lower-memory render settings:

```bash
SimWorld.exe /Game/Maps/demo_1.umap -windowed -ResX=640 -ResY=360 -NoSound -ExecCmds="r.Streaming.LimitPoolSizeToVRAM 1,r.Streaming.PoolSize 512,r.Streaming.MipBias 1,r.ScreenPercentage 50,r.Lumen.DiffuseIndirect.Allow 0,r.Lumen.Reflections.Allow 0"
```

Run a short text-state eval with a small non-thinking model first:

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M

.venv/bin/python experimentations/run_local_render_eval.py \
  --agent-mode text \
  --base-url http://127.0.0.1:11434/v1 \
  --model qwen3:4b-instruct-2507-q4_K_M \
  --steps 3 \
  --resolution 640x360 \
  --max-tokens 512 \
  --model-timeout 15 \
  --no-fallback-on-invalid
```

Then run a very short vision eval with a small non-thinking VLM. Use an
`*-instruct` Qwen3-VL tag; the plain `qwen3-vl` tags are thinking variants and
can spend the whole token budget without returning final action JSON.

```bash
ollama pull qwen3-vl:2b-instruct

.venv/bin/python experimentations/run_local_render_eval.py \
  --agent-mode vision \
  --ollama-native \
  --base-url http://127.0.0.1:11434/v1 \
  --model qwen3-vl:2b-instruct \
  --steps 1 \
  --resolution 640x360 \
  --vision-max-width 256 \
  --max-tokens 512 \
  --no-fallback-on-invalid
```

Each run writes `video.mp4`, `trajectory.json`, `metadata.json`, and
`model_responses.jsonl` under `runs/evals/<timestamp>/`. On 16GB GPUs, keep the
UE render resolution low and use `qwen3-vl:4b` or `qwen3-vl:2b` if UE and the
8B VLM compete for VRAM.

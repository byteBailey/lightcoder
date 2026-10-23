# LightCoder

A minimal, hackable **coding agent** and a **SWE-bench ablation / optimization testbed**.

LightCoder is a small AI coding agent (read / edit / run files in your terminal) built from
scratch. Every load-bearing design choice — parallel vs sequential tool execution, context
compression, sub-agent delegation, the system prompt — is a **first-class toggle**. That turns
the agent into an experiment harness: flip a toggle off to *ablate* a capability, add a new
module as a toggle to *optimize*, and measure the effect on [SWE-bench](https://www.swebench.com/)
resolve rate through the same pipeline.

## Results

Fixed SWE-bench Lite subsets, scored by the official harness, one run per configuration.
The write-ups in `docs/` give per-instance detail and caveats.

| Experiment | Model | Result | Notes |
|---|---|---|---|
| Baseline vs one-knob ablations, 20 instances | Qwen3.5-27B | Baseline 40% (8/20). Turning off compression or the sub-agent, or a minimal prompt, leaves it at 8/20; a ReAct-style prompt drops it to 6/20. | [M3](docs/M3_NOTES.md) |
| Same sweep on a shared endpoint | ~397B | Two operationally identical configs scored 5/20 and 10/20: endpoint nondeterminism, not the knobs. | [397B](docs/M_397b_NOTES.md) |
| Truncation fix, 20 instances | Qwen3.5-27B | Fixes the failure mode (no more cut-off empty patches) but 8 → 6 overall, within noise. | [M4](docs/M4_NOTES.md) |
| Execution-guided verification, 44 instances | DeepSeek-V4-Pro | 47.7% → **54.5%** (4 gained, 1 lost; not yet significant at n = 44). | [M5](docs/M5_NOTES.md) |
| Same | Qwen3.5-27B | 36.4% → 31.8% (2 gained, 4 lost; within noise): the weaker model gains nothing from the extra loop. | [M5](docs/M5_NOTES.md) |
| External reference: nanocoder on the same models and instances | 27B / DeepSeek | nanocoder 38.6% / 40.9% vs LightCoder 36.4% / 47.7% | [M5](docs/M5_NOTES.md) |

## Use it on your own projects (like a local CLI coding agent)

Install once as an isolated global tool (the Python equivalent of `npm install -g`):

```bash
uv tool install /path/to/LightCoder   # exposes `lightcoder` on PATH
```

Register your models once in `~/.lightcoder/models.json`, keyed by a name you choose. Any
OpenAI-compatible endpoint works; the file holds API keys, so keep it private (`chmod 600`):

```json
{
  "default": "deepseek-v4-pro",
  "models": {
    "deepseek-v4-pro": {
      "model": "deepseek-v4-pro",
      "base_url": "https://api.deepseek.com/v1",
      "api_key": "sk-...",
      "max_tokens": 16384
    },
    "local-qwen": {
      "model": "Qwen/Qwen3-32B",
      "base_url": "http://localhost:8000/v1",
      "api_key": "dummy"
    }
  }
}
```

`model` is the id sent to the API. `lightcoder --list-models` lists the names, and `/model`
switches between them in the REPL. Behaviour defaults such as `permission_mode` and
`max_rounds` go in `~/.lightcoder/settings.json`, and a project can override them in its own
`.lightcoder/settings.json`. For a one-off run, `LIGHTCODER_MODEL`, `LIGHTCODER_BASE_URL` and
`LIGHTCODER_API_KEY` in the shell take precedence over both files.

Then run it **inside any project**. LightCoder itself needs no environment, but the agent's
shell inherits yours, so activate the project's environment first if the agent should run its
code or tests with the project's interpreter and deps (exactly like Claude Code / nanocoder /
Codex CLI):

```bash
# Optional: lets the agent's shell commands use the project's Python and deps.
conda activate myproject        # or: source .venv/bin/activate
cd ~/myproject
lightcoder                      # inline REPL: streaming + markdown, output stays in your terminal
lightcoder -p "fix the failing import in main.py"   # one-shot
```

### The terminal UI

The default inline REPL keeps everything in your terminal's own scrollback:

- Each tool call is a full-width block: grey while it runs, green when it succeeds, red when it
  fails (a shell command that exits non-zero counts as a failure). Blocks show the tail of the
  output with a `/expand n` hint when lines are hidden; edits show a numbered diff.
- In `default` mode, edits and commands open a permission dialog showing the diff the edit
  would apply (↑↓ and Enter to choose, Esc to deny). **shift+tab** cycles `default → auto → plan`.
- The footer shows the mode, model, git branch, session tokens (↑ sent, ↓ generated), the
  prompt-cache hit rate, and context use, which turns yellow and then red as it approaches the
  compression thresholds.
- Slash commands: `/model`, `/mode`, `/resume`, `/expand`, `/clear`, `/help` (type `/` to list them).

LightCoder's *own* runtime (openai, rich) stays isolated in the tool env; only the agent's
`run_bash`/file tools operate in your activated project environment. `LIGHTCODER_*` vars are
preferred over `OPENAI_*` so they never collide with a project's own OpenAI config.

## Develop / run the SWE-bench eval

```bash
uv venv && uv pip install -e ".[dev,eval]"
uv run pytest tests/                       # unit tests
uv run python -m eval.run_single --instance pallets__flask-4045   # one instance
```

## Layout

| Path | What |
|---|---|
| `lightcoder/` | the agent (loop, tools, context, prompt) — toggles built in |
| `eval/` | SWE-bench harness + ablation/optimization runner |
| `docs/` | design notes + experiment writeups (index below) |
| `results/` | run outputs (predictions, trajectories); not committed, summaries go in `docs/` |

### `docs/`

| File | What |
|---|---|
| [`DESIGN.md`](docs/DESIGN.md) | architecture, the config knobs, and how each ablation is measured |
| [`M1_NOTES.md`](docs/M1_NOTES.md) | single-instance pipeline end to end: clone, agent, diff, official harness |
| [`M2_NOTES.md`](docs/M2_NOTES.md) | batch runner and the 20-instance baseline; infrastructure bugs found on the way |
| [`M3_NOTES.md`](docs/M3_NOTES.md) | one-factor-at-a-time ablation on 27B |
| [`M4_NOTES.md`](docs/M4_NOTES.md) | max_tokens truncation causing empty patches, the fix, and its honest measurement |
| [`M_397b_NOTES.md`](docs/M_397b_NOTES.md) | the 397B sweep and the endpoint-nondeterminism confound it exposed |
| [`M5_NOTES.md`](docs/M5_NOTES.md) | 44-instance subset, execution-guided verification, nanocoder comparison |
| `ablation_table*.md`, `*_summary*.json`, `ablation_summaries*/` | raw numbers behind the M2–M4 and 397B tables |

# ChemML Purple Agent

A ML engineering agent for [AgentBeats](https://agentbeats.dev) — Sprint 2, Research Agent track (MLE-Bench).

## What This Agent Does

This Purple Agent solves **Kaggle-style ML competitions** end-to-end when evaluated by the [MLE-Bench Green Agent](https://github.com/RDI-Foundation/mle-bench-green):

1. **Receives** a competition dataset (tar.gz) + instructions from the Green Agent
2. **Analyzes** the dataset structure (columns, types, shapes)
3. **Detects** chemistry data (SMILES, InChI, fingerprints) → activates specialized strategies
4. **Generates** a complete Python ML pipeline via LLM
5. **Executes** the code in a sandboxed subprocess
6. **Returns** `submission.csv` for automated grading

## Key Design

- **General-purpose ML** — handles tabular, image, text, time-series, and signal data
- **Chemistry-aware** — when molecular data is detected, uses RDKit for fingerprints and property prediction
- **Robust fallback** — always produces a valid submission, even if model training fails
- **Multi-provider LLM** — uses [LiteLLM](https://docs.litellm.ai/) (works with OpenAI, Anthropic, Google, etc.)

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Set your API key
cp .env.example .env
# Edit .env with your API key

# 3. Run the server
uv run src/server.py

# 4. Test it
uv run pytest --agent-url http://localhost:9009
```

## Docker

```bash
docker build -t chemml-purple-agent .
docker run -p 9009:9009 --env-file .env chemml-purple-agent
```

## Configuration

Set these in `.env`:

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `openai/gpt-4o` | LLM for code generation |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `LLM_TEMPERATURE` | `0.2` | Lower = more deterministic code |
| `LLM_MAX_TOKENS` | `8192` | Max response tokens |
| `CODE_TIMEOUT` | `900` | Max seconds for ML training |

## Architecture

```
src/
├── server.py      # A2A server + agent card
├── executor.py    # Request handling + task lifecycle
├── agent.py       # Core ML engine (data analysis → code gen → execution → submission)
└── messenger.py   # A2A messaging utilities
```

## Competition

- **Track**: Sprint 2 — Research Agent (MLE-Bench)
- **Protocol**: [A2A (Agent-to-Agent)](https://a2a-protocol.org/latest/)
- **Platform**: [AgentBeats](https://agentbeats.dev)
- **Leaderboard**: [MLE-Bench](https://agentbeats.dev/agentbeater/mle-bench)

# Memory Systems Lab Implementation

This `src/` folder contains the offline memory agents, benchmark and tests.

- It keeps the same high-level structure
- Offline mode is deterministic and needs no API key. Provider model construction is available in `model_provider.py`; live agent execution is an optional extension.
- The benchmark structure should include: standard benchmark + long-context stress benchmark
- The runtime should support these providers: `openai`, `custom`, `gemini`, `anthropic`, `ollama`, `openrouter`

Suggested flow:

1. Start with `config.py`
2. Implement `memory_store.py`
3. Finish `agent_baseline.py`
4. Finish `agent_advanced.py`
5. Implement `benchmark.py`
6. Make `test_agents.py` pass

Datasets are available at the repo root in `data/`.

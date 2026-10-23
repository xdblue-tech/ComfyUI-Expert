# distribute_agent (moved)

The agent package now lives at `src/distribute_agent/` and installs from the
repository root `pyproject.toml`; its documentation moved with it to
`src/distribute_agent/README.md`.

Nothing imports this directory. It stays only so links to the old path resolve.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
.venv/bin/python -m unittest discover -s tests/agent
```

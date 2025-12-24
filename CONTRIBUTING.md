# Contributing

Thanks for considering a contribution!

## Development setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Code style
- `ruff` for lint + formatting
- `mypy` for type checking
- `pytest` for tests

Run locally:
```bash
ruff check .
mypy src/tribrain
pytest
```

## Pull requests
Please include:
- a clear description of the change and why it matters
- tests (or a rationale why tests are not applicable)
- example commands in the PR description when behavior changes

## Security
See `SECURITY.md`.

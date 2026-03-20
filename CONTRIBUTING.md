# Contributing

Thanks for taking the time to contribute.

## Development setup

- Create a virtualenv and install dev deps:
  - `pip install -e ".[dev]"`
- Run quality checks:
  - `ruff check .`
  - `ruff format .`
  - `pytest`

## Pull requests

- Keep PRs focused and small.
- Add/adjust tests when changing behavior.
- Ensure CI is green.

## Code style

- Formatting/linting is enforced via Ruff.
- Prefer small, readable functions and explicit error messages.

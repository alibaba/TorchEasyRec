# TorchEasyRec Agent Guide

Canonical instructions for AI coding agents in this repository. `CLAUDE.md` imports this file; keep shared agent guidance here.

## Protobuf

- After editing any `.proto` under `tzrec/protos/`, regenerate bindings with `bash scripts/gen_proto.sh`. `_pb2.py` / `_pb2.pyi` files are generated (gitignored); never edit them.

## Environment

- If a tool you need (python, pip, pre-commit, protoc) is missing, look for a project conda env or venv and activate it; otherwise stop and ask. Do NOT install substitutes or new toolchains on your own.
- There is no editable install: run from the repo root with `PYTHONPATH=.`.
- One-time dev setup: `pip install -r requirements.txt`, then `pre-commit install`.

## Scratch Space

Use the gitignored `experiments/` directory at the repo root for scratch scripts, throwaway configs, and experiment outputs; tests write temp data under `tmp/`. Never commit files from either.

## Testing

- Framework: `unittest`; tests live next to the code as `<module>_test.py`.
- Full suite: `python tzrec/tests/run.py` (`--list_tests`, `--scope gpu|h20`). Single test: `python -m tzrec.modules.fm_test FactorizationMachineTest.test_fm_0`.
- Parameterize case tables with `@parameterized.expand(..., name_func=parameterized_name_func)` from `tzrec/utils/test_util.py`; create scratch dirs with its `make_test_dir()`.
- Declare CI lanes with `@mark_ci_scope("gpu")` / `@mark_ci_scope("h20", "gpu")` at class or method level only; module-level tagging is unsupported.

## Linting and Type Checking

- Before committing: run `pre-commit run -a` (ruff lint+format at line 88, license header insertion, codespell, mdformat, LF endings) and fix everything it reports; type-check with `python scripts/pyre_check.py` (Pyre strict; proto-generated files and tests are excluded).
- Never `# noqa` a line-length violation; let ruff-format decide layout and do not fight it.

## Commits and Pull Requests

- Commit subject: `[tag] short imperative subject`, tag one of `feat`, `bugfix`, `refactor`, `ci`, `doc`, `chore`, `perf`. The `(#123)` suffix is added by squash merge; do not add it yourself.
- Keep commit bodies minimal: no bullet lists of changes. A `[bugfix]` commit message must explain the root cause of the bug and how the fix works.
- The PR body should be clear and informative, with a Test Plan section that describes how you tested the change. If there were multiple potential paths you could have taken, call them out succinctly and justify the one you took.
- Never include customer names or internal resource IDs in branch names, commit messages, or PR bodies.
- Unless specified otherwise, push the branch to a fork and open the PR against the main repository.

## Coding Style

Docstrings and inline comments follow different rules; do not confuse them:

- **Docstrings are mandatory** (ruff-enforced): Google style on public and private classes, functions, and methods; `*_test.py` and `tzrec/ops/` are exempt, as are modules, packages, `__init__`, and dunders. Never strip or skimp docstrings to "minimize comments".
- **Inline `#` comments are minimized**: code should be self-explanatory; comment only non-obvious "why" context, one short line. No commented-out code, no change-history or tombstone comments. Use `# TODO(username): ...` and `# NOTE: ...`; do not introduce FIXME/XXX/HACK. Denser math commentary is fine in `tzrec/ops/` kernels.
- Code, comments, and docstrings are English-only (Chinese belongs in `docs/`); prefer plain ASCII in new comments outside `tzrec/ops/`.
- Naming: `fg` means feature generator (`fg_mode`, `fg_handler`, `pyfg`), never feature group.
- Don't create trivial 1-2 line single-use helpers unless they significantly improve readability.
- Use the repo logger (`from tzrec.utils.logging_util import logger`), not `print()`, in library code; `print()` is acceptable only in CLI tools and benchmarks.
- Assume the reader knows PyTorch and TorchRec. Match existing patterns; if uncertain, choose the simpler, more concise implementation.

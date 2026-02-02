# Repository Guidelines

## Project Structure & Module Organization

- `paper_novelty_pipeline/`: main Python package (pipeline logic).
  - `phases/`: phase implementations (Phase 1–4).
  - `services/`: external integrations (LLM/OpenReview/PDF/Wispaper).
  - `utils/`: shared helpers (paths, IO, logging, text cleaning).
  - `entrypoints.py`: preferred in-process API used by CLI wrappers.
- `scripts/`: runnable entrypoints and orchestration helpers (e.g. `run_phase1_batch.py`, `run_phase3_all.sh`).
- `docs/`: documentation assets (e.g. `docs/images/`).
- `requirements.txt`, `.env_example`, `README.md`: dependency and usage docs.

Generated artifacts typically live under `output/<run>/<paper_id>/phase*/` (see `README.md` for the full layout).

## Build, Test, and Development Commands

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/run_phase1_batch.py --help
```

Common local runs:

```bash
python scripts/run_phase1_batch.py --papers "<pdf_url>" --out-root output/demo --force-year 2026
bash scripts/run_phase3_all.sh output/demo/<paper_id_dir>
bash scripts/run_phase4.sh output/demo/<paper_id_dir>
```

## Coding Style & Naming Conventions

- Python: 4-space indentation; keep formatting Black-compatible (Black/flake8 are optional and listed in `requirements.txt`).
- Naming: `snake_case` for files/functions, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Keep `scripts/` thin: put reusable logic in `paper_novelty_pipeline/` (ideally via `paper_novelty_pipeline/entrypoints.py`), and reserve scripts for argument parsing and orchestration.

## Testing Guidelines

- This repo currently does not ship an automated test suite.
- Before opening a PR, run a smoke check on a single paper and verify key outputs exist (e.g. `phase1/phase1_extracted.json`, `phase3/phase3_complete_report.json`, `phase4/novelty_report.md`).
- If you add tests, use `pytest` (commented in `requirements.txt`), place files under `tests/` as `test_*.py`, and run `pytest -q`.

## Commit & Pull Request Guidelines

- Follow Conventional Commits, consistent with existing history (e.g. `chore: ...`, plus `feat:`, `fix:`, `docs:` as applicable).
- PRs should include: a short description, repro/run commands, and (when relevant) a small excerpt of output paths/log lines. Avoid committing large `output/` artifacts.

## Security & Configuration Tips

- Copy `.env_example` to `.env` for local use; never commit API keys or tokens.
- Consider adding `.env`, `output/`, and `logs/` to `.gitignore` in PRs that introduce those workflows.


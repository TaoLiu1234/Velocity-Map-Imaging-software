# Contributing to VMI_workflow

Thank you for considering a contribution. This document covers the minimum
you need to develop and test the project locally.

## Development setup

Requires Python 3.11+.

```bash
git clone <your-fork-url>
cd vmi-workflow
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install pyarrow pytest ruff                      # optional fast CSV loader + dev tools
python VMI_workflow.py                               # run the app
```

Linux only: Qt also needs a few system libraries even for offscreen runs
(`libgl1 libegl1 libxkbcommon0 libglib2.0-0 libdbus-1-3 libfontconfig1` —
see `.github/workflows/ci.yml` for the exact set).

## Running the tests

```bash
python tests/test_core.py                              # numerics goldens (~15 s)
QT_QPA_PLATFORM=offscreen python tests/test_smoke.py   # end-to-end UI workflow (~60 s)
```

Both scripts are plain-runnable and pytest-collectable. The smoke test
forces the offscreen Qt platform itself and never writes into the
repository (it `chdir`s to a temp workspace). See `tests/README.md` for
details, benchmark scripts, and the full test layout.

## Goldens discipline (important)

The scientific numbers are pinned by `tests/golden_core.json` and
`tests/golden_smoke.json` on a deterministic synthetic triplet. Do **not**
regenerate them as a side effect of a refactor. Only after an intentional,
reviewed behaviour change, regenerate in this order:

```bash
python tests/make_sample_data.py                 # only if the generator changed
python tests/test_core.py --update-golden
QT_QPA_PLATFORM=offscreen python tests/test_smoke.py --update-golden
```

and say explicitly in your PR what shifted and why (see `tests/README.md`
for the current baseline and past re-baselines).

## Style guidance

The codebase predates this repository's public release; consistency with
the existing style matters more than personal preference:

- Match the existing code style of the file you are editing. Do **not**
  mass-reformat, re-quote, or re-indent untouched code — it destroys diff
  and blame readability.
- Docstrings follow the plain-English convention already used in
  `VMI_workflow_core.py` and the newer `MainWindow` methods: a one-line
  summary, then `Physics/Method:`, `Assumptions:`, `Limitations:` /
  `Notes:` blocks where they help. Keep them factual and numerical.
- Keep the pure numerics in `VMI_workflow_core.py` /
  `VMI_workflow_reconstruction.py` free of any Qt imports; the GUI layer is
  `VMI_workflow.py`.
- Any change to drag/overlay interactions must preserve the interaction
  contract documented in `ARCHITECTURE.md` (§12) and must not change
  scientific output for the same data + parameters.
- Before deleting any "dead" code, grep for callers first (several
  seemingly-orphaned helpers are live; the history is in
  `ARCHITECTURE.md` §9/§16).
- Lint gate: `ruff check --select F,E9 .` must stay clean of unused
  imports, undefined names and syntax errors (pyflakes-level only; no
  style-class enforcement).

## Pull requests

- Keep PRs focused; one behaviour change per PR.
- All tests must pass in CI (ubuntu full suite + windows core suite).
- Update `README.md` / `docs/` / `ARCHITECTURE.md` when behaviour,
  workflows, or documented numbers change.
- Repo content is English-only; please keep all new files and comments in
  English.

## Reporting issues

Include: OS + Python version, the `.dat` triplet layout (not the data
itself), the exact tab/step, and the console output. For visual issues,
attach a screenshot.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE).

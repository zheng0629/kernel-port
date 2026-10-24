# Contributing to kernel-port

Thanks for considering a contribution. kernel-port is an agent-callable
toolchain: it handles every deterministic step around cross-language
porting (mine → extract → glue → verify → bench). It never calls an LLM
itself — the calling agent does the translation.

## Setup

- Python 3.10+
- A JDK (17+) for the Java-backend tests
- pip install -e ".[dev]"
- Run the suite: python -m pytest -q

## Workflow

1. Fork the repo and create a branch.
2. Make your change. New behavior needs a test.
3. If you touch the CLI, update both READMEs (English and Chinese) —
   they are verified against --help.
4. Open a pull request against master and describe what changed and
   how you tested it.

## House rules

- Keep the pipeline deterministic: no network calls, no LLM calls
  inside the tool.
- Benchmark claims must survive the ±20% noise band: inside the band,
  the verdict is "inconclusive", not "faster".
- Keep the honest-reporting style: a negative result (e.g. "not worth
  porting") is a valid result, not a failure.

## Good first issues

Issues labeled `good first issue` are scoped for newcomers — each states
its difficulty and acceptance criteria. Ask in the issue if anything is
unclear.

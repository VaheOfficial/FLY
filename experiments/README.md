# experiments

Throwaway and exploratory code. Each experiment is its own directory with its
own `Cargo.toml` (this folder is excluded from the root workspace on purpose).

Rules:
- An experiment answers one question. Name the directory after the question.
- Nothing in here is depended on by `crates/` or `tools/`.
- When an experiment produces something worth keeping, it is rewritten into the
  real crate with tests, and the experiment is deleted.

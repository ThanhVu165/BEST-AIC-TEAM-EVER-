# Team Workflow

## Roles

### Person 1 — Video Processing / Data Layer

Branch: `feature/video-pipeline`

Owns:

- `video_pipeline/`
- data ingestion scripts
- database generation
- vector index generation
- data validation

### Person 2 — Query Engine

Branch: `feature/query-engine`

Owns:

- `query_engine/`
- retrieval experiments
- temporal localization
- reranking
- task solvers
- evaluation

### Person 3 — UI

Branch: `feature/ui`

Owns:

- `ui/`
- Streamlit components
- API client integration

## Shared areas

Changes to the following need team agreement:

- `schemas/`
- `api/`
- `configs/` when changing shared semantics
- `docs/`
- root dependency files

## Branch rules

- Never develop directly on `main` for feature work.
- Keep commits focused.
- Open a PR into `develop` for integration.
- `main` is stable/releasable.
- Avoid unrelated formatting changes in feature PRs.

## AI-assisted development rules

Every contributor may use an AI coding assistant, but the assistant must receive the repository context from `docs/` before changing code.

At the start of an AI-assisted task, provide or point the assistant to:

1. `docs/PROJECT_CONTEXT.md`
2. `docs/ARCHITECTURE.md`
3. `docs/DATA_CONTRACT.md`
4. `docs/API_CONTRACT.md`
5. `docs/AI_CONTEXT.md`

The task prompt should also state:

- current branch
- module being changed
- issue/task number
- expected contract impact
- files that may be changed
- files that must not be changed

The assistant must not silently modify another person's module to make a feature work. If an interface needs to change, it should stop and document the contract change.

## Definition of Done

A task is complete when:

- code follows the current contracts
- tests or a reproducible validation command exist where appropriate
- no dataset/model artifacts are committed
- documentation is updated if behavior or contract changed
- integration impact is stated in the PR description

## Integration order

Preferred order:

```text
schemas / contracts
      ↓
video pipeline data package
      ↓
query-engine baseline
      ↓
API integration
      ↓
UI integration
      ↓
benchmark / optimization
```

UI can use mock API responses before the real Query Engine is available.

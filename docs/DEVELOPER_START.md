# Developer Start Guide

## 1. Clone and create the environment

Windows PowerShell:

```powershell
git clone https://github.com/ThanhVu165/BEST-AIC-TEAM-EVER-.git
cd BEST-AIC-TEAM-EVER-
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[api,ui,dev]"
```

Install ML dependencies only when needed by the Video/Query implementation:

```powershell
pip install -e ".[ml]"
```

## 2. Initialize the local database

```powershell
python scripts/init_db.py
```

This creates `database/aic2026.sqlite`. The database is local generated state and is intentionally ignored by Git.

## 3. Run tests

```powershell
pytest
```

The tests currently validate the contracts and a mock end-to-end API path. They do not measure retrieval quality.

## 4. Run the mock API

Terminal 1:

```powershell
.\scripts\run_api.ps1
```

Health check:

```text
http://127.0.0.1:8000/health
```

Interactive API docs:

```text
http://127.0.0.1:8000/docs
```

## 5. Run the Streamlit UI

Terminal 2:

```powershell
.\scripts\run_ui.ps1
```

The UI sends requests only to FastAPI. The current response is produced by `MockQueryEngine`.

## 6. Team development

Create a feature branch from `develop`:

```powershell
git checkout develop
git pull
git checkout -b feature/<your-task>
```

Ownership:

- Person 1: `video_pipeline/`, data/index implementation.
- Person 2: `query_engine/`.
- Person 3: `ui/`.
- Shared contract changes: `schemas/`, `data_layer/`, `api/`, `docs/` require coordination.

Do not commit generated data or model artifacts.

## 7. Required reading for AI coding assistants

At the start of every AI coding session, provide or point the assistant to:

1. `docs/AI_CONTEXT.md`
2. `docs/PROJECT_CONTEXT.md`
3. `docs/ARCHITECTURE.md`
4. `docs/DATA_CONTRACT.md`
5. `docs/API_CONTRACT.md`
6. `docs/TEAM_WORKFLOW.md`

Then state the exact task and owned module.

## 8. Integration rule

A module is considered integration-ready when:

- it respects the shared Pydantic schemas;
- its public interface is covered by at least one test;
- it does not require another person's private implementation;
- it can run against the mock implementation where appropriate;
- it documents any new assumptions.

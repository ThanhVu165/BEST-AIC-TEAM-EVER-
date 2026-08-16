$ErrorActionPreference = "Stop"

Write-Host "Starting AIC 2026 FastAPI on http://127.0.0.1:8000"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

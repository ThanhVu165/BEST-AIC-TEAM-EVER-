$ErrorActionPreference = "Stop"

$env:AIC_API_BASE_URL = if ($env:AIC_API_BASE_URL) { $env:AIC_API_BASE_URL } else { "http://127.0.0.1:8000" }
Write-Host "Starting Streamlit UI against $env:AIC_API_BASE_URL"
python -m streamlit run ui/app.py

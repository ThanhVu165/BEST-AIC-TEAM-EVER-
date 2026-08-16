param(
    [string]$DataRoot = ".\data\raw"
)

$ErrorActionPreference = "Stop"
$DataRoot = (Resolve-Path $DataRoot).Path

Write-Host "AIC data root: $DataRoot"
Write-Host "[1/3] Auditing official project data..."
python tools/dataset_audit.py $DataRoot
if ($LASTEXITCODE -ne 0) { throw "Dataset audit failed." }

Write-Host "[2/3] Building SQLite + FAISS package..."
python tools/build_data_package.py --data-root $DataRoot
if ($LASTEXITCODE -ne 0) { throw "Data package build failed." }

Write-Host "[3/3] Validating generated package..."
python scripts/validate_batch1.py `
  --db database/aic2026.sqlite `
  --index indexes/clip_vit_b32.faiss `
  --mapping indexes/clip_vit_b32.mapping.json
if ($LASTEXITCODE -ne 0) { throw "Generated package validation failed." }

Write-Host "Local AIC2026 data package is ready."

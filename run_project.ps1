$ErrorActionPreference = "Stop"

Write-Host "[1/4] Validating raw Olist tables..." -ForegroundColor Cyan
python -m src.data.validate_data

Write-Host "[2/4] Building order-level modeling dataset..." -ForegroundColor Cyan
python -m src.features.build_dataset

Write-Host "[3/4] Training and evaluating models..." -ForegroundColor Cyan
python -m src.models.train

Write-Host "[4/4] Running tests..." -ForegroundColor Cyan
pytest -q

Write-Host "SmartShip pipeline completed. Start the app with: streamlit run app/main.py" -ForegroundColor Green

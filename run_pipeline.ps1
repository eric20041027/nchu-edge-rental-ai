# NCHU AI Rental Pipeline Automation Script (Windows PowerShell)
$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "開始執行 興大 AI 租屋推薦系統流水線..." -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

if (Test-Path ".\venv\Scripts\Activate.ps1") {
    Write-Host "[環境] 啟動虛擬環境 venv..." -ForegroundColor Yellow
    . .\venv\Scripts\Activate.ps1
}

Write-Host "Step 1: Crawling..." -ForegroundColor Green
python pipeline/crawlers/rent_info_catcher.py

Write-Host "Step 2: Merging..." -ForegroundColor Green
python pipeline/data_prep/merge_sources.py

Write-Host "Step 3: Commute Data..." -ForegroundColor Green
python pipeline/data_prep/update_commute_data.py

Write-Host "Step 4: Dataset..." -ForegroundColor Green
python pipeline/data_prep/generate_dataset.py
python pipeline/data_prep/precompute_embeddings.py

Write-Host "Step 5: Training..." -ForegroundColor Green
python pipeline/model_training/train_and_export_onnx.py

Write-Host "Step 6: Quantizing & Evaluating..." -ForegroundColor Green
python pipeline/model_training/quantize_model.py
python pipeline/model_training/evaluate_model.py

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "流水線執行完畢！" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

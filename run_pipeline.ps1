# NCHU AI Rental Pipeline Automation Script (Windows PowerShell)
$ErrorActionPreference = "Continue"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "開始執行 興大 AI 租屋推薦系統流水線 (最強升級版)..." -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

if (Test-Path ".\venv\Scripts\Activate.ps1") {
    Write-Host "[環境] 啟動虛擬環境 venv..." -ForegroundColor Yellow
    . .\venv\Scripts\Activate.ps1
}

# Step 1: Data Ingestion
Write-Host "Step 1: Crawling (DD-Room & NCHU)..." -ForegroundColor Green
python pipeline/crawlers/crawler_ddroom.py
python pipeline/crawlers/crawler_nchu.py

# Step 2: Semantic Augmentation
Write-Host "Step 2: Semantic Augmentation & Real-world Commute..." -ForegroundColor Green
python pipeline/data_prep/augment_with_llm.py
python pipeline/data_prep/update_commute_data.py

# Step 3: Massive Silver Labeling
Write-Host "Step 3: Generating 5000+ Silver Labels using Gemini..." -ForegroundColor Green
python pipeline/data_prep/silver_labeling.py
python pipeline/data_prep/generate_dataset.py

# Step 4: Active Hard Negative Mining
Write-Host "Step 4: Active Hard Negative Mining (Detecting Model Blind Spots)..." -ForegroundColor Green
python pipeline/data_prep/mine_hard_negatives.py
python pipeline/data_prep/generate_dataset.py

# Step 5: Training with LTR
Write-Host "Step 5: Fine-tuning with ListNet/RankNet Loss & Exporting ONNX..." -ForegroundColor Green
python pipeline/model_training/train_and_export_onnx.py

# Step 6: Post-Process & Validation
Write-Host "Step 6: Quantizing, Precomputing & Evaluating..." -ForegroundColor Green
python pipeline/model_training/quantize_model.py
python pipeline/data_prep/precompute_embeddings.py
python pipeline/model_training/evaluate_model.py

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "流水線執行完畢！NDCG 最強推薦版本已部署。" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

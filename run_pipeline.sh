#!/bin/bash
# NCHU AI Rental Pipeline Automation Script (Unix/Linux/macOS)
set -e

echo -e "\033[0;36m==========================================\033[0m"
echo -e "\033[0;36m開始執行 興大 AI 租屋推薦系統流水線 (V2)...\033[0m"
echo -e "\033[0;36m==========================================\033[0m"

# Activation logic for venv
if [ -d "./venv" ]; then
    echo -e "\033[0;33m[環境] 啟動虛擬環境 venv...\033[0m"
    source ./venv/bin/activate
fi

# Step 1: Data Ingestion (Crawling)
echo -e "\033[0;32mStep 1: Crawling (DD-Room & NCHU)...\033[0m"
python3 pipeline/crawlers/crawler_ddroom.py
python3 pipeline/crawlers/crawler_nchu.py

# Step 2: Semantic Augmentation & Commute
echo -e "\033[0;32mStep 2: Semantic Augmentation & Real-world Commute...\033[0m"
python3 pipeline/data_prep/augment_with_llm.py
python3 pipeline/data_prep/update_commute_data.py

# Step 3: Dataset Synthesis
echo -e "\033[0;32mStep 3: Merging & Generating Training Dataset...\033[0m"
python3 pipeline/data_prep/merge_sources.py
python3 pipeline/data_prep/generate_dataset.py

# Step 4: Training & Optimization
echo -e "\033[0;32mStep 4: Training RoBERTa & Exporting ONNX...\033[0m"
python3 pipeline/model_training/train_and_export_onnx.py

# Step 5: Post-Process & Validation
echo -e "\033[0;32mStep 5: Quantizing, Precomputing & Evaluating...\033[0m"
python3 pipeline/model_training/quantize_model.py
python3 pipeline/data_prep/precompute_embeddings.py
python3 pipeline/model_training/evaluate_model.py

echo -e "\033[0;36m==========================================\033[0m"
echo -e "\033[0;36m流水線執行完畢！最強推薦版本已部署。\033[0m"
echo -e "\033[0;36m==========================================\033[0m"

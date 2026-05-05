#!/bin/bash
# NCHU AI Rental Pipeline Automation Script
# This script runs the end-to-end workflow from crawling to evaluation.

set -e # Exit on error

echo "============================================================"
echo "Starting NCHU AI Rental Pipeline..."
echo "============================================================"

# 1. Data Collection
echo "[Step 1/6] Running crawlers..."
python pipeline/crawlers/rent_info_catcher.py # NCHU Official
# Note: crawler_ddroom.py is long-running, so we use existing data if not specified
# python pipeline/crawlers/crawler_ddroom.py 

# 2. Data Merging & De-duplication
echo "[Step 2/6] Merging sources with advanced de-duplication..."
python pipeline/data_prep/merge_sources.py

# 3. Commute Data Update
echo "[Step 3/6] Updating real-world commute data (ArcGIS + OSRM)..."
python pipeline/data_prep/update_commute_data.py

# 4. Dataset Generation & Pre-processing
echo "[Step 4/6] Generating training dataset & precomputing embeddings..."
python pipeline/data_prep/generate_dataset.py
python pipeline/data_prep/precompute_embeddings.py

# 4.5 Hard Example Mining (Active Learning)
echo "[Step 4.5/6] Mining hard examples from previous model version..."
python pipeline/model_training/mine_hard_examples.py

# 5. Model Training & Export
echo "[Step 5/6] Training RoBERTa model & exporting to ONNX..."
python pipeline/model_training/train_and_export_onnx.py

# 6. Quantization & Evaluation
echo "[Step 6/6] Quantizing model & running final evaluation..."
python pipeline/model_training/quantize_model.py
python pipeline/model_training/evaluate_model.py

echo "============================================================"
echo "Pipeline completed successfully!"
echo "Check README.md for deployment instructions."
echo "============================================================"

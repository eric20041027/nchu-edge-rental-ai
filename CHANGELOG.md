# Changelog - NCHU AI Rental Recommendation System

All notable changes to this project will be documented in this file.

## [2026-05-08] - Model Upgrading & Automation
### Added
- **Model Architecture Upgrade**: Migrated from `hfl/rbt3` (3-layer) to **`hfl/rbt6` (6-layer RoBERTa)**. This change doubles the transformer depth, significantly enhancing high-level semantic feature extraction for complex natural language queries.
- **Workflow Automation**: Implemented `run_pipeline.ps1` for Windows environments, encapsulating the end-to-end lifecycle from raw data ingestion to INT8 quantization.
### Technical Fixes
- Resolved Git upstream conflicts in `train_and_export_onnx.py` while preserving custom hyperparameter tuning (Batch Size: 64, LR: 2e-5).
- Cleaned up redundant ONNX artifacts (`sym_shape_infer_temp.onnx`) post-shape inference.

## [2026-05-05] - Ranking Optimization & Data Refinement
### Added
- **Ranking Engine Optimization**: Implemented **Temperature Calibration (Option C)** within the `WeightedTrainer` class. Scaling logits by $T=2.0$ effectively flattens the probability distribution, preventing sigmoid saturation and improving the resolution of high-relevance samples.
- **Advanced Relevance Weighting**: Introduced aggressive relevance weights (Perfect=15.0, None=6.0) to penalize hard negatives and boost NDCG@5 performance.
- **Data Engineering**: Implemented Taichung-specific geolocation filtering and address normalization logic to ensure data purity.

## [2026-04-02] - Operational Robustness
### Added
- **Feature Matching Layer**: Integrated specific amenities (furniture, appliances) into the Rule-based Matching System (RMS) to ensure 100% precision for explicit requirement filtering.
- **Refactoring**: Standardized Python type hinting across `pipeline/` for better maintainability.
- **Documentation**: Enhanced README with Mermaid diagrams illustrating the Data Pipeline and Inference Flow.

## [2026-03-12 ~ 2026-03-19] - Edge AI Deployment & UI Overhaul
### Added
- **WASM Optimization**: Configured `onnxruntime-web` with specific WASM paths and CORS isolation headers to enable multi-threaded inference on client-side browsers.
- **UI Design**: Implemented a **Glassmorphism Design System** using CSS Backdrop-filter and Mesh Gradients.
- **UX Enhancements**: Added pagination logic and contact information injection into the recommendation response.
### Fixed
- **Quantization Pipeline**: Resolved Protocol Buffer loop errors in the `quantize_model.py` script, ensuring stable INT8 conversion.

## [2026-03-08 ~ 2026-03-09] - Capability Breakthrough
### Added
- **Hard Negative Mining (HNM)**: Implemented an active learning loop to identify and oversample samples where the previous model version failed (e.g., similar budget but different district).
- **Metric-Driven Training**: Integrated Accuracy and F1-Score tracking per epoch, ensuring model convergence during fine-tuning.
- **NER Integration**: Deployed a custom 10k-sample trained Named Entity Recognition (NER) model to parse spatial and budget constraints from raw user strings.
- **Scoring Logic**: Transitioned from simple weighted scoring to **Masked Cosine Similarity**, allowing the model to focus only on user-specified dimensions.

## [2026-03-07] - Project Inception
### Added
- **Foundation**: Established initial Web ML architecture using `transformers.js` and Flask.
- **Feature Set**: Implemented basic proximity budget rules and proximity-based scoring.
- **Data Ingestion**: Deployed initial crawlers for NCHU official housing and 3rd party platforms.

---
*Last Updated: 2026-05-08*

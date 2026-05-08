# Changelog

All notable changes to the **NCHU Rental Recommendation AI** project will be documented in this file.

## [2026-05-08] - Optimization for Generalization & Precision
### Added
- **Adversarial Training (FGM)**: Implemented Fast Gradient Method in `WeightedTrainer` to inject perturbations into embedding layers, improving semantic robustness.
- **LLM Hard Traps (500 samples)**: Generated 500 high-difficulty query-property pairs using Gemini API targeting subtle constraint violations (pet policies, utilities, locations).
- **Dropout Regularization**: Increased `hidden_dropout_prob` and `attention_probs_dropout_prob` to **0.15** to prevent overfitting.

### Changed
- **Training Schedule**: Increased `num_train_epochs` to 15 and set `early_stopping_patience` to 12.
- **Dataset Pipeline**: Updated `generate_dataset.py` to handle pre-paired dictionary samples from LLM augmentation.
- **Learning Rate**: Fine-tuned to `2e-5` for more stable convergence in the final ranking optimization phase.
- **Centralized Augmentation**: Merged trap generation logic into `augment_with_llm.py` and consolidated output to `llm_queries.json`.

---

## [2026-05-07] - Ranking-Aware Training & Calibration
### Added
- **Temperature Calibration (Option C)**: Introduced Logit Scaling with $T=2.0$ in the cross-entropy loss function to prevent sigmoid saturation and improve ranking resolution for NDCG.
- **Aggressive Weighting Scheme**: Implemented a non-linear weight map (Perfect: 15.0, Good: 4.0, None: 6.0) to prioritize high-relevance matches.

### Fixed
- **Security Audit**: Removed all hardcoded API keys and implemented `.env` file support for `GEMINI_API_KEY`.
- **Git Safety**: Updated `.gitignore` to strictly exclude sensitive credentials and large temporary model artifacts.

### Optimized
- **Inference Pipeline**: Successfully exported and quantized the model to INT8 ONNX format for edge deployment.

---

## [2026-05-06] - Core Pipeline & Data Integration
### Added
- **Multi-Source Crawlers**: Implemented automated data fetching from 591, Dcard, and Facebook rental groups.
- **Hard Negative Mining**: Established the initial hard negative sampling ratio (1:2.5) to boost classification precision.
- **Feature Engineering**: Implemented `property_to_text` canonical description logic for transformer input.

---

## [Project Goals]
- [x] Baseline F1 Score > 0.85
- [ ] Graded NDCG@5 > 0.85 (In progress with May 8th updates)
- [x] Edge AI Deployment (ONNX Quantized)
- [x] Automated Data Augmentation

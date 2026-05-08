# Changelog

All 140+ commits of the **NCHU Rental Recommendation AI** project summarized into key technical milestones.

## 🚀 [Phase 5] Extreme Optimization & Generalization (Current)
*Target: Hit NDCG@5 > 0.85*

### [2026-05-08] - Generalization & Adversarial Training
- **Adversarial Training (FGM)**: Integrated Fast Gradient Method to boost semantic robustness.
- **LLM Hard Traps**: Synthesized 500+ high-difficulty trap samples using Gemini API.
- **Dropout Regularization**: Increased dropout to 0.15 for better model stability.

### [2026-05-07] - Ranking Calibration
- **Temperature Scaling (T=2.0)**: Calibrated Softmax logits to improve ranking resolution.
- **Weighted Learning**: Implemented aggressive weights (Perfect: 15.0) for the ranking loss.

---

## 📈 [Phase 4] Precision Metrics & Commute Awareness (2026-04-27 to 2026-05-06)
*Target: Transition from Binary to Graded Evaluation*

- **Graded Metrics**: Adopted **NDCG@5** and **MRR** for statistical performance tracking.
- **Commute Logic**: Integrated OSRM data for precise travel time/distance scoring.
- **Hard Negative Mining**: Optimized pos:neg ratio to 1:2.5 to reduce false positives.
- **Budget Scaling**: Implemented Chinese numeral parsing (e.g., 6k, 1萬5) for budget extraction.

---

## 🎨 [Phase 3] Cross-Encoder & UI Overhaul (2026-03-18 to 2026-04-02)
*Target: Professional UX and Deep Semantic Matching*

- **Sentence-Pair Architecture**: Switched to a robust Cross-Encoder for direct query-property matching.
- **UI/UX Excellence**: Implemented Glassmorphism, Mesh Gradients, and Framer Motion-style animations.
- **Quantization**: Successfully implemented INT8 quantization to reduce model size for web delivery.
- **Stopwords & Cleaning**: Refined the NLP cleaning pipeline for higher training signal.

---

## 🤖 [Phase 2] Web AI & ONNX Integration (2026-03-08 to 2026-03-12)
*Target: 100% Client-side AI Inference*

- **ONNX Runtime Web**: Migrated inference from Python backend to browser-based WASM runtime.
- **Transformers.js**: Integrated Hugging Face's web framework for tokenizer compatibility.
- **NER Model**: Trained and deployed a 3-class NER model to extract location/budget/features.
- **Vercel Deployment**: Configured WASM and CORS headers for successful cloud hosting.

---

## 🏗️ [Phase 1] Foundation & Multi-Source ETL (2026-02-27 to 2026-03-07)
*Target: Data Acquisition and Backend Baseline*

- **Web Scrapers**: Developed automated crawlers for 591, Dcard, and FB Rental groups.
- **Flask Backend**: Built the initial Python recommendation API.
- **Keyword Engine**: Implemented the first-generation Regex-based matching logic.
- **Initial Dataset**: Collected and normalized the first batch of NCHU-area rental listings.

---

## [Project Stats]
- **Total Commits**: 140+
- **Model Architecture**: RoBERTa-tiny (RBT6)
- **Deployment Platform**: Vercel + ONNXRuntime Web
- **Key Metrics**: F1=0.87, NDCG@5 Target=0.85

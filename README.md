# NCHU AI Rental Recommendation System

A fully browser-side **AI-powered rental recommendation platform** for National Chung Hsing University (NCHU) students. Users describe their housing needs in natural language, and the system performs real-time semantic analysis using a custom-trained ALBERT NER model, then recommends the best-matching rental properties via a Content-Based Filtering algorithm.

> **Core Highlight**: The entire AI inference pipeline runs on **WebAssembly (WASM)** — no backend server, no API keys. Just open the webpage and go.

## Key Features

- **Custom-Trained NER Model** — Fine-tuned from `clue/albert_chinese_tiny` with 10,000+ annotated samples, achieving **99.99%** validation accuracy
- **Browser-Side Real-Time Inference** — Powered by ONNXRuntime-Web (WebAssembly), all model inference runs directly in the frontend with zero latency
- **Intelligent Semantic Parsing** — Supports Chinese numerals, English abbreviations (6K), and colloquial expressions
- **Content-Based Filtering** — Multi-dimensional CBF scoring algorithm covering budget, room type, distance, amenities, and 11 feature dimensions
- **Google Maps Integration** — Each property card embeds a live map for instant location awareness
- **Responsive Dark Theme** — Modern Glassmorphism design, fully optimized for both mobile and desktop

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML Model** | ALBERT (chinese_tiny) + PyTorch + HuggingFace Transformers |
| **Model Format** | ONNX (Open Neural Network Exchange) + External Data Weights |
| **Frontend Inference** | ONNXRuntime-Web (WebAssembly backend) |
| **Tokenizer** | Xenova/Transformers.js (`AutoTokenizer`) |
| **Frontend** | Vanilla HTML5 + CSS3 + JavaScript (ES Modules) |
| **Data Parsing** | PapaParse (CSV to JSON) |
| **Typography** | Google Fonts — Noto Sans TC |
| **Icons** | FontAwesome 6 |
| **Version Control** | Git + GitHub |

---

## Architecture

### System Flow

```
User Input (Natural Language)
        |
        v
+-------------------------+
|   Tokenizer (ALBERT)    |  <- Xenova/Transformers.js
|   Character Encoding    |
+-----------+-------------+
            | input_ids, attention_mask, token_type_ids
            v
+-------------------------+
|   ONNX Runtime (WASM)   |  <- model.onnx + model.onnx.data
|   Token Classification  |
|   3-Class NER Inference  |
+-----------+-------------+
            | logits -> argmax -> B-Target / I-Target / O
            v
+-------------------------+
|   Feature Extractor     |  <- tagFeatures() in inference.js
|   Structured Feature    |
|   Extraction            |
+-----------+-------------+
            | CBF Feature Vector
            v
+-------------------------+
|   Scoring Engine        |  <- recommend() in inference.js
|   Content-Based         |
|   Filtering             |
+-----------+-------------+
            | Top-K Results
            v
+-------------------------+
|   UI Renderer           |  <- app.js -> index.html
|   Property Cards + Maps |
+-------------------------+
```

### Directory Structure

```
Renting_model_ONNX/
├── index.html                  # Main page (rental recommendation UI)
├── styles.css                  # Dark theme stylesheet
├── app.js                      # Frontend interaction logic + card rendering
├── inference.js                # AI inference engine (NER + CBF scoring)
├── nchu_rental_info.csv        # Rental property database
│
├── custom_onnx_model_dir/      # ONNX model directory for frontend
│   ├── model.onnx              # ONNX model graph
│   ├── model.onnx.data         # External weight file (~16MB)
│   ├── tokenizer.json          # ALBERT tokenizer vocabulary
│   ├── tokenizer_config.json   # Tokenizer configuration
│   └── special_tokens_map.json # Special token mapping
│
├── README.md                   # Project documentation
└── .gitignore                  # Git ignore rules
```

### NER Labeling Format (BIO Tagging)

The model performs token-level classification using 3 labels:

| Label ID | Label Name | Description |
|----------|-----------|-------------|
| 0 | `O` | Outside — non-feature tokens (verbs, prepositions, etc.) |
| 1 | `B-Target` | Begin — first character of a feature entity |
| 2 | `I-Target` | Inside — subsequent characters of a feature entity |

**Example:**
```
Input:  want find budget six thousand within suite room
Labels: O    O    B      I   I        I      B     I
             |budget     |6000-range        |suite room
```

### CBF Recommendation Algorithm

The `recommend()` function scores each property across 11 feature dimensions:

| Feature Dimension | Max Score | Description |
|-------------------|-----------|-------------|
| Budget Match | 20 pts | Full score if difference <= 500, linear decay beyond |
| Region Match | 20 pts | Address contains specified region keyword |
| Room Type Match | 20 pts | Layout matches specified room type |
| Building Type | 15 pts | Building type matches specification |
| Distance Match | 25 pts | Distance within requested limit |
| Furniture/Amenities | 5 pts each | Per-item match against property listing |
| Security Features | 5 pts each | Per-item match against security listing |

**Hard Exclusion Rules:**
- Budget "above" constraint — properties below budget are excluded entirely
- Budget "below/within" constraint — properties exceeding budget are excluded entirely
- Gender restriction conflicts — excluded
- Pet restriction conflicts — excluded

---

## Prerequisites

- A modern web browser (Chrome / Edge / Firefox with WebAssembly support)
- Any static file server (e.g., Python's `http.server`)

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/eric20041027/Renting-recommendation-ONNX.git
cd Renting-recommendation-ONNX
```

### 2. Start a Local Development Server

```bash
python3 -m http.server 5002
```

### 3. Open in Browser

Navigate to [http://localhost:5002](http://localhost:5002)

### 4. Usage

Type your rental requirements in the input box at the bottom, for example:

- `Budget under 6000, suite room`
- `Dali district, private laundry, air conditioning`
- `5 min scooter ride, under 8K, pets allowed`
- `Near NCHU main gate, single room`

The system will parse your input in real time and recommend the best-matching properties.

---

## Model Training Guide

> Training scripts (`train_and_export_onnx.py`) and datasets (`train.json`, `test.json`) are kept locally and not included in the repository.

### Training Data Format

Each entry is a JSON object with `text` (character array) and `tags` (BIO label array):

```json
[
  {
    "text": ["b", "u", "d", "g", "e", "t", " ", "6", "K"],
    "tags": ["O", "O", "O", "O", "O", "O", "O", "B-Target", "I-Target"]
  }
]
```

### Run Training

```bash
python3 train_and_export_onnx.py
```

This script will:
1. Load `train.json` and `test.json`
2. Fine-tune `clue/albert_chinese_tiny` for token classification
3. Report validation accuracy after each epoch
4. Export the final model to ONNX format

### Deploy Updated Model

```bash
cp my_custom_model.onnx custom_onnx_model_dir/model.onnx
cp my_custom_model.onnx.data custom_onnx_model_dir/model.onnx.data
cp my_trained_albert/tokenizer.json custom_onnx_model_dir/
cp my_trained_albert/tokenizer_config.json custom_onnx_model_dir/
cp my_trained_albert/special_tokens_map.json custom_onnx_model_dir/
```

### Configurable Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_train_epochs` | 3 | Number of training epochs |
| `learning_rate` | 2e-5 | Learning rate |
| `per_device_train_batch_size` | 8 | Batch size per device |
| `max_length` | 16 | Maximum token sequence length |

---

## Deployment

This project is a fully static website and can be deployed to any static hosting service.

### GitHub Pages

1. Go to Repository Settings -> Pages
2. Set Source to `main` branch
3. Access via `https://your-username.github.io/repo-name/`

> **Note**: The model weight file is approximately 16MB. Initial page load may take a few seconds.

### Netlify / Vercel

Connect your GitHub repository directly for automatic deployment. No additional configuration required.

---

## Troubleshooting

### Model Loading Error: `Module.MountedFiles`

**Cause**: The ONNX model internally references `my_custom_model.onnx.data` as its external weight file name.

**Solution**: Ensure `inference.js` correctly maps the internal name to the actual file path:
```javascript
externalData: [{
    path: 'my_custom_model.onnx.data',
    data: window.location.origin + '/custom_onnx_model_dir/model.onnx.data'
}]
```

### Tokenizer Error: `local_files_only=true`

**Cause**: `env.localModelPath` is not pointing to the correct model directory.

**Solution**: Verify path settings in `inference.js`:
```javascript
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = window.location.origin + '/';
```

### Inaccurate Recommendations

1. **Check NER output**: Open browser DevTools Console and look for `ALBERT Segmented Words:` logs
2. **Add training data**: Generate more samples for poorly recognized sentence patterns
3. **Adjust scoring weights**: Modify score allocations in the `recommend()` function within `inference.js`

---

## License

This project is a course project for National Chung Hsing University.

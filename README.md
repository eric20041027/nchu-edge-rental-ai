# NCHU AI Rental Recommendation System (Frontend-Only ML Architecture)

This project is a **100% Serverless / Frontend-Only** rental recommendation web application designed for university students. By leveraging Web Machine Learning technologies, we successfully migrated a heavy Python-backend NLP Transformer model directly into the user's browser for offline inference.

Users can input their rental requirements in natural language (e.g., "Budget under 6000, near the main gate, suite with independent washing machine"), and the browser will instantly download and execute the ONNX model to analyze the semantic features and recommend the best properties.

## 💡 Capstone Highlights

- **100% Offline AI Inference**: Utilizes `ONNXRuntime-Web` to execute a 16MB ALBERT model directly within the browser ecosystem.
- **Zero Backend Costs & Latency**: Eliminates the need for Python servers. Achieves zero network latency for Content-Based Filtering (CBF) recommendations.
- **Edge Computing Privacy**: Semantic analysis is performed entirely on the client's device, ensuring maximum privacy for user inputs.
- **Seamless Static Deployment**: Capable of being deployed to static hosting platforms like Vercel or GitHub Pages in seconds.

## 🛠 Tech Stack

- **Core**: HTML5, CSS3, Vanilla JavaScript (ES6+)
- **Natural Language Processing (NLP)**:
  - [Transformers.js](https://huggingface.co/docs/transformers.js/index) (Tokenization)
  - [ONNX Runtime Web](https://onnxruntime.ai/docs/api/javascript/api/interfaces/Session.html) (WebAssembly Tensor Matrix Operations)
  - **Model**: ALBERT (A Lite BERT) exported to `.onnx` format
- **Data Parsing**: [PapaParse](https://www.papaparse.com/) (Fast client-side CSV parsing)
- **Deployment**: [Vercel](https://vercel.com) (Static Web Hosting)

## 🚀 Getting Started

Due to browser CORS (Cross-Origin Resource Sharing) security policies, you cannot load the local `.onnx` and `.csv` files simply by double-clicking the `index.html`. You must serve the application using a local HTTP server.

### 1. Clone the Repository

```bash
git clone https://github.com/eric20041027/Renting-recommendation-ONNX.git
cd Renting-recommendation-ONNX
```

### 2. Start the Local Server

You can use any local web server. Python's built-in `http.server` is commonly available:

```bash
# Start server on port 5002
python3 -m http.server 5002
```

### 3. Begin Inference

Open your browser and navigate to:
[http://localhost:5002](http://localhost:5002)

Upon the first visit, the browser will asynchronously cache the 16MB `model.onnx` file and the property database in the background. Once loaded, you can input your requirements for instant offline predictions.

## 🔬 Deep Dive: ONNX WebAssembly Tensor Conversions

The most technically challenging aspect of migrating from a PyTorch backend to a frontend JavaScript architecture is handling high-precision multi-dimensional matrix operations in a loosely-typed language.

Here is the precise workflow implemented in `inference.js`:

### 1. Tokenization
We utilize HuggingFace's open-source JS tools to load the vocabulary dictionary:
```javascript
const tokens = await tokenizer("Budget 6000 suite", { return_tensor: false });
```
This generates the required inputs for the neural network:
*   **`input_ids`**: The integer IDs mapped from the vocabulary dictionary (e.g., `[101, 7521, 5050... 102]`).
*   **`attention_mask`**: Identifies valid characters for gradient calculation (1 for valid, 0 for padding).
*   **`token_type_ids`**: Differentiates sentence structures (all 0 for a single sentence).

### 2. Casting Arrays to ONNX Tensors (Typed Memory Allocation)
WebAssembly (WASM) cannot directly read standard JS arrays. We must manually allocate and cast the data into contiguous memory blocks (`BigInt64Array`) that C++ can understand:
```javascript
// Allocate 64-bit high-precision integer memory with shape [Batch Size=1, Sequence Length]
const input_ids_tensor = new ort.Tensor(
    'int64', 
    BigInt64Array.from(tokens.input_ids.map(BigInt)), 
    [1, tokens.input_ids.length]
);
```

### 3. WASM Core Execution (Session Run)
The encapsulated tensors are passed into the `feeds` object, and `onnxruntime-web` initiates the matrix multiplication operations:
```javascript
const results = await session.run(feeds);
const logits = results.logits.data; // <- Float32Array
```
The output `logits` is an extremely flat, one-dimensional `Float32Array`. Its theoretical shape is `[1, sequence_length, 2]`, meaning every character in the sequence receives 2 hidden-layer probability values.

### 4. Decision Boundaries & Argmax Decoding
We iterate through all characters and compare their respective probability values.
By calculating the maximum value (Argmax), if index 0 is higher, it is classified as `B (Begin)`. If index 1 is higher, it is classified as `I (Inside)`. We reconstruct the continuous B/I characters into tangible requirements (e.g., Target: 6000), which are then evaluated by our Content-Based Filtering (CBF) recommendation algorithm against the CSV.

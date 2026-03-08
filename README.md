# NCHU AI Rental Recommendation System: Frontend-Only ML Architecture

This repository hosts a Serverless, Frontend-Only rental recommendation web application optimized for offline inference. By leveraging WebAssembly (WASM) and the Open Neural Network Exchange (ONNX) format, this project successfully migrates a transformer-based Natural Language Processing (NLP) model from a traditional Python backend to the client's browser.

## Architecture Overview

The system architecture focuses on entirely eliminating server-side inference by shifting the computational payload to the client device. This provides zero-latency interactions and complete data privacy for the user.

- **Inference Engine:** ONNX Runtime Web (`ort.min.js`)
- **Tokenization:** Transformers.js (`@xenova/transformers`)
- **Model Representation:** ALBERT (A Lite BERT) exported to `.onnx` statically
- **Data Filtering:** Client-side Content-Based Filtering (CBF) mapping via `PapaParse`
- **Deployment Strategy:** Vercel Static Hosting

## Core Mechanisms: ONNX WebAssembly Integration

Handling multi-dimensional matrix operations in a loosely-typed language like JavaScript requires explicit type casting and memory allocation. Below is the technical workflow implemented in the `inference.js` module:

### 1. Vocabulary Tokenization
Natural language input is encoded using the cached vocabulary model:
```javascript
const tokens = await tokenizer(inputText, { return_tensor: false });
```
This produces structured integer arrays representing the `input_ids`, `attention_mask`, and `token_type_ids` required for the transformer architecture.

### 2. Contiguous Memory Allocation (Tensor Casting)
WebAssembly bindings require parameters passed as strictly-typed contiguous memory blocks. We allocate a 64-bit high-precision integer array (`BigInt64Array`) representing the tensor shape `[Batch Size=1, Sequence Length]`:
```javascript
const input_ids_tensor = new ort.Tensor(
    'int64', 
    BigInt64Array.from(tokens.input_ids.map(BigInt)), 
    [1, tokens.input_ids.length]
);
```

### 3. Inference Execution via WASM
The defined tensors are passed into the cached ONNX session context. `onnxruntime-web` then computes the matrix operations locally:
```javascript
const results = await session.run(feeds);
const logits = results.logits.data; // Type: Float32Array
```
The output, `logits`, is returned as a flattened floating-point array corresponding to the theoretical shape `[1, sequence_length, 2]`. 

### 4. Argmax Decoding and Feature Extraction
The resulting probability distribution is iterated sequentially. An Argmax function calculates the predicted class (`B-Begin` or `I-Inside`) for each token. Discovered features are concatenated and piped to the Content-Based Filtering logic to score properties parsed from the `.csv` database.

## Local Development and Serving

Due to browser Strict MIME checking and CORS (Cross-Origin Resource Sharing) policies natively restricting `file://` protocols, the application requires a local HTTP daemon to fetch the `.csv` and `.onnx` artifacts.

### Prerequisites
- Python 3.x or Node.js

### Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/eric20041027/Renting-recommendation-ONNX.git
   cd Renting-recommendation-ONNX
   ```

2. Initialize a local web server:
   ```bash
   # Using Python 3
   python3 -m http.server 5002
   ```

3. Access the environment:
   Navigate to `http://localhost:5002`. The application will asynchronously fetch and cache the 16MB `.onnx` graph and the `.csv` corpus upon initialization.

## Production Deployment

This application operates entirely statelessly, bypassing the need for a WSGI or ASGI Python runtime.

To deploy via Vercel:
1. Import this repository into the Vercel dashboard.
2. Ensure the Framework Preset is configured as `Other`.
3. Overrides for Build Commands or Environment Variables are strictly unnecessary.
4. Deploy the application. Vercel will act strictly as a CDN for the `.onnx` model and static assets.

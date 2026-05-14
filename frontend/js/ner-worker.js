/**
 * ner-worker.js - NER inference worker (BertForTokenClassification, INT8 ONNX)
 *
 * Loads the quantized NER model and extracts LOC/BGT/FEAT entities from
 * user queries before the main constraint parser runs. Runs in a Web Worker
 * to avoid blocking the UI thread.
 *
 * Labels: O=0, B-LOC=1, I-LOC=2, B-BGT=3, I-BGT=4, B-FEAT=5, I-FEAT=6
 */

const MAX_LEN   = 64;
const CLS_ID    = 101;
const SEP_ID    = 102;
const PAD_ID    = 0;
const UNK_ID    = 100;

const ID2LABEL = {
    0: 'O',
    1: 'B-LOC', 2: 'I-LOC',
    3: 'B-BGT', 4: 'I-BGT',
    5: 'B-FEAT', 6: 'I-FEAT',
};

let ort   = null;
let vocab = null;  // Map<token_str, id>
let session = null;

// Cache API key — bump version string when NER model weights change
const NER_CACHE_NAME  = 'rental-models-v20260514';
const NER_MODEL_KEY   = 'ner-quant-v20260514';
const NER_VOCAB_KEY   = 'ner-tokenizer-v20260514';

/**
 * Loads a resource from Cache API, or fetches + streams it from network.
 * Reports download progress via onProgress(loaded, total) when downloading.
 */
async function cachedFetch(url, cacheKey, cacheName, onProgress = null) {
    try {
        const cache = await caches.open(cacheName);
        const cached = await cache.match(cacheKey);
        if (cached) return await cached.arrayBuffer();
    } catch (_) { /* caches unavailable */ }

    const response = await fetch(url);
    if (!response.ok) throw new Error(`Fetch failed: ${response.status} ${url}`);

    if (onProgress && response.body) {
        const contentLength = +response.headers.get('Content-Length') || 38000000;
        const reader = response.body.getReader();
        const chunks = [];
        let received = 0, lastReport = 0;
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
            received += value.length;
            if (received - lastReport > 512 * 1024) {
                onProgress(received, contentLength);
                lastReport = received;
            }
        }
        const buf = new Uint8Array(received);
        let pos = 0;
        for (const c of chunks) { buf.set(c, pos); pos += c.length; }

        try {
            const cache = await caches.open(cacheName);
            await cache.put(cacheKey, new Response(buf.buffer, {
                headers: { 'Content-Type': 'application/octet-stream' }
            }));
        } catch (_) { /* quota exceeded — non-fatal */ }

        return buf.buffer;
    }

    const buf = await response.arrayBuffer();
    try {
        const cache = await caches.open(cacheName);
        await cache.put(cacheKey, new Response(buf.slice(0), {
            headers: { 'Content-Type': 'application/octet-stream' }
        }));
    } catch (_) {}
    return buf;
}

// ── Initialization ──────────────────────────────────────────────────────────

async function init(origin) {
    try {
        postMessage({ type: 'ner_status', message: '載入 NER 模組...' });

        const ortModule = await import('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.mjs');
        ort = ortModule.default ?? ortModule;

        // Load vocab and model in parallel (both may be cached)
        postMessage({ type: 'ner_status', message: '載入 NER 詞表與模型...' });

        const [vocabBuf, modelBuf] = await Promise.all([
            cachedFetch(
                origin + '/models/ner_model_dir/tokenizer.json',
                NER_VOCAB_KEY,
                NER_CACHE_NAME
            ),
            cachedFetch(
                origin + '/models/ner_model_dir/ner_model_quant.onnx',
                NER_MODEL_KEY,
                NER_CACHE_NAME,
                (loaded, total) => postMessage({ type: 'ner_progress', loaded, total })
            )
        ]);

        const tokenizerJson = JSON.parse(new TextDecoder().decode(vocabBuf));
        vocab = new Map(Object.entries(tokenizerJson.model.vocab));

        session = await ort.InferenceSession.create(modelBuf, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all',
        });

        postMessage({ type: 'ner_ready' });
    } catch (err) {
        postMessage({ type: 'ner_error', message: err.message });
    }
}

// ── Tokenisation ────────────────────────────────────────────────────────────

function charToId(ch) {
    // Direct lookup; try lowercase as fallback, then UNK
    return vocab.get(ch) ?? vocab.get(ch.toLowerCase()) ?? UNK_ID;
}

function tokenize(text) {
    // bert-base-chinese: each CJK character is its own token
    // Spaces and punctuation are included if in vocab
    const chars = Array.from(text.replace(/\s+/g, ''));  // collapse whitespace

    const ids   = [CLS_ID];
    const chars_used = [];  // parallel array tracking original chars

    for (const ch of chars) {
        if (ids.length >= MAX_LEN - 1) break;  // leave room for [SEP]
        ids.push(charToId(ch));
        chars_used.push(ch);
    }
    const content_len = ids.length;  // length including [CLS] but before [SEP]
    ids.push(SEP_ID);

    const pad_len = MAX_LEN - ids.length;
    for (let i = 0; i < pad_len; i++) ids.push(PAD_ID);

    const attention_mask = ids.map((id, i) => (i < ids.length - pad_len) ? 1 : 0);
    const token_type_ids = new Array(MAX_LEN).fill(0);

    return { ids, attention_mask, token_type_ids, chars_used, content_len };
}

// ── Inference ───────────────────────────────────────────────────────────────

async function extractEntities(text) {
    if (!session || !vocab) return { locations: [], budgets: [], features: [] };

    const { ids, attention_mask, token_type_ids, chars_used, content_len } =
        tokenize(text);

    const toTensor = (arr) => new ort.Tensor(
        'int64',
        BigInt64Array.from(arr.map(v => BigInt(v))),
        [1, MAX_LEN]
    );

    const outputs = await session.run({
        input_ids:      toTensor(ids),
        attention_mask: toTensor(attention_mask),
        token_type_ids: toTensor(token_type_ids),
    });

    // logits: shape [1, MAX_LEN, 7]
    const logits = outputs.logits.data;  // flat Float32Array
    const num_labels = 7;

    // Decode: skip position 0 ([CLS]) and last content position ([SEP])
    // chars_used aligns with positions 1..content_len-1 (before [SEP])
    const pred_labels = [];
    for (let pos = 1; pos < content_len; pos++) {
        const offset = pos * num_labels;
        let best = 0, bestVal = logits[offset];
        for (let l = 1; l < num_labels; l++) {
            if (logits[offset + l] > bestVal) { bestVal = logits[offset + l]; best = l; }
        }
        pred_labels.push(ID2LABEL[best]);
    }

    // BIO span extraction
    const locations = [];
    const budgets   = [];
    const features  = [];

    let spanType = null;
    let spanChars = [];

    const flush = () => {
        if (!spanType || spanChars.length === 0) return;
        const text = spanChars.join('');
        if (spanType === 'LOC')  locations.push(text);
        if (spanType === 'BGT')  budgets.push(text);
        if (spanType === 'FEAT') features.push(text);
        spanChars = [];
        spanType  = null;
    };

    for (let i = 0; i < pred_labels.length; i++) {
        const label = pred_labels[i];
        const ch    = chars_used[i];

        if (label === 'O') {
            flush();
        } else if (label.startsWith('B-')) {
            flush();
            spanType  = label.slice(2);
            spanChars = [ch];
        } else if (label.startsWith('I-') && spanType === label.slice(2)) {
            spanChars.push(ch);
        } else {
            // I- without matching B- → treat as new B-
            flush();
            spanType  = label.slice(2);
            spanChars = [ch];
        }
    }
    flush();

    return { locations, budgets, features };
}

// ── Message handler ─────────────────────────────────────────────────────────

onmessage = async (e) => {
    const { type, data } = e.data;

    if (type === 'ner_init') {
        await init(data.origin);
    } else if (type === 'ner_extract') {
        const { query, id } = data;
        try {
            const entities = await extractEntities(query);
            postMessage({ type: 'ner_result', entities, id });
        } catch (err) {
            postMessage({ type: 'ner_result', entities: { locations: [], budgets: [], features: [] }, id });
        }
    }
};

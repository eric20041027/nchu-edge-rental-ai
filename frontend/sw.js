/**
 * sw.js — Service Worker for Rental Recommendation App
 *
 * Strategy:
 *  - Static assets (HTML/JS/CSS/JSON data): stale-while-revalidate
 *  - ONNX model files: cache-first (very large, rarely changes)
 *  - CDN assets (transformers.js / ORT): cache-first
 *
 * Version key: bump CACHE_VERSION when deploying new model weights or major JS changes.
 */

const CACHE_VERSION  = 'v20260614e';
const STATIC_CACHE   = `rental-static-${CACHE_VERSION}`;
const MODEL_CACHE    = `rental-models-${CACHE_VERSION}`;
const CDN_CACHE      = `rental-cdn-${CACHE_VERSION}`;

// Pre-cached on install (critical path assets only — keep small)
const PRECACHE_ASSETS = [
    '/',
    '/index.html',
    '/js/app.js',
    '/js/inference.js',
    '/js/inference-worker.js',
    '/js/ner-worker.js',
    '/assets/property_data.json',
    '/models/custom_onnx_model_dir/tokenizer.json',
    '/models/custom_onnx_model_dir/tokenizer_config.json',
    '/models/custom_onnx_model_dir/special_tokens_map.json',
    '/models/custom_onnx_model_dir/vocab.txt',
    '/models/custom_onnx_model_dir/config.json',
    '/models/ner_model_dir/tokenizer.json',
];

// ── Install: pre-cache static assets ─────────────────────────────────────────
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(STATIC_CACHE)
            .then(cache => cache.addAll(PRECACHE_ASSETS))
            .then(() => self.skipWaiting())
            .catch(err => {
                // Non-fatal: some assets may not be available at install time
                console.warn('[SW] Pre-cache partial failure:', err);
                return self.skipWaiting();
            })
    );
});

// ── Activate: remove stale caches ────────────────────────────────────────────
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys()
            .then(keys => Promise.all(
                keys
                    .filter(k => !k.endsWith(CACHE_VERSION))
                    .map(k => {
                        console.log('[SW] Deleting old cache:', k);
                        return caches.delete(k);
                    })
            ))
            .then(() => self.clients.claim())
    );
});

// ── Fetch: routing logic ──────────────────────────────────────────────────────
self.addEventListener('fetch', (event) => {
    const req = event.request;
    const url = new URL(req.url);

    // Only handle GET requests
    if (req.method !== 'GET') return;

    // 1. ONNX model files — cache-first (57 MB + 37 MB, never changes without deploy)
    if (url.pathname.endsWith('.onnx')) {
        event.respondWith(cacheFirst(req, MODEL_CACHE));
        return;
    }

    // 2. CDN assets (transformers.js, onnxruntime-web, fontawesome, etc.) — cache-first
    if (url.origin !== self.location.origin) {
        event.respondWith(cacheFirst(req, CDN_CACHE));
        return;
    }

    // 3. Model directory JSON/txt files — cache-first (vocab, config — almost never changes)
    if (url.pathname.includes('/models/')) {
        event.respondWith(cacheFirst(req, MODEL_CACHE));
        return;
    }

    // 4. Everything else (HTML, JS, CSS, property data) — stale-while-revalidate
    event.respondWith(staleWhileRevalidate(req, STATIC_CACHE));
});

// ── Cache strategies ──────────────────────────────────────────────────────────

async function cacheFirst(request, cacheName) {
    try {
        const cache  = await caches.open(cacheName);
        // Strip query string for ONNX files so ?v=xxx doesn't create duplicate entries
        const cacheKey = request.url.split('?')[0];
        const cached   = await cache.match(cacheKey);
        if (cached) return cached;

        const response = await fetch(request);
        if (response.ok) {
            cache.put(cacheKey, response.clone()); // async, non-blocking
        }
        return response;
    } catch (err) {
        // Offline and not in cache — return a minimal error response
        return new Response('Offline and not cached.', { status: 503 });
    }
}

async function staleWhileRevalidate(request, cacheName) {
    const cache  = await caches.open(cacheName);
    const cached = await cache.match(request);

    const networkFetch = fetch(request).then(response => {
        if (response.ok) cache.put(request, response.clone());
        return response;
    }).catch(() => null);

    return cached || await networkFetch || new Response('Offline.', { status: 503 });
}

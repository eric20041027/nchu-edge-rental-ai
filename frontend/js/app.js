import { initData, initNLP, initNER, recommend } from './inference.js';

// --- Constants & State ---
const PAGE_SIZE = 5;
let allRecommendedHouses = [];
let visibleCount = 0;

// --- DOM Elements ---
let userRequirement, welcomeScreen, resultsScreen, processingStatus, recommendationList, mainContent, chips, aiLoadingOverlay, loadingProgressFill;

document.addEventListener('DOMContentLoaded', async () => {
    // 1. Initialize DOM Elements
    userRequirement = document.getElementById('userRequirement');
    welcomeScreen = document.getElementById('welcomeScreen');
    resultsScreen = document.getElementById('resultsScreen');
    processingStatus = document.getElementById('processingStatus');
    recommendationList = document.getElementById('recommendationList');
    mainContent = document.getElementById('mainContent');
    chips = document.querySelectorAll('.chip');
    aiLoadingOverlay = document.getElementById('aiLoadingOverlay');
    loadingProgressFill = document.getElementById('loadingProgressFill');
    
    // 2. Setup AI and Handle Loading Status
    await setupApplication();

    // 3. Attach Event Listeners
    setupEventListeners();
});

// --- Application Initialization Setup ---
async function setupApplication() {
    // ── Progress bar UI ──────────────────────────────────────────────────────
    const loadStatus = document.createElement('div');
    loadStatus.id = 'load-status-panel';
    loadStatus.style.cssText = 'padding:12px 16px;color:var(--primary-color,#00FFD1);font-size:0.85rem;';
    loadStatus.innerHTML = `
        <div id="ls-ce" style="margin-bottom:6px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                <span><i class="fa-solid fa-brain"></i> 推薦模型</span>
                <span id="ls-ce-pct">等待中…</span>
            </div>
            <div style="background:rgba(255,255,255,0.1);border-radius:4px;height:6px;overflow:hidden;">
                <div id="ls-ce-bar" style="height:100%;width:0%;background:var(--primary-color,#00FFD1);transition:width 0.3s;border-radius:4px;"></div>
            </div>
        </div>
        <div id="ls-ner">
            <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                <span><i class="fa-solid fa-tag"></i> 語意模型</span>
                <span id="ls-ner-pct">等待中…</span>
            </div>
            <div style="background:rgba(255,255,255,0.1);border-radius:4px;height:6px;overflow:hidden;">
                <div id="ls-ner-bar" style="height:100%;width:0%;background:#a78bfa;transition:width 0.3s;border-radius:4px;"></div>
            </div>
        </div>`;
    welcomeScreen.insertBefore(loadStatus, welcomeScreen.children[2]);

    const ceBar  = document.getElementById('ls-ce-bar');
    const cePct  = document.getElementById('ls-ce-pct');
    const nerBar = document.getElementById('ls-ner-bar');
    const nerPct = document.getElementById('ls-ner-pct');

    function setBar(bar, pct, label, pctEl) {
        bar.style.width = pct + '%';
        pctEl.textContent = label;
    }

    userRequirement.disabled = true;
    userRequirement.placeholder = "請稍候，資料庫與 AI 模型準備中...";

    try {
        await Promise.all([
            initData(),
            initNLP((progress) => {
                if (progress.status === 'progress') {
                    // Detect cached load (loaded === total === 1 means instant from cache)
                    if (progress.message && progress.message.includes('快取')) {
                        setBar(ceBar, 100, '⚡ 快取', cePct);
                    } else if (progress.total > 0) {
                        const pct = Math.round((progress.loaded / progress.total) * 100);
                        setBar(ceBar, pct, pct + '%', cePct);
                    } else {
                        cePct.textContent = Math.round((progress.loaded || 0) / 1024) + ' KB';
                    }
                }
            })
        ]);
        setBar(ceBar, 100, '完成 ✓', cePct);
        ceBar.style.background = '#4ade80';

        userRequirement.disabled = false;
        userRequirement.placeholder = "輸入租屋需求，例如：預算 6000 以內、有冷氣...";

        // NER loads in background — update its bar via worker messages
        initNER(
            (nerProgress) => {
                if (nerProgress.loaded && nerProgress.total > 0) {
                    // loaded=1,total=1 → instant cache hit
                    if (nerProgress.loaded === nerProgress.total && nerProgress.total === 1) {
                        setBar(nerBar, 100, '⚡ 快取', nerPct);
                    } else {
                        const pct = Math.round((nerProgress.loaded / nerProgress.total) * 100);
                        setBar(nerBar, pct, pct + '%', nerPct);
                    }
                }
            },
            () => {
                setBar(nerBar, 100, '完成 ✓', nerPct);
                nerBar.style.background = '#4ade80';
                setTimeout(() => { loadStatus.style.display = 'none'; }, 1500);
            }
        ).catch(e => {
            console.warn('NER init failed (non-fatal):', e);
            nerPct.textContent = '略過';
            nerBar.style.background = '#f87171';
            setTimeout(() => { loadStatus.style.display = 'none'; }, 2000);
        });

    } catch (e) {
        console.error("Initialization error:", e);
        loadStatus.innerHTML = '<div style="color:#ff6b6b"><i class="fa-solid fa-triangle-exclamation"></i> 載入失敗，請刷新或確認網路。</div>';
    }
}

// --- Event Listeners Registration ---
function setupEventListeners() {
    // Textarea Auto-expand & Content check
    userRequirement.addEventListener('input', function () {
        const text = this.value.trim();
        if (!text) {
            resultsScreen.style.display = 'none';
            welcomeScreen.style.display = 'flex';
            this.style.height = '40px';
            return;
        }
        if (this.scrollHeight > this.clientHeight || this.value.length < (this.lastLen || 0)) {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 150) + 'px';
        }
        this.lastLen = this.value.length;
    });

    // Enter Key Search
    userRequirement.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            const text = userRequirement.value.trim();
            if (text) fetchRecommendations(text);
        }
    });

    // Suggestion Chips Click
    chips.forEach(chip => {
        chip.addEventListener('click', () => {
            userRequirement.value = chip.textContent;
            userRequirement.dispatchEvent(new Event('input'));
            userRequirement.focus();
            // Automatically trigger search
            fetchRecommendations(chip.textContent);
        });
    });

    // Analyze Button Click
    const btnAnalyze = document.getElementById('btnAnalyze');
    if (btnAnalyze) {
        btnAnalyze.addEventListener('click', () => {
            const text = userRequirement.value.trim();
            if (text) fetchRecommendations(text);
        });
    }
}

// --- Recommendation Core Logic ---
async function fetchRecommendations(inputText) {
    console.log("fetchRecommendations triggered with:", inputText);
    welcomeScreen.style.display = 'none';
    resultsScreen.style.display = 'block';
    processingStatus.style.display = 'flex';
    recommendationList.style.opacity = '0.4';
    mainContent.scrollTop = 0;
    
    // Show premium loading overlay
    if (aiLoadingOverlay) {
        aiLoadingOverlay.style.display = 'flex';
        document.getElementById('loadingText').innerText = "AI 正在尋找最適合的房源...";
        document.getElementById('loadingSubtext').innerText = "正在進行深度語意匹配";
        if (loadingProgressFill) loadingProgressFill.style.width = '10%';
    }

    // Yield to browser to paint the overlay before heavy computation starts
    await new Promise(resolve => setTimeout(resolve, 100));

    try {
        // No more hard-coded whitelist guard. Let the AI decide based on results.
        const isRelevant = true; // Always proceed to scoring stage

        if (!isRelevant && inputText.length > 1) {
            recommendationList.innerHTML = `<div style="text-align: center; color: #ff6b6b; padding: 2rem;">
                <i class="fa-solid fa-circle-question" style="font-size: 2rem; margin-bottom: 1rem; display: block;"></i>
                偵測到不相干的文字，請重新輸入更具體的租屋需求。<br>
                <small style="color: #aaa;">例如：「預算 6000 南區 套房」</small>
            </div>`;
            return;
        }

        // Progressive rendering: show rule-based results immediately, then AI re-ranks
        let partialShown = false;
        const data = await recommend(inputText, 20, (partialResults) => {
            if (partialResults && partialResults.length > 0) {
                partialShown = true;
                allRecommendedHouses = partialResults;
                visibleCount = 0;
                renderCards(true);
                // Keep showing spinner while AI re-ranks
                processingStatus.style.display = 'flex';
                recommendationList.style.opacity = '0.7';
                if (loadingProgressFill) loadingProgressFill.style.width = '50%';
            }
        });

        if (data === null) {
            return;
        } else if (data && data.length > 0) {
            // Check relevance based on score (Handle both "43%" string and 7.48 number)
            const scoreRaw = data[0].score || 0;
            const topScore = (typeof scoreRaw === 'string') 
                ? parseFloat(scoreRaw.replace('%', '')) 
                : scoreRaw;
            
            console.log("AI Top Match Score (Robust Parsed):", topScore);
            
            if (topScore < 5) {
                recommendationList.innerHTML = `<div style="text-align: center; color: #ff6b6b; padding: 2rem;">
                    <i class="fa-solid fa-circle-question" style="font-size: 3rem; margin-bottom: 1rem;"></i><br>
                    偵測到不相干的文字，請重新輸入更具體的租屋需求。<br>
                    <small style="color: #888;">例如：「預算 6000 南區 套房」</small>
                </div>`;
                return;
            }
            allRecommendedHouses = data;
            visibleCount = 0;
            renderCards(true);
        } else if (data && data.length === 0) {
            recommendationList.innerHTML = `<div style="text-align: center; color: white; padding: 2rem;">
                找不到符合條件的房屋，試著放寬預算或是區域限制吧！
            </div>`;
        } else if (!partialShown) {
            throw new Error("回傳格式不正確");
        }
    } catch (error) {
        console.error("Fetch Error:", error);
        recommendationList.innerHTML = `<div style="text-align: center; color: white; padding: 2rem;">無法取得推薦結果，請檢查系統狀態。</div>`;
    } finally {
        processingStatus.style.display = 'none';
        recommendationList.style.opacity = '1';
        if (aiLoadingOverlay) {
            if (loadingProgressFill) loadingProgressFill.style.width = '100%';
            setTimeout(() => {
                aiLoadingOverlay.style.display = 'none';
            }, 500);
        }
    }
}

// --- UI Rendering ---
function renderCards(reset = false) {
    if (reset) {
        recommendationList.innerHTML = '';
        visibleCount = 0;
    }

    if (allRecommendedHouses.length === 0) {
        recommendationList.innerHTML = `<div style="text-align: center; color: white; padding: 2rem;">找不到符合條件的房屋，試著放寬預算或是區域限制吧！</div>`;
        return;
    }

    const nextBatch = allRecommendedHouses.slice(visibleCount, visibleCount + PAGE_SIZE);
    
    nextBatch.forEach((house, index) => {
        const card = document.createElement('div');
        card.className = 'property-card';
        card.style.animationDelay = `${index * 0.1}s`;
        
        // Apply opacity directly to the card if there's a conflict
        if (house.conflict_reason) {
            card.style.opacity = '0.7';
        }

        let badgeClass = '';
        if (visibleCount === 0 && index === 0) {
            card.classList.add('top-match');
            badgeClass = 'premium';
        }

        card.innerHTML = createPropertyCardHTML(house, badgeClass);
        recommendationList.appendChild(card);
    });

    visibleCount += nextBatch.length;
    updateLoadMoreButton();
}

// --- HTML Template Generation ---
function createPropertyCardHTML(house, badgeClass) {
    const displayScore = Math.round(house.score);
    const imgUrl = house.imgUrl ? house.imgUrl : "https://images.unsplash.com/photo-1522708323590-d24dbb6b0267?auto=format&fit=crop&w=600&q=80";
    
    let commuteHtml = `<i class="fa-solid fa-location-dot"></i> 距離未提供`;
    let dist = parseFloat(house.distance);
    
    if (!isNaN(dist) && dist > 0) {
        let walkMins = house.walk_mins || Math.ceil(dist / 0.08); 
        let scooterMins = house.scooter_mins || Math.max(1, Math.ceil(dist / 0.5));
        
        const queryText = userRequirement ? userRequirement.value : "";
        let showWalk = walkMins <= 10;
        
        if (queryText.includes("走路") || queryText.includes("步行")) {
            showWalk = true;
        } else if (queryText.includes("機車") || queryText.includes("騎車")) {
            showWalk = false;
        }

        if (showWalk) {
            commuteHtml = `<i class="fa-solid fa-person-walking"></i> 走路約 ${walkMins} 分鐘 (${dist} 公里)`;
        } else {
            commuteHtml = `<i class="fa-solid fa-motorcycle"></i> 機車約 ${scooterMins} 分鐘 (${dist} 公里)`;
        }
    }

    // [Explainable AI] Generate Reason Tags
    let reasonsHtml = "";
    if (house.match_reasons && house.match_reasons.length > 0) {
        reasonsHtml = `<div class="match-reasons" style="display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px;">
            ${house.match_reasons.map(r => `<span style="background: rgba(0, 255, 209, 0.15); color: #00FFD1; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; border: 1px solid rgba(0, 255, 209, 0.3);"><i class="fa-solid fa-check"></i> ${r}</span>`).join('')}
        </div>`;
    }

    // [Hybrid Filtering] Conflict Warning
    let conflictHtml = "";
    if (house.conflict_reason) {
        conflictHtml = `<div class="conflict-alert" style="background: rgba(255, 107, 107, 0.15); color: #ff6b6b; padding: 8px 12px; border-radius: 8px; font-size: 0.8rem; margin-bottom: 12px; border: 1px solid rgba(255, 107, 107, 0.3); display: flex; align-items: center; gap: 8px;">
            <i class="fa-solid fa-triangle-exclamation"></i>
            <span>${house.conflict_reason}</span>
        </div>`;
    }

    const badgeText = badgeClass === 'premium' ? '<i class="fa-solid fa-crown"></i> 最佳推薦 TOP 1' : '系統推薦';

    return `
        <div class="card-image">
            <img src="${imgUrl}" alt="房間照片">
            <span class="badge ${badgeClass}">${badgeText}</span>
        </div>
        <div class="card-content">
            ${conflictHtml}
            <div class="card-price">NT$ ${house.price_str}</div>
            <h4 class="card-title" style="margin-bottom: 8px;">${house.title}</h4>
            
            ${reasonsHtml}

            <div class="card-details" style="display: flex; gap: 10px; font-size: 0.85rem; color: #ccc; margin-bottom: 5px;">
                <span><i class="fa-solid fa-vector-square"></i> ${house.size}</span>
                <span><i class="fa-solid fa-building"></i> ${house.floor}</span>
            </div>
            
            <details style="font-size: 0.8rem; color: #ccc; margin-bottom: 10px; cursor: pointer; background: rgba(255,255,255,0.03); padding: 5px 8px; border-radius: 6px;">
                <summary style="outline: none; font-weight: 500;"><i class="fa-solid fa-couch"></i> 查看附屬家具設施</summary>
                <div style="margin-top: 5px; line-height: 1.4; padding-left: 18px;">
                    ${house.furniture.split('/').join(', ')}
                </div>
            </details>
            
            <div style="font-size: 0.85rem; color: var(--primary-color); margin-bottom: 12px; font-weight: 500;">
                ${commuteHtml}
            </div>

            <div class="features-grid" style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin-bottom: 15px; font-size: 0.8rem; color: #eee;">
                ${(house.features || house.特色 || "").split('/').map(f => {
                    f = f.trim();
                    if (!f) return '';
                    let icon = "fa-check";
                    let label = f;
                    
                    if (f.includes("台電")) { icon = "fa-bolt"; label = "電費照台電"; }
                    else if (f.includes("台水")) { icon = "fa-droplet"; label = "水費照台水"; }
                    else if (f.includes("補助") || f.includes("租補")) { icon = "fa-hand-holding-dollar"; label = "可申請補助"; }
                    else if (f.includes("窗")) icon = "fa-window-maximize";
                    else if (f.includes("陽台")) icon = "fa-house-chimney-window";
                    else if (f.includes("電梯")) icon = "fa-elevator";
                    else if (f.includes("垃圾")) icon = "fa-trash-can";
                    else if (f.includes("隔間")) icon = "fa-border-all";
                    else if (f.includes("開伙")) icon = "fa-fire-burner";
                    else if (f.includes("飲水機")) icon = "fa-faucet-drip";
                    else if (f.includes("寵物")) icon = "fa-paw";
                    else if (f.includes("保全") || f.includes("監控")) icon = "fa-shield-halved";
                    
                    return `<div style="display: flex; align-items: center; gap: 6px;">
                        <i class="fa-solid ${icon}" style="width: 16px; color: var(--primary-color); opacity: 0.8;"></i>
                        <span>${label}</span>
                    </div>`;
                }).join('')}
            </div>

            <div class="contact-info" style="margin-bottom: 15px; padding: 10px; background: rgba(255, 255, 255, 0.03); border-radius: 12px; border: 1px solid var(--border-glass);">
                <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 5px; display: flex; align-items: center; gap: 8px;">
                    <i class="fa-solid fa-user-tie" style="color: var(--primary-color);"></i>
                    <span>聯絡人：${(house.contact || '不具名').replace(/^(聯絡)?人[:：]\s*/, '')}</span>
                </div>
                <div style="font-size: 0.95rem; color: #F8FAFC; display: flex; align-items: center; gap: 8px;">
                    <i class="fa-solid fa-phone" style="color: var(--accent-color);"></i>
                    <a href="tel:${house.phone}" style="color: inherit; text-decoration: none; font-weight: 600;">${house.phone || '無資料'}</a>
                </div>
            </div>

            <div class="map-container" style="margin-bottom: 15px; border-radius: 8px; overflow: hidden; height: 120px;">
                <iframe width="100%" height="100%" frameborder="0" style="border:0" 
                    src="https://maps.google.com/maps?q=${encodeURIComponent(house.address)}&output=embed" 
                    allowfullscreen>
                </iframe>
            </div>
            <div class="card-link">
                <a href="${house.url}" target="_blank" style="color: var(--primary-color); text-decoration: none; font-size: 0.9rem; display: inline-block;">
                    <i class="fa-solid fa-link"></i> 前往查看物件
                </a>
            </div>

            <div class="feedback-bar" style="display:flex; align-items:center; gap:10px; margin-top:12px; padding-top:10px; border-top:1px solid var(--border-glass); font-size:0.8rem; color:#94A3B8;">
                <span>這個推薦有幫助嗎？</span>
                <button class="feedback-btn" data-id="${house.id || house.url}" data-v="1"
                    style="background:none; border:1px solid rgba(255,255,255,0.15); border-radius:20px; padding:3px 10px; cursor:pointer; color:#94A3B8; transition:all 0.2s;"
                    onmouseover="this.style.borderColor='#4ade80';this.style.color='#4ade80'"
                    onmouseout="if(!this.dataset.active){this.style.borderColor='rgba(255,255,255,0.15)';this.style.color='#94A3B8'}">
                    <i class="fa-solid fa-thumbs-up"></i> 有用
                </button>
                <button class="feedback-btn" data-id="${house.id || house.url}" data-v="-1"
                    style="background:none; border:1px solid rgba(255,255,255,0.15); border-radius:20px; padding:3px 10px; cursor:pointer; color:#94A3B8; transition:all 0.2s;"
                    onmouseover="this.style.borderColor='#f87171';this.style.color='#f87171'"
                    onmouseout="if(!this.dataset.active){this.style.borderColor='rgba(255,255,255,0.15)';this.style.color='#94A3B8'}">
                    <i class="fa-solid fa-thumbs-down"></i> 不符
                </button>
                <span class="feedback-thanks" style="display:none; color:#4ade80; font-size:0.78rem;">感謝回饋！</span>
            </div>
        </div>
    `;
}

function updateLoadMoreButton() {
    const oldBtn = document.getElementById('btnLoadMore');
    if (oldBtn) oldBtn.remove();

    if (visibleCount < allRecommendedHouses.length) {
        const loadMoreBtn = document.createElement('button');
        loadMoreBtn.id = 'btnLoadMore';
        loadMoreBtn.className = 'btn-load-more';
        loadMoreBtn.innerHTML = '<i class="fa-solid fa-chevron-down"></i> 載入更多推薦';
        loadMoreBtn.onclick = () => renderCards(false);
        recommendationList.appendChild(loadMoreBtn);
    }
}

// ── Feedback system ────────────────────────────────────────────────────────────
const FEEDBACK_KEY = 'renting_feedback_log';

function loadFeedbackLog() {
    try { return JSON.parse(localStorage.getItem(FEEDBACK_KEY) || '[]'); }
    catch { return []; }
}

function saveFeedback(query, propertyId, vote) {
    const log = loadFeedbackLog();
    log.push({
        ts:         new Date().toISOString(),
        query:      query,
        propertyId: propertyId,
        vote:       vote,   // 1 = helpful, -1 = not relevant
    });
    // Keep latest 500 entries to avoid unbounded growth
    if (log.length > 500) log.splice(0, log.length - 500);
    localStorage.setItem(FEEDBACK_KEY, JSON.stringify(log));
}

// Event delegation — one listener on the list container handles all cards
document.addEventListener('DOMContentLoaded', () => {
    const list = document.getElementById('recommendation-list') ||
                 document.querySelector('.recommendation-list');
    if (!list) return;

    list.addEventListener('click', e => {
        const btn = e.target.closest('.feedback-btn');
        if (!btn) return;

        const bar    = btn.closest('.feedback-bar');
        const propId = btn.dataset.id;
        const vote   = parseInt(btn.dataset.v, 10);
        const query  = document.getElementById('query-input')?.value ||
                       document.getElementById('searchInput')?.value || '';

        // Persist
        saveFeedback(query, propId, vote);

        // Visual feedback
        bar.querySelectorAll('.feedback-btn').forEach(b => {
            b.dataset.active = '';
            b.style.borderColor = 'rgba(255,255,255,0.15)';
            b.style.color = '#94A3B8';
        });
        btn.dataset.active = '1';
        if (vote === 1) {
            btn.style.borderColor = '#4ade80';
            btn.style.color = '#4ade80';
        } else {
            btn.style.borderColor = '#f87171';
            btn.style.color = '#f87171';
        }
        const thanks = bar.querySelector('.feedback-thanks');
        if (thanks) thanks.style.display = 'inline';
    });
});

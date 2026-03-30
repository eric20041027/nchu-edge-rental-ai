import { initData, initNLP, recommend } from './inference.js?v=20260311_V2';

document.addEventListener('DOMContentLoaded', async () => {
    const userRequirement = document.getElementById('userRequirement');
    const welcomeScreen = document.getElementById('welcomeScreen');
    const resultsScreen = document.getElementById('resultsScreen');
    const processingStatus = document.getElementById('processingStatus');
    const recommendationList = document.getElementById('recommendationList');
    const mainContent = document.getElementById('mainContent');
    const chips = document.querySelectorAll('.chip');
    
    // Pagination State
    let allRecommendedHouses = [];
    let visibleCount = 0;
    const PAGE_SIZE = 5;

    // Initialize loading status indicator
    const loadStatus = document.createElement('div');
    loadStatus.style.padding = '10px';
    loadStatus.style.color = '#64ffda';
    loadStatus.style.textAlign = 'center';
    loadStatus.style.fontSize = '0.9rem';
    loadStatus.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> 正在背景加載房屋資料...';
    welcomeScreen.insertBefore(loadStatus, welcomeScreen.children[2]);

    userRequirement.disabled = true;
    userRequirement.placeholder = "請稍候，資料庫與 AI 模型準備中...";

    try {
        await Promise.all([
            initData(),
            initNLP((progress) => {
                if (progress.status === 'progress') {
                    let percent = Math.round((progress.loaded / progress.total) * 100);
                    loadStatus.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> 正在下載 AI 模組與資料 (${percent}%)...`;
                }
            })
        ]);
        loadStatus.innerHTML = '<i class="fa-solid fa-check"></i> 系統準備就緒！';
        setTimeout(() => loadStatus.style.display = 'none', 2000);
        userRequirement.disabled = false;
        userRequirement.placeholder = "輸入租屋需求，例如：預算 6000 以內、有冷氣...";
    } catch (e) {
        console.error("Initialization error:", e);
        loadStatus.innerHTML = '<i class="fa-solid fa-triangle-exclamation" style="color: #ff6b6b"></i> 載入失敗，請刷新或確認網路。';
        loadStatus.style.color = '#ff6b6b';
    }

    let debounceTimer;

    userRequirement.addEventListener('input', function () {
        const text = this.value.trim();

        if (!text) {
            if (resultsScreen.style.display !== 'none') {
                resultsScreen.style.display = 'none';
                welcomeScreen.style.display = 'flex';
            }
            this.style.height = '40px';
            return;
        }

        if (this.scrollHeight > this.clientHeight || this.value.length < (this.lastLen || 0)) {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 150) + 'px';
        }
        this.lastLen = this.value.length;
    });

    userRequirement.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            const text = userRequirement.value.trim();
            if (text) fetchRecommendations(text);
        }
    });

    chips.forEach(chip => {
        chip.addEventListener('click', () => {
            userRequirement.value = chip.textContent;
            userRequirement.dispatchEvent(new Event('input'));
            userRequirement.focus();
        });
    });

    async function fetchRecommendations(inputText) {
        console.log("fetchRecommendations triggered with:", inputText);
        welcomeScreen.style.display = 'none';
        resultsScreen.style.display = 'block';
        processingStatus.style.display = 'flex';
        recommendationList.style.opacity = '0.4';
        mainContent.scrollTop = 0;

        try {
            const housingKeywords = [
                '房', '租', '預算', '萬', '千', 'k', '元', '近', '走', '分', '坪', '樓', 
                '東區', '南區', '西區', '大里', '中興', '興大', '路', '街', '巷', '大道', 
                '套', '雅', '工', '學', '國光', '學府', '忠明'
            ];
            const isRelevant = housingKeywords.some(key => inputText.toLowerCase().includes(key)) || /\d+/.test(inputText);

            if (!isRelevant && inputText.length > 1) {
                recommendationList.innerHTML = `<div style="text-align: center; color: #ff6b6b; padding: 2rem;">
                    <i class="fa-solid fa-circle-question" style="font-size: 2rem; margin-bottom: 1rem; display: block;"></i>
                    偵測到不相干的文字，請重新輸入更具體的租屋需求。<br>
                    <small style="color: #aaa;">例如：「預算 6000 南區 套房」</small>
                </div>`;
                return;
            }

            const data = await recommend(inputText, 20);

            if (data && data.length >= 0) {
                allRecommendedHouses = data;
                visibleCount = 0;
                renderCards(true); // Initial render
            } else {
                throw new Error("回傳格式不正確");
            }
        } catch (error) {
            console.error("Fetch Error:", error);
            recommendationList.innerHTML = `<div style="text-align: center; color: white; padding: 2rem;">無法取得推薦結果，請檢查系統狀態。</div>`;
        } finally {
            processingStatus.style.display = 'none';
            recommendationList.style.opacity = '1';
        }
    }

    // Render recommendation results as property cards
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
            // Add staggered entrance animation delay
            card.style.animationDelay = `${index * 0.1}s`;

            let badgeClass = '';
            let displayScore = Math.round(house.score);

            if (visibleCount === 0 && index === 0) {
                card.classList.add('top-match');
                badgeClass = 'premium';
            }

            const imgUrl = house.imgUrl ? house.imgUrl : "https://images.unsplash.com/photo-1522708323590-d24dbb6b0267?auto=format&fit=crop&w=600&q=80";

            let commuteHtml = '';
            let dist = parseFloat(house.distance);
            if (!isNaN(dist) && dist > 0) {
                let walkMins = Math.round(dist / 0.075);
                let scooterMins = Math.max(1, Math.round(dist / 0.417));
                if (walkMins <= 10) {
                    commuteHtml = `<i class="fa-solid fa-person-walking"></i> 走路約 ${walkMins} 分鐘 (${dist} 公里)`;
                } else {
                    commuteHtml = `<i class="fa-solid fa-motorcycle"></i> 機車約 ${scooterMins} 分鐘 (${dist} 公里)`;
                }
            } else {
                commuteHtml = `<i class="fa-solid fa-location-dot"></i> 距離未提供`;
            }

            card.innerHTML = `
                <div class="card-image">
                    <img src="${imgUrl}" alt="房間照片">
                    <span class="badge ${badgeClass}">配對相符度 ${displayScore}%</span>
                </div>
                <div class="card-content">
                    <div class="card-price">NT$ ${house.price_str}</div>
                    <h4 class="card-title">${house.title}</h4>
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

                    <div class="contact-info" style="margin-bottom: 15px; padding: 10px; background: rgba(255, 255, 255, 0.03); border-radius: 12px; border: 1px solid var(--border-glass);">
                        <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 5px; display: flex; align-items: center; gap: 8px;">
                            <i class="fa-solid fa-user-tie" style="color: var(--primary-color);"></i>
                            <span>聯絡人：${house.contact || '不具名'}</span>
                        </div>
                        <div style="font-size: 0.95rem; color: #F8FAFC; display: flex; align-items: center; gap: 8px;">
                            <i class="fa-solid fa-phone" style="color: var(--accent-color);"></i>
                            <a href="tel:${house.phone}" style="color: inherit; text-decoration: none; font-weight: 600;">${house.phone || '無資料'}</a>
                        </div>
                    </div>

                    <div class="map-container" style="margin-bottom: 15px; border-radius: 8px; overflow: hidden; height: 120px;">
                        <iframe 
                            width="100%" 
                            height="100%" 
                            frameborder="0" 
                            style="border:0" 
                            src="https://maps.google.com/maps?q=${encodeURIComponent(house.address)}&output=embed" 
                            allowfullscreen>
                        </iframe>
                    </div>
                    <div class="card-link">
                        <a href="${house.url}" target="_blank" style="color: #64ffda; text-decoration: none; font-size: 0.9rem; display: inline-block;">
                            <i class="fa-solid fa-link"></i> 前往查看物件
                        </a>
                    </div>
                </div>
            `;
            recommendationList.appendChild(card);
        });

        visibleCount += nextBatch.length;
        updateLoadMoreButton();
    }

    // Logic to show/hide 'Load More' button based on remaining results
    function updateLoadMoreButton() {
        // 移除舊的按鈕
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

    // Analysis button click handler
    const btnAnalyze = document.getElementById('btnAnalyze');
    if (btnAnalyze) {
        btnAnalyze.addEventListener('click', () => {
            const text = userRequirement.value.trim();
            if (text) fetchRecommendations(text);
        });
    }
});

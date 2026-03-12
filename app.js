console.log(">>> APP_VERSION: 20260311_V2 <<<");
import { initData, initNLP, recommend } from './inference.js?v=20260311_V2';

document.addEventListener('DOMContentLoaded', async () => {
    const userRequirement = document.getElementById('userRequirement');
    const welcomeScreen = document.getElementById('welcomeScreen');
    const resultsScreen = document.getElementById('resultsScreen');
    const processingStatus = document.getElementById('processingStatus');
    const recommendationList = document.getElementById('recommendationList');
    const mainContent = document.getElementById('mainContent');
    const chips = document.querySelectorAll('.chip');

    // 初始化加載提示
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

        // 1. 處理介面切換 (僅在此處檢查，且只在狀態改變時更新 DOM)
        if (!text) {
            if (resultsScreen.style.display !== 'none') {
                resultsScreen.style.display = 'none';
                welcomeScreen.style.display = 'flex';
            }
            this.style.height = '40px';
            return;
        }

        // 2. 效率化調整高度
        if (this.scrollHeight > this.clientHeight || this.value.length < (this.lastLen || 0)) {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 150) + 'px';
        }
        this.lastLen = this.value.length;
    });

    // 支援按下 Enter 送出
    userRequirement.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault(); // 避免換行
            const text = userRequirement.value.trim();
            if (text) fetchRecommendations(text);
        }
    });

    // 點擊建議標籤快速輸入
    chips.forEach(chip => {
        chip.addEventListener('click', () => {
            userRequirement.value = chip.textContent;
            // 觸發 input 事件與自動拉高
            userRequirement.dispatchEvent(new Event('input'));
            userRequirement.focus(); // 輸入框保持 Focus
        });
    });

    // 動態變更推薦結果
    async function fetchRecommendations(inputText) {
        console.log("fetchRecommendations triggered with:", inputText);
        // console.trace("Trace for trigger:"); // 用於調試意外觸發

        // 先跳到結果螢幕並顯示讀取中
        welcomeScreen.style.display = 'none';
        resultsScreen.style.display = 'block';
        processingStatus.style.display = 'flex';
        recommendationList.style.opacity = '0.4';
        mainContent.scrollTop = 0;

        try {
            // 關鍵字初步過濾 (檢查是否為租屋相關或純數字預算)
            const housingKeywords = ['房', '租', '預算', '萬', '千', 'k', '元', '近', '走', '分', '坪', '樓', '東區', '南區', '西區', '大里', '中興', '興大'];
            const isRelevant = housingKeywords.some(key => inputText.toLowerCase().includes(key)) || /\d+/.test(inputText);

            if (!isRelevant && inputText.length > 1) {
                recommendationList.innerHTML = `<div style="text-align: center; color: #ff6b6b; padding: 2rem;">
                    <i class="fa-solid fa-circle-question" style="font-size: 2rem; margin-bottom: 1rem; display: block;"></i>
                    偵測到不相干的文字，請重新輸入更具體的租屋需求。<br>
                    <small style="color: #aaa;">例如：「預算 6000 南區 套房」</small>
                </div>`;
                return;
            }

            const data = await recommend(inputText, 5);

            if (data && data.length >= 0) {
                renderCards(data);
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

    // 將 API 傳回的資料轉譯為 HTML 卡片
    function renderCards(houses) {
        recommendationList.innerHTML = '';

        if (houses.length === 0) {
            recommendationList.innerHTML = `<div style="text-align: center; color: white; padding: 2rem;">找不到符合條件的房屋，試著放寬預算或是區域限制吧！</div>`;
            return;
        }

        houses.forEach((house, index) => {
            const card = document.createElement('div');
            card.className = 'property-card';

            let badgeClass = '';
            // 由於已經在後端轉化為 0~100 的數字，直接取用
            let displayScore = Math.round(house.score);

            if (index === 0) {
                card.classList.add('top-match');
                badgeClass = 'premium';
            }

            // 使用後端傳來的真實圖片網址，如果沒有就給個預設圖片
            const imgUrl = house.imgUrl ? house.imgUrl : "https://images.unsplash.com/photo-1522708323590-d24dbb6b0267?auto=format&fit=crop&w=600&q=80";

            let commuteHtml = '';
            let dist = parseFloat(house.distance);
            if (!isNaN(dist) && dist > 0) {
                // 走路時速約 4.5 km/h → 每分鐘走 0.075 km
                let walkMins = Math.round(dist / 0.075);
                // 機車市區均速約 25 km/h → 每分鐘走 0.417 km
                let scooterMins = Math.max(1, Math.round(dist / 0.417));

                if (walkMins <= 10) {
                    // 步行 10 分鐘以內顯示走路
                    commuteHtml = `<i class="fa-solid fa-person-walking"></i> 走路約 ${walkMins} 分鐘 (${dist} 公里)`;
                } else {
                    // 超過 10 分鐘步行改顯示機車
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
                    
                    <details style="font-size: 0.8rem; color: #ccc; margin-bottom: 10px; cursor: pointer; background: rgba(15,23,42,0.4); padding: 5px 8px; border-radius: 6px;">
                        <summary style="outline: none; font-weight: 500;"><i class="fa-solid fa-couch"></i> 查看附屬家具設施</summary>
                        <div style="margin-top: 5px; line-height: 1.4; padding-left: 18px;">
                            ${house.furniture.split('/').join(', ')}
                        </div>
                    </details>
                    
                    <div style="font-size: 0.85rem; color: #64ffda; margin-bottom: 10px; font-weight: 500;">
                        ${commuteHtml}
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
    }

    // 發送按鈕的點擊
    const btnAnalyze = document.getElementById('btnAnalyze');
    if (btnAnalyze) {
        btnAnalyze.addEventListener('click', () => {
            const text = userRequirement.value.trim();
            if (text) fetchRecommendations(text);
        });
    }
});

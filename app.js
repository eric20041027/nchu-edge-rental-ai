import { initData, recommend } from './inference.js';

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
    userRequirement.placeholder = "請稍候，資料庫準備中...";

    try {
        await initData();
        loadStatus.innerHTML = '<i class="fa-solid fa-check"></i> 系統準備就緒！';
        setTimeout(() => loadStatus.style.display = 'none', 2000);
        userRequirement.disabled = false;
        userRequirement.placeholder = "輸入租屋需求，例如：預算 6000 以內、有冷氣...";
    } catch (e) {
        console.error("Initialization error:", e);
        loadStatus.innerHTML = '<i class="fa-solid fa-triangle-exclamation" style="color: #ff6b6b"></i> 載入失敗，請刷新或確認網路。';
        loadStatus.style.color = '#ff6b6b';
    }

    // 初始化 textarea 高度
    userRequirement.style.height = "auto";
    userRequirement.style.height = (userRequirement.scrollHeight) + "px";

    // 處理輸入框自動長高
    userRequirement.addEventListener('input', function () {
        this.style.height = 'auto'; // 重置高度
        this.style.height = (this.scrollHeight) + 'px'; // 設為內容高度

        // 限縮最高高度
        if (this.scrollHeight > 120) {
            this.style.overflowY = 'auto';
        } else {
            this.style.overflowY = 'hidden';
            // 自動滾到底部確保輸入框在視野內
            mainContent.scrollTop = mainContent.scrollHeight;
        }
    });

    let debounceTimer;

    // 監聽文字輸入事件 (即時響應)
    userRequirement.addEventListener('input', () => {
        const text = userRequirement.value.trim();

        // 如果清空內容，則顯示歡迎畫面，隱藏結果
        if (!text) {
            resultsScreen.style.display = 'none';
            welcomeScreen.style.display = 'flex';
            return;
        }

        // 當有字輸入時，切換為結果畫面
        welcomeScreen.style.display = 'none';
        resultsScreen.style.display = 'block';

        // 每次打字時，顯示小型的運算動畫
        processingStatus.style.display = 'flex';
        // 降低透明度模擬正在更新
        recommendationList.style.opacity = '0.4';

        // 保證結果區是在畫面上方的，讓使用者不必下滑
        mainContent.scrollTop = 0;

        // 清除上一次的計時器
        clearTimeout(debounceTimer);

        // 設定 600ms 後使用者如果沒繼續打字，才執行尋找
        debounceTimer = setTimeout(() => {
            // 執行真正的 API 請求
            fetchRecommendations(text).then(() => {
                // 下載完成後隱藏運算動畫，恢復透明度
                processingStatus.style.display = 'none';
                recommendationList.style.opacity = '1';
            });
        }, 600);
    });

    // 支援按下 Enter 送出
    userRequirement.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault(); // 避免換行
            clearTimeout(debounceTimer); // 取消輸入延遲防抖
            const text = userRequirement.value.trim();
            if (text) {
                // 立即觸發 API 更新
                processingStatus.style.display = 'flex';
                recommendationList.style.opacity = '0.4';
                fetchRecommendations(text).then(() => {
                    processingStatus.style.display = 'none';
                    recommendationList.style.opacity = '1';
                });
            }
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
        try {
            const data = await recommend(inputText, 5);

            if (data && data.length >= 0) {
                renderCards(data);
            } else {
                throw new Error("回傳格式不正確");
            }
        } catch (error) {
            console.error("Fetch Error:", error);
            // 發生錯誤時顯示友善提示
            recommendationList.innerHTML = `<div style="text-align: center; color: white; padding: 2rem;">無法取得推薦結果，請檢查系統狀態。</div>`;
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
            if (house.distance > 0) {
                // 如果距離 <= 0.42 km (約 5 min 步行，時速 5 km/h)
                if (house.distance <= 0.42) {
                    let walkMins = Math.ceil(house.distance / 0.083);
                    commuteHtml = `<i class="fa-solid fa-person-walking"></i> 走路約 ${walkMins} 分鐘 (${house.distance} 公里)`;
                } else {
                    // 大於 0.42 km 使用機車計算，市區均速約 30 km/h (0.5 km/min)
                    let scooterMins = Math.ceil(house.distance / 0.5);
                    commuteHtml = `<i class="fa-solid fa-motorcycle"></i> 機車約 ${scooterMins} 分鐘 (${house.distance} 公里)`;
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
                    <div style="font-size: 0.8rem; color: #aaa; margin: 0.5rem 0;" class="match-details">
                        <i class="fa-solid fa-check"></i> 系統加分紀錄：${house.match_details}
                    </div>
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

    // 發送按鈕的點擊與鍵盤 Enter 直接觸發防抖立刻執行
    const btnAnalyze = document.getElementById('btnAnalyze');
    if (btnAnalyze) {
        btnAnalyze.addEventListener('click', () => {
            clearTimeout(debounceTimer);
            const text = userRequirement.value.trim();
            if (text) fetchRecommendations(text);
            processingStatus.style.display = 'none';
            recommendationList.style.opacity = '1';
        });
    }
});

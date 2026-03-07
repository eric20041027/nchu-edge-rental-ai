document.addEventListener('DOMContentLoaded', () => {
    const userRequirement = document.getElementById('userRequirement');
    const welcomeScreen = document.getElementById('welcomeScreen');
    const resultsScreen = document.getElementById('resultsScreen');
    const processingStatus = document.getElementById('processingStatus');
    const recommendationList = document.getElementById('recommendationList');
    const mainContent = document.getElementById('mainContent');
    const chips = document.querySelectorAll('.chip');

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
            // 隱藏運算動畫
            processingStatus.style.display = 'none';

            // 模擬更新結果
            updateMockResults(text);

            // 恢復透明度
            recommendationList.style.opacity = '1';

        }, 600);
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
            const response = await fetch('/api/recommend', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: inputText })
            });

            if (!response.ok) {
                throw new Error(`伺服器錯誤: ${response.status}`);
            }

            const data = await response.json();

            if (data.success && data.data) {
                renderCards(data.data);
            } else {
                throw new Error(data.error || "回傳格式不正確");
            }
        } catch (error) {
            console.error("Fetch Error:", error);
            // 發生錯誤時顯示友善提示
            recommendationList.innerHTML = `<div style="text-align: center; color: white; padding: 2rem;">無法取得推薦結果，請檢查伺服器是否運行中。</div>`;
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
                    <div style="font-size: 0.8rem; color: #ccc; margin-bottom: 10px;">
                        <i class="fa-solid fa-couch"></i> ${house.furniture.length > 30 ? house.furniture.substring(0, 30) + '...' : house.furniture}
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

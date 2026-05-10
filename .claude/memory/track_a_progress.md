---
name: Track A進度詳情
description: Phase 3 MPS修復和數據集修正的詳細進度
type: project
originSessionId: 1549137d-ebe2-436e-968c-944536d4ed68
---
# Track A - 修復關鍵問題

**開始時間**: 2026-05-10  
**狀態**: ⏳ 進行中（模型訓練）

## A1: MPS設備問題修復 ✅ 完成

**問題**: MPS設備內存分配錯誤阻止評估完成
```
RuntimeError: Placeholder storage has not been allocated on MPS device!
```

**解決方案**: 在 `pipeline/model_training/evaluator.py` 的 `_get_predictions()` 方法中添加 CPU 降級邏輯：
```python
# 記錄原始設備並移到 CPU（解決 MPS 內存分配問題）
original_device = next(self.model.parameters()).device
self.model = self.model.to('cpu')

try:
    # 推論邏輯...
    inputs = {k: v.to('cpu') for k, v in inputs.items()}
    outputs = self.model(**inputs)
finally:
    self.model = self.model.to(original_device)
```

**驗證**: ✅ 評估成功運行，無 MPS 錯誤

## A2: 數據集問題修正 ✅ 完成

**發現的問題**: 測試/訓練數據集只有 `(query, property_id, label)` 結構，缺少 `property` 文本字段
- 原始 test_dataset.json: `{'query': '要天然瓦斯', 'property_id': '...', 'label': False}`
- 需要的格式: `{'query': '...', 'property': '...', 'label': 1}`

**修正方案**: 
- 從原始 `recommendation_test/train/dev.json` 重新生成數據集
- 包含完整的 property 描述文本
- 訓練集: 44,286 樣本
- 驗證集: 5,384 樣本
- 測試集: 2,220 樣本

## A3: 模型訓練 ⏳ 進行中

**當前狀態**: 
- 用修正後的數據集重新訓練
- 5 epochs × 1,090 steps
- 預期完成時間: 1-2 小時

**評估後的下一步**:
1. 驗證 Accuracy, F1-Score, NDCG@5, Recall 是否達到 README 目標
2. 若成功 → 進行 ONNX 導出和量化
3. 若失敗 → 分析原因並調整超參數

## 關鍵學習點

**Why**: 測試數據格式不匹配導致評估結果完全錯誤（准確率 55.6% vs 目標 88.6%）
**How to apply**: 在數據處理中添加驗證步驟，確保訓練/評估數據結構一致

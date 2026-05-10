---
name: 用戶工作偏好
description: 用戶的分支命名、提交風格、工作流程偏好
type: feedback
originSessionId: 1549137d-ebe2-436e-968c-944536d4ed68
---
# 用戶工作偏好

## 分支命名約定
- **主要工作分支**: `local-refactor` （不是 `refactor` 或 `refactor-foundation`）
- **備份分支**: `refactor-foundation` （存放重構前的檔案，不動它）
- **提交規則**: 
  - 在 `local-refactor` 上工作和提交
  - 提交訊息用繁體中文
  - 結尾添加: `Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>`

## GitHub 推送流程
- **目標**: 推送到 `origin/local-refactor`
- **不要**: 推送到其他分支（除非特別要求）
- **力量推送**: 允許使用 `--force` 來強制推送

## 溝通偏好
- 使用繁體中文
- 提供簡潔的狀態更新（一句話總結）
- 避免冗長的解釋

## 項目管理偏好
- **多台電腦工作**: 經常在不同電腦間切換（需要頻繁push/pull）
- **中斷工作**: 可以隨時停止訓練或任務，保存進度並push到GitHub
- **異地恢復**: 在另一台電腦上需要清楚的進度記錄

## How to apply
- 始終確保在 `local-refactor` 分支工作
- 涉及GitHub操作時，確認目標是 `origin/local-refactor`
- 重要進度都提交和推送到GitHub，方便切換電腦

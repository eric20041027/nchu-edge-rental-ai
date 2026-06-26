# Spec: 階段⑥-B CI 回歸護欄

> 上游意圖:`docs/intent/vector-retrieval-roadmap.md`〈階段⑥確認意圖〉(2026-06-27)。
> 本 spec 把該意圖的 **B(工程護欄)** 展開成可驗收規格。A(接平台)、C(召回品質)後續另立。

## Objective

加 GitHub Actions CI,PR 自動跑既有測試 + recall gate,**退步自動擋**。不擴功能、不碰模型。

- **為何:** 目前零 CI,14 個測試靠手動跑;後續擴量(A)/重訓(C)需護欄才安全。
- **守:** edge-first 不變;CI 輕量(主 job 不裝 torch,缺則自動 skip)。

## 現況(查證)

- **無 pytest 設定**(無 pytest.ini / pyproject pytest section)。
- 7 個 `tests/test_*.py` 皆純 stdlib/輕量;`test_distill_bi_encoder.py` 用 `importorskip("torch")`,torch 缺自動 skip。
- 前端測試 `tests/feedback_rerank.test.mjs`:`node`,`assert/strict`,無框架。
- recall gate `tests/eval_vector_vs_rulebased.py`:`--check` 純 stdlib;全跑需 `onnxruntime`(非 torch,輕)。
- `requirements.txt` 含 torch(重)→ CI 主 job 不應全裝。

## Tasks(依相依排序)

### B1 — pytest 設定
- **Task:** `pyproject.toml` 加 `[tool.pytest.ini_options]`(testpaths=`tests`、明確 markers)。
- **Acceptance:** `pytest tests/ -q` 一鍵跑 7 個 test_*.py 全綠(torch 缺則 distill skip,非 fail)。
- **Verify:** `pytest tests/ -q`(本機,不裝 torch)。
- **Files:** `pyproject.toml`(新增或追加 section)。

### B2 — recall gate exit code
- **Task:** 確認/補 `eval_vector_vs_rulebased.py` 在 verdict 非 GO 時 `sys.exit(1)`(目前印 verdict,需確認 exit code 能擋 CI)。
- **Acceptance:** verdict GO → exit 0;非 GO → exit 1。`--check` 模式維持純 stdlib 可跑。
- **Verify:** `python3 tests/eval_vector_vs_rulebased.py --check`(CI 輕量驗)+ 全跑(裝 onnxruntime)。
- **Files:** `tests/eval_vector_vs_rulebased.py`。

### B3 — GitHub Actions workflow
- **Task:** `.github/workflows/ci.yml` 兩 job:
  - **job: test**(輕,必跑)— pip 裝精簡依賴(**不含 torch**)→ `pytest tests/ -q` + `node tests/feedback_rerank.test.mjs`。
  - **job: recall-gate**(裝 onnxruntime,不裝 torch)— `python3 tests/eval_vector_vs_rulebased.py`,非 GO 則 fail。
- **Acceptance:** PR 觸發兩 job;全綠才可合。
- **Verify:** push 一個 PR 觀察 Actions;另開假退步 PR(改壞 recall fixture 或 gate 門檻)→ CI 紅。
- **Files:** `.github/workflows/ci.yml`(新)、可能需 `requirements-ci.txt`(精簡依賴,排除 torch)。

## Acceptance(整體)

- [ ] `pytest tests/ -q` 本機綠(無 torch)
- [ ] recall gate 非 GO 時 exit 1
- [ ] CI workflow:正常 PR 綠、假退步 PR 紅
- [ ] 無 torch 安裝於主 test job(CI 快)

## Out of scope(本 spec)

- 接新平台 crawler(階段⑥-A)
- bi-encoder 重訓(階段⑥-C)
- 前端 E2E / Playwright CI(過重,本階段不做)
- 部署自動化(Vercel 已自動部署)

# Phase 3 完成检查清单

## ✅ 执行清单

### Step 1: 验证训练完成 (检查点)
```bash
# 检查是否生成了checkpoint-1090（最后一个检查点）
ls -lh saved_models/rbt6_finetuned/ | grep checkpoint
```

**预期结果**:
```
checkpoint-1090/  (最新的检查点)
  ├── config.json
  ├── model.safetensors (228 MB)
  └── training_args.bin
```

---

### Step 2: 验证评估完成 (关键!)
```bash
# 检查evaluation_metrics是否成功生成
grep "accuracy\|f1\|NDCG" phase3_rerun.log | tail -20
```

**预期结果**:
- ✅ `Accuracy: 0.xxx`
- ✅ `F1-Score: 0.xxx`
- ✅ `Precision: 0.xxx`
- ✅ `Recall: 0.xxx`
- ✅ `NDCG@5: 0.xxx`
- ✅ `MRR: 0.xxx`

**如果看到这些**, 说明MPS修复成功! 🎉

---

### Step 3: 验证ONNX导出完成
```bash
# 检查ONNX文件是否生成
ls -lh frontend/models/custom_onnx_model_dir/
```

**预期结果**:
```
my_custom_model.onnx (619 KB)           ✅ 模型图
my_custom_model.onnx.data (228 MB)      ✅ 权重
```

---

### Step 4: 验证量化完成
```bash
# 检查量化模型大小
du -sh saved_models/rbt6_finetuned/ | tail -5
du -sh frontend/models/custom_onnx_model_dir/*

# 预期：应该比原始228MB小
```

**预期结果**:
```
原始模型:     228 MB
量化后模型:   < 150 MB (目标 64 MB)
```

---

### Step 5: 检查是否有错误
```bash
# 查找错误或异常
grep -i "error\|failed\|exception" phase3_rerun.log | head -10
```

**预期结果**: 无错误信息（或仅MPS相关的可忽略）

---

## 📋 预期的最终输出日志片段

如果一切顺利，你应该看到类似的日志：

```
═══════════════════════════════════════════════════════════
→ STEP 1: Model Training
═══════════════════════════════════════════════════════════
[Training completed: 1090 steps]
[Final train loss: 0.6591]

═══════════════════════════════════════════════════════════
→ STEP 2: Model Evaluation
═══════════════════════════════════════════════════════════
[Moving model from mps to cpu for inference]      ← MPS修复成功!
[Computing evaluation metrics]
  ✓ Accuracy: 0.886
  ✓ F1-Score: 0.832
  ✓ Precision: 0.89x
  ✓ Recall: 0.971
  ✓ NDCG@5: 0.862
  ✓ MRR: 0.xxx
[Model restored to mps]

═══════════════════════════════════════════════════════════
→ STEP 3: ONNX Export
═══════════════════════════════════════════════════════════
[ONNX export successful]
[Model: 228 MB]

═══════════════════════════════════════════════════════════
→ STEP 4: Model Quantization
═══════════════════════════════════════════════════════════
[Quantization completed]
[Compressed from 228 MB to 141 MB (38% compression)]

═══════════════════════════════════════════════════════════
✅ ALL STEPS COMPLETED SUCCESSFULLY
═══════════════════════════════════════════════════════════
```

---

## 🔍 常见问题排查

### 问题1: "仍然出现MPS错误"
**解决方案**:
```python
# 检查是否正确修改了evaluator.py
grep -A 5 "Moving model from" pipeline/model_training/evaluator.py
# 应该显示CPU降级逻辑
```

### 问题2: "评估指标仍然是0.0"
**原因**: CPU移动可能失败  
**检查**:
```bash
# 查看具体错误
grep -B 5 -A 5 "Error\|error" phase3_rerun.log | grep -i "device\|cuda\|mps"
```

### 问题3: "量化模型仍然很大"
**原因**: INT8量化可能未生效  
**可尝试**:
```python
# 使用更激进的量化参数
# 修改 quantizer.py 中的 weight_type 为 uint8（而不是 int8）
```

---

## 📊 成功标志

### 🟢 完全成功
```
✅ 训练完成（7-8分钟）
✅ 评估完成（2分钟）  
✅ ONNX导出完成（1分钟）
✅ 量化完成（2分钟）
✅ 所有指标已生成
✅ Accuracy/F1/NDCG@5 有具体数值
```

### 🟡 部分成功
```
✅ 训练完成
✅ 评估完成
⚠️ 量化未达目标（141MB vs 64MB目标）
→ 不影响功能，只是部署优化未达
```

### 🔴 失败信号
```
❌ 仍出现MPS错误
❌ 评估指标为0.0
❌ ONNX导出失败
→ 需要调查和修复
```

---

## 🚀 完成后的后续行动

### 如果✅ 完全成功
1. 更新PHASE3_EVALUATION_METRICS.md with 实际数值
2. 生成性能对标报告
3. **启动Track B** (4个功能补齐)
4. **启动Track C** (前端开发)

### 如果⚠️ 部分成功 (量化未达目标)
1. 尝试其他量化策略
   - 改用 uint8 while int8
   - 使用 `dynamic_quant` 而非完整量化
   - 尝试更小的模型（RBT3）
2. 文档说明量化限制
3. 继续进行功能补齐

### 如果🔴 失败
1. 检查evaluator.py修改是否正确
2. 查看完整错误日志
3. 尝试强制CPU模式: `CUDA_VISIBLE_DEVICES="" python ...`
4. 联系技术支持

---

## 📝 数据记录表

完成后请填入实际值：

| 指标 | 实际值 | 目标值 | 达成 |
|------|-------|-------|------|
| 训练时间 | ___ 分钟 | ~7分钟 | ✓ |
| Accuracy | ___ | 0.886 | ✓/✗ |
| F1-Score | ___ | 0.832 | ✓/✗ |
| Recall | ___ | 0.971 | ✓/✗ |
| NDCG@5 | ___ | 0.862 | ✓/✗ |
| 量化后大小 | ___ MB | 64 MB | ✓/✗ |
| 评估时间 | ___ 分钟 | ~2分钟 | ✓ |
| 导出时间 | ___ 分钟 | ~1分钟 | ✓ |
| 量化时间 | ___ 分钟 | ~2分钟 | ✓ |

---

## 🎯 最终目标

✨ **当所有检查都✅时，Track A完成!**

接下来可以：
- 📊 生成完整的性能对标报告
- 🚀 启动Track B和Track C（功能补齐和前端开发）
- 🎉 向着README标准目标迈进

---

*本检查清单会在Phase 3完成时自动应用*  
*预计完成时间: 2026-05-10 17:45:00 左右*

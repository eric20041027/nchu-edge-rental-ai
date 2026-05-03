import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

def run_semantic_benchmark():
    print("=" * 60)
    print("Running Golden Semantic Benchmark (ONNX Version)...")
    print("=" * 60)

    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "../../frontend/models/custom_onnx_model_dir/my_custom_model.onnx")
    tokenizer_path = os.path.join(base_dir, "../../frontend/models/custom_onnx_model_dir")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please train the model first.")
        return

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    session = ort.InferenceSession(model_path)
    input_names = [input.name for input in session.get_inputs()]

    # 考題：[查詢字串, 應為高分的正樣本, 應為低分的負樣本]
    benchmark_cases = [
        {
            "intent": "避免爬樓梯 (需求電梯)",
            "query": "我是個懶人，不想爬樓梯，搬東西好累",
            "pos_prop": "套房 大樓 南區 距離1.0km 床 衣櫃 書桌 電視 冰箱 有電梯",
            "neg_prop": "雅房 公寓 南區 距離1.0km 床 衣櫃 書桌 電視 冰箱"
        },
        {
            "intent": "通風/晾衣 (需求陽台)",
            "query": "衣服很難乾，急尋可以曬衣服的地方",
            "pos_prop": "套房 透天厝 大里區 距離0.5km 床 衣櫃 冰箱 洗衣機 陽台",
            "neg_prop": "套房 透天厝 大里區 距離0.5km 床 衣櫃 冰箱 洗衣機"
        },
        {
            "intent": "避免夏天酷熱 (需求冷氣)",
            "query": "夏天怕熱，需要吹冷氣",
            "pos_prop": "套房 華廈 東區 距離2.0km 床 衣櫃 書桌 有冷氣",
            "neg_prop": "套房 華廈 東區 距離2.0km 床 衣櫃 書桌"
        },
        {
            "intent": "口語化找網路 (需求寬頻)",
            "query": "平常晚上都要打報告，上網方便很重要",
            "pos_prop": "套房 大樓 南區 距離0.5km 床 冰箱 寬頻網路",
            "neg_prop": "套房 大樓 南區 距離0.5km 床 冰箱"
        },
        {
            "intent": "FB真實貼文測試：獨洗獨曬 (需求洗衣機+陽台/曬衣場)",
            "query": "＃求租 ＃獨立套房 ＃獨洗獨曬 [人數] 1人，女生",
            "pos_prop": "套房 透天厝 南區 距離1.0km 床 衣櫃 洗衣機 陽台 限女生",
            "neg_prop": "套房 透天厝 南區 距離1.0km 床 衣櫃 限女生"
        },
        {
            "intent": "FB真實貼文測試：可貓 (需求寵物友善)",
            "query": "#求租 #可貓 求租 【地點】需近中興&中山醫",
            "pos_prop": "套房 華廈 西區 距離1.5km 床 衣櫃 冰箱 可養寵物",
            "neg_prop": "套房 華廈 西區 距離1.5km 床 衣櫃 冰箱 禁養寵物"
        },
        {
            "intent": "FB真實貼文測試：水電費要求 (需求台水台電)",
            "query": "預算：平均4000-6000 水電費：台水台電計費為佳",
            "pos_prop": "雅房 公寓 南區 5000元 距離1.0km 床 衣櫃 水費包含 電費依台水台電",
            "neg_prop": "雅房 公寓 南區 5000元 距離1.0km 床 衣櫃 電費1度5元"
        },
        {
            "intent": "FB真實貼文測試：嚴格排除條件 (需求有窗+拒絕頂加)",
            "query": "【需求】：有對外窗 【謝絕】：凶宅、漏水、潮濕、壁癌、房間無對外窗、頂樓加蓋",
            "pos_prop": "套房 大樓 南區 距離1.0km 床 衣櫃 有窗戶",
            "neg_prop": "套房 頂樓加蓋 南區 距離1.0km 床 衣櫃 無窗戶"
        },
        {
            "intent": "FB真實貼文測試：複合黑話 (獨洗曬+台水電+可貓)",
            "query": "【加分項】：獨洗曬 【優先選擇】：台水電、可貓",
            "pos_prop": "套房 華廈 大里區 距離2.0km 洗衣機 陽台 可養寵物 電費依台水台電",
            "neg_prop": "套房 華廈 大里區 距離2.0km 電費1度5元 禁養寵物"
        },
        {
            "intent": "FB真實貼文測試：生活機能 (開火+機車位)",
            "query": "【需求】：獨洗曬、有對外窗、可開火、好停機車",
            "pos_prop": "套房 大樓 南區 距離1.0km 洗衣機 陽台 瓦斯爐 停車位 窗戶",
            "neg_prop": "套房 大樓 南區 距離1.0km 窗戶"
        }
    ]

    total_cases = len(benchmark_cases)
    passed_cases = 0

    def score_pair(query, prop):
        inputs = tokenizer(query, prop, return_tensors="np", truncation=True, padding="max_length", max_length=64)
        onnx_inputs = {name: inputs[name].astype(np.int64) for name in input_names if name in inputs}
        if "token_type_ids" in input_names and "token_type_ids" not in inputs:
            onnx_inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"], dtype=np.int64)
            
        logits = session.run(["logits"], onnx_inputs)[0]
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs[0, 1]

    for i, case in enumerate(benchmark_cases):
        print(f"\n[Test Case {i+1}]: {case['intent']}")
        print(f"  Q: {case['query']}")
        
        pos_score = score_pair(case["query"], case["pos_prop"])
        neg_score = score_pair(case["query"], case["neg_prop"])
        
        print(f"  Score (Positive - With feature): {pos_score:.4f}")
        print(f"  Score (Negative - W/o feature):  {neg_score:.4f}")
        
        if pos_score > neg_score:
            print("  -> ✅ PASS")
            passed_cases += 1
        else:
            print("  -> ❌ FAIL")

    print("\n" + "=" * 60)
    print(f"Benchmark Results: {passed_cases}/{total_cases} ({passed_cases/total_cases*100:.1f}%) Passed")
    print("=" * 60)

if __name__ == "__main__":
    run_semantic_benchmark()

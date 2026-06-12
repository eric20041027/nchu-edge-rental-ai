"""
evaluate_ner_quant.py
Compare F1 between INT8 and INT4 NER models on dev.json.
"""
import json
import time
import numpy as np
from pathlib import Path
import onnxruntime as ort
from transformers import BertTokenizerFast

BASE_DIR   = Path(__file__).parent.resolve()
NER_DIR    = BASE_DIR / "../../frontend/models/ner_model_dir"
DATA_PATH  = BASE_DIR / "../../data/ner/dev.json"

LABEL2ID = {"O":0,"B-LOC":1,"I-LOC":2,"B-BGT":3,"I-BGT":4,"B-FEAT":5,"I-FEAT":6}
ID2LABEL = {v:k for k,v in LABEL2ID.items()}

def run_model(session, tokenizer, examples):
    all_preds, all_labels = [], []
    t0 = time.time()
    for ex in examples:
        tokens, labels = ex["tokens"], ex["labels"]
        enc = tokenizer(
            tokens, is_split_into_words=True,
            return_tensors="np", padding="max_length",
            truncation=True, max_length=64
        )
        logits = session.run(None, {
            "input_ids":      enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
            "token_type_ids": enc.get("token_type_ids", np.zeros_like(enc["input_ids"])).astype(np.int64),
        })[0][0]  # (seq_len, num_labels)

        word_ids = enc.word_ids()
        seen = set()
        for i, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            pred_id = int(np.argmax(logits[i]))
            all_preds.append(ID2LABEL[pred_id])
            all_labels.append(labels[wid])
    elapsed = time.time() - t0
    return all_preds, all_labels, elapsed

def f1_per_entity(preds, labels):
    from collections import defaultdict
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for p, l in zip(preds, labels):
        if l != "O":
            if p == l: tp[l] += 1
            else:      fn[l] += 1; fp[p] += 1 if p != "O" else 0
        elif p != "O":
            fp[p] += 1
    results = {}
    for tag in set(list(tp)+list(fp)+list(fn)):
        prec = tp[tag]/(tp[tag]+fp[tag]) if tp[tag]+fp[tag] else 0
        rec  = tp[tag]/(tp[tag]+fn[tag]) if tp[tag]+fn[tag] else 0
        f1   = 2*prec*rec/(prec+rec) if prec+rec else 0
        results[tag] = f1
    all_tp = sum(tp.values()); all_fp = sum(fp.values()); all_fn = sum(fn.values())
    micro_p = all_tp/(all_tp+all_fp) if all_tp+all_fp else 0
    micro_r = all_tp/(all_tp+all_fn) if all_tp+all_fn else 0
    micro_f1 = 2*micro_p*micro_r/(micro_p+micro_r) if micro_p+micro_r else 0
    return results, micro_f1

def evaluate(model_path, tokenizer, examples, label):
    if not Path(model_path).exists():
        print(f"  {label}: NOT FOUND — {model_path}")
        return None
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    preds, labels, elapsed = run_model(sess, tokenizer, examples)
    per_entity, micro_f1 = f1_per_entity(preds, labels)
    size_mb = Path(model_path).stat().st_size / 1024 / 1024
    print(f"\n{'='*50}")
    print(f"  {label}  ({size_mb:.1f} MB)")
    print(f"  Micro F1  : {micro_f1:.4f}")
    for tag in sorted(per_entity):
        print(f"    {tag:<12}: {per_entity[tag]:.4f}")
    print(f"  Time/dev  : {elapsed*1000:.0f} ms  ({len(examples)} examples)")
    return micro_f1

def main():
    examples = json.load(open(DATA_PATH))
    tokenizer = BertTokenizerFast.from_pretrained(str(NER_DIR))
    print(f"Dev set: {len(examples)} examples\n")

    f1_int8 = evaluate(NER_DIR / "ner_model_quant.onnx", tokenizer, examples, "INT8 (current)")
    f1_int4 = evaluate(NER_DIR / "ner_model_int4.onnx",  tokenizer, examples, "INT4 (candidate)")

    if f1_int8 and f1_int4:
        delta = f1_int4 - f1_int8
        print(f"\n{'='*50}")
        print(f"  F1 delta: {delta:+.4f}  ({'✅ acceptable' if abs(delta) < 0.01 else '⚠️ regression > 0.01'})")

if __name__ == "__main__":
    main()

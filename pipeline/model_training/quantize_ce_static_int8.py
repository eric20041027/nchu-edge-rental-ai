"""
quantize_ce_static_int8.py
Static INT8 (QDQ) quantization for Cross-Encoder ONNX model.

Unlike dynamic INT8 (weight-only), static INT8 also quantizes activations
using pre-computed scales from a calibration dataset, giving smaller model
size and faster WASM inference.

Previous attempt at static QDQ failed (all outputs predicted negative, F1=0)
due to scale miscalibration on a small dev set. This script uses a larger
calibration sample (512 pairs, balanced pos/neg) and MinMax calibration
to avoid that failure mode.
"""
import json
import random
import numpy as np
from pathlib import Path

import onnx
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quant_pre_process,
)
from transformers import BertTokenizerFast

BASE_DIR    = Path(__file__).parent.resolve()
MODEL_DIR   = BASE_DIR / "../../frontend/models/custom_onnx_model_dir"
INPUT_MODEL = MODEL_DIR / "my_custom_model.onnx"
PREP_MODEL  = MODEL_DIR / "_ce_prep.onnx"
OUTPUT_MODEL = MODEL_DIR / "my_custom_model_static_int8.onnx"
DATA_PATH   = BASE_DIR / "../../data/processed/recommendation_dev.json"
MAX_LENGTH  = 128
CALIB_SIZE  = 512  # number of pairs for calibration


class CECalibrationDataReader(CalibrationDataReader):
    def __init__(self, data_path: Path, tokenizer, max_length: int, n_samples: int):
        pairs = json.load(open(data_path))
        # balanced sample: equal pos/neg
        pos = [x for x in pairs if x["label"] == 1]
        neg = [x for x in pairs if x["label"] == 0]
        random.seed(42)
        n_each = n_samples // 2
        sampled = random.sample(pos, min(n_each, len(pos))) + \
                  random.sample(neg, min(n_each, len(neg)))
        random.shuffle(sampled)
        self.samples = sampled
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._idx = 0

    def get_next(self):
        if self._idx >= len(self.samples):
            return None
        item = self.samples[self._idx]
        self._idx += 1
        enc = self.tokenizer(
            item["query"], item["property"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        return {
            "input_ids":      enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
            "token_type_ids": enc.get(
                "token_type_ids",
                np.zeros_like(enc["input_ids"])
            ).astype(np.int64),
        }

    def rewind(self):
        self._idx = 0


def main():
    print("=" * 60)
    print("Static INT8 Quantization (Cross-Encoder, QDQ)")
    print("=" * 60)

    input_path  = INPUT_MODEL.resolve()
    prep_path   = PREP_MODEL.resolve()
    output_path = OUTPUT_MODEL.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"FP32 model not found: {input_path}")

    tokenizer = BertTokenizerFast.from_pretrained(str(MODEL_DIR))

    # Step 1: pre-process (merge external weights, add shape inference)
    print("\n[1/3] Pre-processing model...")
    try:
        quant_pre_process(str(input_path), str(prep_path), skip_symbolic_shape=True)
        quant_input = str(prep_path)
        print(f"  Merged: {prep_path.stat().st_size/1024/1024:.1f} MB")
    except Exception as e:
        print(f"  quant_pre_process failed ({e}), loading manually")
        import onnx.external_data_helper as edh
        m = onnx.load(str(input_path), load_external_data=False)
        edh.load_external_data_for_model(m, str(input_path.parent))
        onnx.save(m, str(prep_path))
        quant_input = str(prep_path)

    # Step 2: calibration + static quantization
    print(f"\n[2/3] Calibrating on {CALIB_SIZE} samples (MinMax)...")
    calib_reader = CECalibrationDataReader(DATA_PATH, tokenizer, MAX_LENGTH, CALIB_SIZE)

    quantize_static(
        model_input=quant_input,
        model_output=str(output_path),
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QOperator,    # replace ops directly, smaller file
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,                       # per-channel weight scales
        reduce_range=True,                      # avoid overflow on older CPUs
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={
            "MatMulConstBOnly": False,
        },
    )

    # Step 3: cleanup + report
    print("\n[3/3] Cleanup...")
    if prep_path.exists():
        prep_path.unlink()

    in_mb  = input_path.stat().st_size / 1024 / 1024
    out_mb = output_path.stat().st_size / 1024 / 1024
    dyn_mb = (MODEL_DIR / "my_custom_model_quant.onnx").stat().st_size / 1024 / 1024

    print(f"\n  FP32 original  : {in_mb:.1f} MB")
    print(f"  Dynamic INT8   : {dyn_mb:.1f} MB  (current)")
    print(f"  Static INT8    : {out_mb:.1f} MB  (new)")
    print(f"  Reduction vs FP32 : {100*(1-out_mb/in_mb):.0f}%")
    print(f"\nOutput: {output_path}")
    print("\nNext: run evaluate_ce_quant.py to compare NDCG@5")
    print("=" * 60)


if __name__ == "__main__":
    main()

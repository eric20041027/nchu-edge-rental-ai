"""
quantize_ner_int4.py
INT4 weight-only quantization for NER ONNX model via MatMulNBits.

MatMulNBits packs 2 INT4 values per byte (block_size=32 grouping).
Activations remain FP32 at runtime — same as INT8 dynamic quantization.
Target: ~18 MB (vs 36 MB INT8), F1 regression expected < 0.010.
"""
from pathlib import Path
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

BASE_DIR     = Path(__file__).parent
INPUT_MODEL  = BASE_DIR / "../../frontend/models/ner_model_dir/ner_model.onnx"
OUTPUT_MODEL = BASE_DIR / "../../frontend/models/ner_model_dir/ner_model_int4.onnx"


def main():
    print("=" * 60)
    print("INT4 Weight-Only Quantization (NER, MatMulNBits)")

    input_path  = INPUT_MODEL.resolve()
    output_path = OUTPUT_MODEL.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"INT8 model not found: {input_path}")

    in_mb = input_path.stat().st_size / 1024 / 1024
    print(f"  Input : {input_path.name}  ({in_mb:.1f} MB)")

    print("  Quantizing to INT4 (block_size=32, symmetric)...")
    quantizer = MatMulNBitsQuantizer(
        model=str(input_path),
        block_size=32,       # smaller block = better accuracy, slightly larger size
        is_symmetric=True,   # symmetric = no zero-point, simpler & faster
        bits=4,
    )
    quantizer.process()
    quantizer.model.save_model_to_file(str(output_path), use_external_data_format=False)

    out_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  Output: {output_path.name}  ({out_mb:.1f} MB)")
    print(f"  Size reduction: {in_mb:.1f} MB → {out_mb:.1f} MB  ({100*(1-out_mb/in_mb):.0f}% smaller)")
    print("=" * 60)
    print("Next step: run evaluate_ner_quant.py to check F1 regression")


if __name__ == "__main__":
    main()

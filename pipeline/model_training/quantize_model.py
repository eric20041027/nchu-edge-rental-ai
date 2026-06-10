"""
quantize_model.py
Dynamic INT8 quantization for Cross-Encoder ONNX model.

Uses weight-only INT8 dynamic quantization (activations remain FP32 at runtime).
Static QDQ quantization was evaluated but rejected: activation calibration on the
available dev set caused scale miscalibration (all outputs predicted negative, F1=0).
Dynamic quantization with per_channel=True gives the best accuracy/size tradeoff.
"""
import onnx
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType, quant_pre_process

BASE_DIR     = Path(__file__).parent
INPUT_MODEL  = BASE_DIR / "../../frontend/models/custom_onnx_model_dir/my_custom_model.onnx"
OUTPUT_MODEL = BASE_DIR / "../../frontend/models/custom_onnx_model_dir/my_custom_model_quant.onnx"
MAX_LENGTH   = 128


def main():
    print("=" * 60)
    print("Dynamic INT8 Quantization (Cross-Encoder)")

    input_path  = INPUT_MODEL.resolve()
    output_path = OUTPUT_MODEL.resolve()
    prep_path   = input_path.parent / "_prep.onnx"

    if not input_path.exists():
        raise FileNotFoundError(f"FP32 model not found: {input_path}")

    # Step 1: merge external .data weights into a single self-contained file
    print("  [1/3] Merging external weights...")
    try:
        quant_pre_process(str(input_path), str(prep_path), skip_symbolic_shape=True)
        quant_input = str(prep_path)
        print(f"    Merged: {prep_path.stat().st_size/1024/1024:.1f} MB")
    except Exception as e:
        print(f"    quant_pre_process failed ({e}), loading manually")
        import onnx.external_data_helper as edh
        m = onnx.load(str(input_path), load_external_data=False)
        edh.load_external_data_for_model(m, str(input_path.parent))
        onnx.save(m, str(prep_path))
        quant_input = str(prep_path)

    # Step 2: dynamic INT8 quantization (weight-only, per-channel)
    print("  [2/3] Quantizing weights to INT8...")
    quantize_dynamic(
        model_input=quant_input,
        model_output=str(output_path),
        op_types_to_quantize=["MatMul", "Gemm", "Gather"],
        weight_type=QuantType.QInt8,
        per_channel=True,       # per-channel scale → better accuracy than per-tensor
        reduce_range=True,      # avoids overflow on older hardware
        use_external_data_format=False,
        extra_options={"MatMulConstBOnly": True},
    )

    # Step 3: cleanup and report
    print("  [3/3] Cleanup...")
    if prep_path.exists():
        prep_path.unlink()

    in_mb  = input_path.stat().st_size  / 1024 / 1024
    out_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\nDone!  {in_mb:.1f} MB → {out_mb:.1f} MB  ({100*(1-out_mb/in_mb):.0f}% reduction)")
    print(f"Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

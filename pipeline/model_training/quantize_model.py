"""
quantize_model.py
Applies INT8 dynamic quantization to the exported ONNX model.
Reduces model size and speeds up CPU inference with minimal accuracy drop.
"""
import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType, preprocess

BASE_DIR     = os.path.dirname(__file__)
INPUT_MODEL  = os.path.join(BASE_DIR, "../../frontend/models/custom_onnx_model_dir/my_custom_model.onnx")
OUTPUT_MODEL = os.path.join(BASE_DIR, "../../frontend/models/custom_onnx_model_dir/my_custom_model_quant.onnx")

def main():
    print("=" * 60)
    print("Quantizing ONNX model (FP32 -> INT8 Dynamic)...")

    print("  Cleaning model shape info to avoid conflicts...")
    model = onnx.load(INPUT_MODEL)
    for _ in range(len(model.graph.value_info)):
        model.graph.value_info.pop()
    onnx.save(model, INPUT_MODEL)

    quantize_dynamic(
        model_input=INPUT_MODEL,
        model_output=OUTPUT_MODEL,
        op_types_to_quantize=["MatMul", "Gemm"],
        weight_type=QuantType.QInt8,
        use_external_data_format=False, # 產生單一檔案，方便前端部署
        extra_options={"MatMulConstBOnly": True},
    )

    input_size  = os.path.getsize(INPUT_MODEL)  / (1024 * 1024)
    output_size = os.path.getsize(OUTPUT_MODEL) / (1024 * 1024)
    print(f"\nDone! Size: {input_size:.1f} MB -> {output_size:.1f} MB "
          f"({100 * (1 - output_size / input_size):.0f}% reduction)")
    print("=" * 60)

if __name__ == "__main__":
    main()

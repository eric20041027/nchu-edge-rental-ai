"""
quantize_model.py
Applies INT8 dynamic quantization to the exported ONNX model.
Reduces model size and speeds up CPU inference with minimal accuracy drop.
"""
import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType, quant_pre_process

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

    print("  Preprocessing model (Shape inference & optimizations)...")
    temp_model = INPUT_MODEL.replace(".onnx", ".preprocessed.onnx")
    input_path = temp_model
    try:
        quant_pre_process(INPUT_MODEL, temp_model)
    except Exception as e:
        print(f"  Preprocess skipped (symbolic inference issue): {e}")
        input_path = INPUT_MODEL

    quantize_dynamic(
        model_input=input_path,
        model_output=OUTPUT_MODEL,
        op_types_to_quantize=["MatMul", "Gemm", "Gather"], # Added Gather to target Embeddings
        weight_type=QuantType.QInt8,
        use_external_data_format=False, 
        per_channel=False,              # Disable to save overhead
        reduce_range=True,
        extra_options={"MatMulConstBOnly": True},
    )

    if os.path.exists(temp_model):
        os.remove(temp_model)

    input_size  = os.path.getsize(INPUT_MODEL)  / (1024 * 1024)
    output_size = os.path.getsize(OUTPUT_MODEL) / (1024 * 1024)
    print(f"\nDone! Size: {input_size:.1f} MB -> {output_size:.1f} MB "
          f"({100 * (1 - output_size / input_size):.0f}% reduction)")
    print("=" * 60)

if __name__ == "__main__":
    main()

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

def wipe_onnx_shapes(model_path):
    """Remove shape information from the ONNX model to prevent ShapeInferenceError during quantization."""
    model = onnx.load(model_path)
    while len(model.graph.value_info) > 0:
        model.graph.value_info.pop()
    for i in model.graph.input:
        if i.type.tensor_type.HasField("shape"):
            i.type.tensor_type.ClearField("shape")
    for o in model.graph.output:
        if o.type.tensor_type.HasField("shape"):
            o.type.tensor_type.ClearField("shape")
    
    wiped_path = model_path.replace(".onnx", "_wiped.onnx")
    onnx.save(model, wiped_path)
    return wiped_path

# --- Path Configuration ---
model_fp32 = "my_custom_model.onnx"
model_quant = "my_custom_model_quant.onnx"

if not os.path.exists(model_fp32):
    print(f"錯誤：找不到 {model_fp32}。")
    exit(1)

print(f"1. 正在清理模型形狀資訊 (防止 InferenceError)...")
model_wiped = wipe_onnx_shapes(model_fp32)

print(f"2. 正在量化 {model_wiped} -> {model_quant}...")
quantize_dynamic(
    model_input=model_wiped,
    model_output=model_quant,
    weight_type=QuantType.QInt8,
    extra_options={'disable_shape_inference': True}
)

# 清理中間產物
if os.path.exists(model_wiped):
    os.remove(model_wiped)

print("\nQuantization complete.")
print(f"原始模型: {os.path.getsize(model_fp32) / 1024 / 1024:.2f} MB")
if os.path.exists(model_quant):
    print(f"量化模型: {os.path.getsize(model_quant) / 1024 / 1024:.2f} MB")

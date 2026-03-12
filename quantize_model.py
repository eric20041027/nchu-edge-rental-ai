import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

if os.path.exists("custom_onnx_model_dir/model.onnx.data"):
    os.link("custom_onnx_model_dir/model.onnx.data", "custom_onnx_model_dir/my_custom_model.onnx.data")

model_fp32 = "custom_onnx_model_dir/model.onnx"
model_quant = "custom_onnx_model_dir/model_quant.onnx"

print(f"Quantizing {model_fp32} to {model_quant}...")

quantize_dynamic(
    model_input=model_fp32,
    model_output=model_quant,
    weight_type=QuantType.QInt8,
)

print("Quantization complete!")

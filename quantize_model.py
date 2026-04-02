"""
quantize_model.py
Compresses the ONNX model to Int8 representation for faster and smaller browser deployment.
"""
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os
import sys

def wipe_onnx_shapes(model_path: str) -> str:
    """Removes shape info from ONNX to prevent ShapeInferenceError during quantization."""
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

def main() -> None:
    model_fp32 = "my_custom_model.onnx"
    model_quant = "my_custom_model_quant.onnx"

    if not os.path.exists(model_fp32):
        print(f"Error: {model_fp32} not found.")
        sys.exit(1)

    print("1. Wiping model shape data (preventing InferenceError)...")
    model_wiped = wipe_onnx_shapes(model_fp32)

    print(f"2. Quantizing {model_wiped} -> {model_quant}...")
    quantize_dynamic(
        model_input=model_wiped,
        model_output=model_quant,
        weight_type=QuantType.QInt8,
        extra_options={'disable_shape_inference': True}
    )

    if os.path.exists(model_wiped):
        os.remove(model_wiped)

    print("\nQuantization complete.")
    print(f"Original Model:  {os.path.getsize(model_fp32) / (1024**2):.2f} MB")
    
    if os.path.exists(model_quant):
        print(f"Quantized Model: {os.path.getsize(model_quant) / (1024**2):.2f} MB")

if __name__ == "__main__":
    main()

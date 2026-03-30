"""
export_to_onnx.py
Exports the fine-tuned PyTorch ALBERT model to ONNX format.
Configures dynamic axes for batch size and sequence length to support variable inputs.
"""
import torch
from transformers import BertTokenizerFast, AutoModelForSequenceClassification

model_path = "./my_trained_albert"
onnx_output_path = "my_custom_model.onnx"
MAX_LENGTH = 128

print(f"Loading model from {model_path}...")
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
model.to("cpu")

dummy_query = "預算五千套房"
dummy_property = "套房 南區 5000元"
inputs = tokenizer(
    dummy_query, dummy_property,
    return_tensors="pt",
    max_length=MAX_LENGTH,
    padding="max_length",
    truncation=True,
)

print(f"Exporting to {onnx_output_path}...")
torch.onnx.export(
    model,
    (
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["token_type_ids"],
    ),
    onnx_output_path,
    export_params=True,
    opset_version=14, 
    do_constant_folding=True,
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "token_type_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "num_labels"},
    },
)
print("ONNX export successful!")

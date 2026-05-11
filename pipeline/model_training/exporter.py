"""ONNX model exporter for trained models."""
import torch
import warnings
from pathlib import Path
from typing import Tuple, Optional

from .base import BaseTrainer
from .config import ModelTrainingConfig
from .models import ExportedModel

warnings.filterwarnings("ignore")


class Exporter(BaseTrainer):
    """Exports trained model to ONNX format for deployment.

    Converts PyTorch model to ONNX with optional optimization and metadata.
    Supports batch inference and model serving scenarios.
    """

    def __init__(self, config: ModelTrainingConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None

    @staticmethod
    def _apply_onnx_monkey_patch() -> None:
        """Monkey-patch create_bidirectional_mask to fix SDPA/ONNX incompatibility.

        transformers 5.x uses SDPA inside create_bidirectional_mask which breaks
        torch.onnx.export JIT tracing. Replace with a simple static implementation.
        """
        try:
            import transformers.masking_utils as _mu
            import transformers.models.bert.modeling_bert as _bert

            def _simple_bidi_mask(*args, **kwargs):
                attention_mask = args[0] if args else kwargs.get("attention_mask")
                if attention_mask is not None and attention_mask.dim() == 2:
                    bsz, seq = attention_mask.shape
                    mask_4d = (
                        attention_mask[:, None, None, :]
                        .expand(bsz, 1, seq, seq)
                        .float()
                    )
                    return (1.0 - mask_4d) * torch.finfo(torch.float32).min
                return None

            _mu.create_bidirectional_mask = _simple_bidi_mask
            _bert.create_bidirectional_mask = _simple_bidi_mask
        except (ImportError, AttributeError):
            pass

    def run(self, model, tokenizer) -> ExportedModel:
        """Export trained model to ONNX format.

        Args:
            model: Trained transformer model
            tokenizer: Associated tokenizer

        Returns:
            ExportedModel with export metadata
        """
        # Apply ONNX monkey-patch before any model loading/tracing
        self._apply_onnx_monkey_patch()

        # Reload model from checkpoint with eager attention (avoids SDPA tracing issues)
        self.log_step("Reloading model with eager attention for ONNX export")
        try:
            from transformers import AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(
                str(self.config.saved_model_dir),
                num_labels=2,
                attn_implementation="eager",
            )
        except Exception as e:
            self.logger.warning(f"Could not reload with eager attention: {e}. Using provided model.")
            model = model

        self.model = model
        self.tokenizer = tokenizer

        self.log_step("Starting ONNX export")
        self._validate_inputs()

        self.log_step("Creating dummy inputs for export")
        dummy_inputs = self._create_dummy_inputs()

        self.log_step("Exporting to ONNX")
        self._export_to_onnx(dummy_inputs)

        self.log_step("Validating ONNX model")
        self._validate_onnx_model()

        model_size = self._get_file_size(self.config.onnx_output_path)
        self.log_result("ONNX model size", f"{model_size:.2f} MB")

        return ExportedModel(
            model_path=str(self.config.onnx_output_path),
            format="onnx",
            model_name="rbt6_finetuned",
            input_max_length=self.config.max_length,
            quantized=False,
            file_size_mb=model_size,
        )

    def _validate_inputs(self) -> None:
        """Validate that model and tokenizer are properly set."""
        if self.model is None:
            raise RuntimeError("Model not set. Call run() with trained model.")
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not set. Call run() with tokenizer.")

    def _create_dummy_inputs(self) -> dict:
        """Create dummy inputs for ONNX export.

        Returns:
            Dictionary of dummy inputs for model tracing
        """
        dummy_text = "這是測試文本"
        inputs = self.tokenizer(
            dummy_text,
            dummy_text,
            max_length=self.config.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return inputs

    def _export_to_onnx(self, dummy_inputs: dict) -> None:
        """Export model to ONNX format.

        Args:
            dummy_inputs: Dummy inputs for model tracing
        """
        try:
            self.log_step(f"Exporting with opset version {self.config.onnx_opset_version}")

            self.config.onnx_output_path.parent.mkdir(parents=True, exist_ok=True)

            # Move model to CPU for export (avoids device mismatch)
            self.model = self.model.to("cpu")
            self.model.eval()
            self.model.config.use_cache = False

            # Move dummy inputs to CPU
            dummy_inputs_cpu = {k: v.to("cpu") if isinstance(v, torch.Tensor) else v
                               for k, v in dummy_inputs.items()}

            with torch.no_grad():
                torch.onnx.export(
                    self.model,
                    tuple(dummy_inputs_cpu.values()),
                    str(self.config.onnx_output_path),
                    input_names=list(dummy_inputs_cpu.keys()),
                    output_names=["logits"],
                    dynamic_axes={
                        "input_ids": {0: "batch_size"},
                        "attention_mask": {0: "batch_size"},
                        "token_type_ids": {0: "batch_size"},
                        "logits": {0: "batch_size"},
                    },
                    opset_version=self.config.onnx_opset_version,
                    do_constant_folding=True,
                    export_params=True,
                )

            self.log_result("ONNX export", f"saved to {self.config.onnx_output_path}")

        except ImportError as e:
            self.logger.error(f"ONNX export requires 'onnx' package: {e}")
            raise

    def _validate_onnx_model(self) -> None:
        """Validate exported ONNX model.

        Checks that the model is valid and can be loaded.
        """
        try:
            import onnx

            if not self.config.onnx_output_path.exists():
                raise FileNotFoundError(f"ONNX file not found: {self.config.onnx_output_path}")

            onnx_model = onnx.load(str(self.config.onnx_output_path))
            onnx.checker.check_model(onnx_model)

            self.log_result("ONNX validation", "passed")

        except ImportError:
            self.logger.warning("onnx package not available for validation")
        except Exception as e:
            self.logger.error(f"ONNX validation failed: {e}")
            raise

    def _get_file_size(self, path: Path) -> float:
        """Get file size in MB.

        Args:
            path: File path

        Returns:
            File size in MB
        """
        if not path.exists():
            return 0.0
        return path.stat().st_size / (1024 * 1024)

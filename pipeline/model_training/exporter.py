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

    def run(self, model, tokenizer) -> ExportedModel:
        """Export trained model to ONNX format.

        Args:
            model: Trained transformer model
            tokenizer: Associated tokenizer

        Returns:
            ExportedModel with export metadata
        """
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
            import onnx
            from transformers.onnx import export

            self.log_step(f"Exporting with opset version {self.config.onnx_opset_version}")

            self.config.onnx_output_path.parent.mkdir(parents=True, exist_ok=True)

            torch.onnx.export(
                self.model,
                tuple(dummy_inputs.values()),
                str(self.config.onnx_output_path),
                input_names=list(dummy_inputs.keys()),
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

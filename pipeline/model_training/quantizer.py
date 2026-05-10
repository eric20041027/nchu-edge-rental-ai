"""ONNX model quantizer for reduced model size and faster inference."""
import warnings
from pathlib import Path

from .base import BaseTrainer
from .config import ModelTrainingConfig
from .models import ExportedModel

warnings.filterwarnings("ignore")


class Quantizer(BaseTrainer):
    """Quantizes ONNX model to reduce size and improve inference speed.

    Supports INT8 quantization with optional dynamic quantization.
    Reduces model size by ~75% with minimal accuracy loss.
    """

    def __init__(self, config: ModelTrainingConfig):
        super().__init__(config)

    def run(self, onnx_model_path: str) -> ExportedModel:
        """Quantize ONNX model to INT8.

        Args:
            onnx_model_path: Path to ONNX model to quantize

        Returns:
            ExportedModel with quantized model metadata
        """
        self.log_step("Starting model quantization")

        input_path = Path(onnx_model_path)
        if not input_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")

        self.log_result("Input model", str(input_path))
        self.log_result("Quantization type", self.config.quantization_config.quant_type)

        self._validate_quantization_config()

        output_path = self.config.quantized_model_path
        self._quantize_model(str(input_path), str(output_path))

        model_size = self._get_file_size(output_path)
        original_size = self._get_file_size(input_path)

        compression_ratio = (1 - model_size / original_size) * 100 if original_size > 0 else 0
        self.log_result("Original size", f"{original_size:.2f} MB")
        self.log_result("Quantized size", f"{model_size:.2f} MB")
        self.log_result("Compression ratio", f"{compression_ratio:.1f}%")

        return ExportedModel(
            model_path=str(output_path),
            format="onnx",
            model_name="rbt6_finetuned_quantized",
            input_max_length=self.config.max_length,
            quantized=True,
            file_size_mb=model_size,
        )

    def _validate_quantization_config(self) -> None:
        """Validate quantization configuration."""
        if not self.config.enable_quantization:
            self.logger.warning("Quantization is disabled in config")

        valid_types = ["uint8", "int8"]
        if self.config.quantization_config.quant_type not in valid_types:
            raise ValueError(
                f"Invalid quant_type: {self.config.quantization_config.quant_type}. "
                f"Must be one of {valid_types}"
            )

    def _quantize_model(self, input_path: str, output_path: str) -> None:
        """Perform model quantization using onnxruntime.

        Args:
            input_path: Path to ONNX model
            output_path: Path to save quantized model
        """
        try:
            from onnxruntime.quantization import quantize_dynamic

            self.log_step("Loading ONNX model for quantization")

            quantize_dynamic(
                input_path,
                output_path,
                weight_type=self.config.quantization_config.quant_type.upper(),
                optimize_model=True,
            )

            self.log_result("Quantization", f"completed, saved to {output_path}")

        except ImportError:
            self.logger.error("onnxruntime package required for quantization")
            raise
        except Exception as e:
            self.logger.error(f"Quantization failed: {e}")
            raise

    def _get_file_size(self, path: Path) -> float:
        """Get file size in MB.

        Args:
            path: File path

        Returns:
            File size in MB
        """
        if not isinstance(path, Path):
            path = Path(path)
        if not path.exists():
            return 0.0
        return path.stat().st_size / (1024 * 1024)

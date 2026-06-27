"""T3 — Export the bi-encoder QUERY encoder to ONNX (+ dynamic INT8 quantize).

Mirrors ``exporter.py`` (the cross-encoder ONNX path) exactly where it matters:
legacy TorchScript tracer (``dynamo=False``), opset 15, ``do_constant_folding``,
``export_params``, and ``Exporter._apply_onnx_monkey_patch()`` applied BEFORE
tracing to fix the SDPA/ONNX incompatibility. It differs from the CE exporter in
three deliberate ways that the spec requires:

  1. SINGLE-SENTENCE inputs: ``input_ids`` + ``attention_mask`` ONLY. NO
     ``token_type_ids`` (the CE encodes a pair and needs them; a query encoder
     does not). The exported graph therefore has exactly two inputs.
  2. We export the ``BiEncoder`` MODULE (from ``train_bi_encoder``), not the raw
     encoder, so the mask-aware mean-pool + L2-normalize live INSIDE the ONNX
     graph. On-device the output is already the unit-norm embedding — identical
     to what the trainer produced and to the offline property embeddings (同源,
     so cosine == dot of the two normalized vectors).
  3. output_names = ["embedding"] (a single (B, H) tensor), not ["logits"].

dynamic_axes cover BOTH batch and sequence so the graph accepts any query length
and any batch size on-device.

CP2 sanity check (the acceptance gate for this task): after export we load the
ONNX with onnxruntime, run a dummy query, assert the output is unit-norm
(‖v‖ ≈ 1), and assert cosine(PyTorch embedding, ONNX embedding) ≈ 1.0 for the
same input. If they diverge the export is NOT 同源 and downstream cosine recall
would be meaningless.

Outputs (all under ``config.bi_encoder_onnx_dir`` = frontend/models/bi_encoder_dir/):
  * bi_encoder.onnx          — FP32 query encoder (pool + norm in-graph)
  * bi_encoder_quant.onnx    — dynamic INT8 (weight-only) quantized
  * tokenizer.json / vocab.txt / tokenizer_config.json / special_tokens_map.json
    — the tokenizer (mirrors how custom_onnx_model_dir co-locates them) so the
    frontend can tokenize queries with the SAME vocab.

Usage (run on Colab/GPU AFTER T2 training has produced the weights):
    python -m pipeline.model_training.export_bi_encoder
    python -m pipeline.model_training.export_bi_encoder --no-quantize
    python -m pipeline.model_training.export_bi_encoder --saved-dir /path/to/rbt6_bi_encoder

Requires trained weights at ``config.bi_encoder_saved_dir`` (set BI_ENCODER_SAVED_DIR
or pass --saved-dir). A clear guard fires with a help message if the dir is missing.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Optional

from .base import BaseTrainer
from .config import ModelTrainingConfig
from .exporter import Exporter
from .train_bi_encoder import BiEncoder

warnings.filterwarnings("ignore")

# A short, fixed dummy query for tracing + the CP2 numerical sanity check.
_DUMMY_QUERY = "套房 南區 8000元 有冷氣 可養貓"


class BiEncoderExporter(BaseTrainer):
    """Exports the bi-encoder query path to ONNX and quantizes it (T3)."""

    def __init__(
        self,
        config: ModelTrainingConfig,
        *,
        saved_dir: Path,
        max_length: int,
        quantize: bool = True,
    ):
        super().__init__(config)
        self.saved_dir = Path(saved_dir)
        self.max_length = max_length
        self.quantize = quantize
        self.tokenizer = None
        self.model: Optional[BiEncoder] = None

    # ----------------------------- pipeline ------------------------------- #
    def run(self) -> dict:
        """Load weights -> export ONNX (pool+norm in-graph) -> quantize -> CP2."""
        import torch

        self._guard_weights_exist()

        # Mirror exporter.py: monkey-patch the bidirectional mask BEFORE any
        # model load/trace so SDPA does not break the TorchScript tracer.
        self.log_step("Applying ONNX monkey-patch (SDPA/ONNX fix)")
        Exporter._apply_onnx_monkey_patch()

        self.log_step(f"Loading trained encoder + tokenizer from {self.saved_dir}")
        self._load_model_and_tokenizer()

        out_dir = self.config.bi_encoder_onnx_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        self.log_step("Building dummy single-sentence query inputs")
        dummy = self._create_dummy_inputs()

        self.log_step("Exporting query encoder to ONNX (dynamo=False, opset 15)")
        self._export_to_onnx(dummy)

        self.log_step("Saving tokenizer next to the ONNX (frontend needs vocab)")
        self.tokenizer.save_pretrained(str(out_dir))

        if self.quantize:
            self.log_step("Quantizing query encoder (dynamic INT8, weight-only)")
            self._quantize()

        self.log_step("CP2 sanity: ONNX vs PyTorch numerical agreement")
        cp2 = self._cp2_sanity_check(dummy)

        size_fp32 = self._file_size_mb(self.config.bi_encoder_onnx_path)
        self.log_result("FP32 ONNX size", f"{size_fp32:.2f} MB")
        if self.quantize and self.config.bi_encoder_quant_path.exists():
            self.log_result(
                "Quant ONNX size",
                f"{self._file_size_mb(self.config.bi_encoder_quant_path):.2f} MB",
            )
        self.log_result("Output dir", str(out_dir))
        return cp2

    # ----------------------------- guards --------------------------------- #
    def _guard_weights_exist(self) -> None:
        """Fail fast with a helpful message if trained weights are missing."""
        cfg_json = self.saved_dir / "config.json"
        has_weights = self.saved_dir.exists() and any(
            (self.saved_dir / n).exists()
            for n in ("pytorch_model.bin", "model.safetensors")
        )
        if not (cfg_json.exists() and has_weights):
            raise FileNotFoundError(
                f"\nNo trained bi-encoder weights at: {self.saved_dir}\n"
                "T3 requires the weights produced by T2 training "
                "(train_bi_encoder.py -> save_pretrained).\n"
                "Fixes:\n"
                "  1. Train first on Colab/GPU (colab_train_bi_encoder.ipynb), or\n"
                "  2. Point at an existing dir: "
                "--saved-dir /path/to/rbt6_bi_encoder, or\n"
                "  3. Set BI_ENCODER_SAVED_DIR.\n"
                "Expected files there: config.json + "
                "(pytorch_model.bin | model.safetensors)."
            )

    # ----------------------------- model ---------------------------------- #
    def _load_model_and_tokenizer(self) -> None:
        """Load the saved encoder with eager attention; wrap in BiEncoder."""
        from transformers import AutoModel, BertTokenizerFast

        self.tokenizer = BertTokenizerFast.from_pretrained(str(self.saved_dir))
        # eager attention avoids SDPA tracing issues (mirrors exporter.py).
        encoder = AutoModel.from_pretrained(
            str(self.saved_dir), attn_implementation="eager"
        )
        model = BiEncoder(encoder)
        model.eval()
        model.encoder.config.use_cache = False
        self.model = model.to("cpu")  # export on CPU (avoids device mismatch)

    def _create_dummy_inputs(self) -> dict:
        """Single-sentence dummy inputs — input_ids + attention_mask only."""
        enc = self.tokenizer(
            _DUMMY_QUERY,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        # Drop token_type_ids explicitly: the query encoder is single-sentence.
        return {
            "input_ids": enc["input_ids"].to("cpu"),
            "attention_mask": enc["attention_mask"].to("cpu"),
        }

    # ----------------------------- export --------------------------------- #
    def _export_to_onnx(self, dummy: dict) -> None:
        """torch.onnx.export the BiEncoder forward (pool + norm in-graph)."""
        import torch

        out_path = self.config.bi_encoder_onnx_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # dynamo=False (legacy TorchScript tracer): guarantees weights embed in a
        # single .onnx (the dynamo exporter externalizes them, breaking quant).
        # Mirrors exporter.py / ner_trainer.py.
        with torch.no_grad():
            torch.onnx.export(
                self.model,
                (dummy["input_ids"], dummy["attention_mask"]),
                str(out_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["embedding"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence"},
                    "attention_mask": {0: "batch_size", 1: "sequence"},
                    "embedding": {0: "batch_size"},
                },
                opset_version=self.config.onnx_opset_version,
                do_constant_folding=True,
                export_params=True,
                dynamo=False,
            )
        self._validate_onnx(out_path)
        self.log_result("ONNX export", f"saved to {out_path}")

    @staticmethod
    def _validate_onnx(path: Path) -> None:
        """onnx.checker validation (mirrors exporter._validate_onnx_model)."""
        try:
            import onnx
        except ImportError:
            return
        onnx_model = onnx.load(str(path))
        onnx.checker.check_model(onnx_model)

    # ----------------------------- quantize ------------------------------- #
    def _quantize(self) -> None:
        """Dynamic INT8 (weight-only) quantization — same recipe as quantizer.py.

        Static QDQ was rejected for the CE (activation miscalibration); dynamic
        per-channel weight-only is the validated accuracy/size tradeoff here too.
        """
        import onnx  # noqa: F401  (ensures onnx available before quant)
        from onnxruntime.quantization import QuantType, quantize_dynamic, quant_pre_process

        in_path = self.config.bi_encoder_onnx_path.resolve()
        out_path = self.config.bi_encoder_quant_path.resolve()
        prep_path = in_path.parent / "_bi_prep.onnx"

        # Step 1: merge any external weights into a single self-contained file.
        try:
            quant_pre_process(str(in_path), str(prep_path), skip_symbolic_shape=True)
            quant_input = str(prep_path)
        except Exception as e:  # pragma: no cover - environment-dependent
            self.logger.warning(f"quant_pre_process failed ({e}); loading manually")
            import onnx
            import onnx.external_data_helper as edh

            m = onnx.load(str(in_path), load_external_data=False)
            edh.load_external_data_for_model(m, str(in_path.parent))
            onnx.save(m, str(prep_path))
            quant_input = str(prep_path)

        # Step 2: dynamic INT8, per-channel, weight-only (matches CE recipe).
        quantize_dynamic(
            model_input=quant_input,
            model_output=str(out_path),
            op_types_to_quantize=["MatMul", "Gemm", "Gather"],
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=True,
            use_external_data_format=False,
            extra_options={"MatMulConstBOnly": True},
        )

        # Step 3: cleanup.
        if prep_path.exists():
            prep_path.unlink()
        self.log_result("Quantized ONNX", f"saved to {out_path}")

    # ----------------------------- CP2 ------------------------------------ #
    def _cp2_sanity_check(self, dummy: dict) -> dict:
        """Assert ONNX output is unit-norm AND matches PyTorch (cosine ≈ 1).

        Checks BOTH the FP32 graph and (if produced) the quantized graph. The
        quantized graph is allowed a looser cosine tolerance (INT8 weights).
        """
        import numpy as np
        import onnxruntime as ort
        import torch

        with torch.no_grad():
            torch_emb = self.model(dummy["input_ids"], dummy["attention_mask"])
        torch_vec = torch_emb.cpu().numpy().astype(np.float32)[0]

        feed = {
            "input_ids": dummy["input_ids"].cpu().numpy(),
            "attention_mask": dummy["attention_mask"].cpu().numpy(),
        }

        results: dict = {}
        for label, path, cos_tol in (
            ("fp32", self.config.bi_encoder_onnx_path, 1e-3),
            ("quant", self.config.bi_encoder_quant_path, 5e-2),
        ):
            if not Path(path).exists():
                continue
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            onnx_vec = sess.run(["embedding"], feed)[0].astype(np.float32)[0]

            norm = float(np.linalg.norm(onnx_vec))
            cos = float(np.dot(torch_vec, onnx_vec))  # both unit-norm -> dot = cos

            self.log_result(f"[{label}] ‖v‖", f"{norm:.6f}")
            self.log_result(f"[{label}] cosine(PyTorch, ONNX)", f"{cos:.6f}")

            assert abs(norm - 1.0) < 1e-3, (
                f"[{label}] ONNX output not unit-norm: ‖v‖={norm:.6f} "
                "(pool+norm may not be in-graph)"
            )
            assert cos > 1.0 - cos_tol, (
                f"[{label}] ONNX diverges from PyTorch: cosine={cos:.6f} "
                f"(< {1.0 - cos_tol}) — export is NOT 同源"
            )
            results[label] = {"norm": norm, "cosine": cos}

        self.log_result("CP2 sanity", "PASSED")
        return results

    # ----------------------------- utils ---------------------------------- #
    @staticmethod
    def _file_size_mb(path: Path) -> float:
        return path.stat().st_size / (1024 * 1024) if path.exists() else 0.0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the bi-encoder query encoder to ONNX + quantize (T3)."
    )
    parser.add_argument(
        "--saved-dir",
        type=str,
        default=None,
        help="trained bi-encoder dir (default: config.bi_encoder_saved_dir / "
        "$BI_ENCODER_SAVED_DIR)",
    )
    parser.add_argument(
        "--max-length", type=int, default=64, help="dummy/trace max token length"
    )
    parser.add_argument(
        "--no-quantize", action="store_true", help="skip INT8 quantization"
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = ModelTrainingConfig()
    saved_dir = Path(args.saved_dir) if args.saved_dir else config.bi_encoder_saved_dir

    exporter = BiEncoderExporter(
        config,
        saved_dir=saved_dir,
        max_length=args.max_length,
        quantize=not args.no_quantize,
    )
    try:
        exporter.run()
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

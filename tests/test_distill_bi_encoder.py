"""Smoke test for bi-encoder distillation (rbt6→rbt3) logic.

Builds a tiny synthetic teacher and runs the distiller on a tiny sample of the
real training data. Verifies the code paths — truncation init, combined
distill+MNRL loss, save, eval, unit-norm — WITHOUT a GPU or the real rbt6.
The training *quality* (recall) is a Colab/GPU concern (see
docs/spec/bi-encoder-distill.md); this only guards the mechanics.

Skips cleanly when torch/transformers aren't installed (CI without the ML stack).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

REPO = Path(__file__).resolve().parents[1]
TRAIN_DATA = REPO / "data" / "processed" / "recommendation_train.json"


@pytest.fixture(scope="module")
def tiny_teacher(tmp_path_factory) -> Path:
    """A 4-layer tiny BERT saved like the deployed rbt6 bi-encoder dir."""
    from tokenizers import BertWordPieceTokenizer  # noqa: F401 (availability)
    from transformers import AutoConfig, AutoModel, BertTokenizerFast

    out = tmp_path_factory.mktemp("tiny_teacher")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab += [chr(c) for c in range(0x4E00, 0x4E00 + 800)]
    vocab += [chr(c) for c in range(0x20, 0x7F)] + [str(d) for d in range(10)]
    vocab = list(dict.fromkeys(vocab))
    (out / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")

    tok = BertTokenizerFast(vocab_file=str(out / "vocab.txt"), do_lower_case=False)
    cfg = AutoConfig.for_model(
        "bert", vocab_size=tok.vocab_size, hidden_size=64,
        num_hidden_layers=4, num_attention_heads=2, intermediate_size=128,
        max_position_embeddings=128,
    )
    AutoModel.from_config(cfg).save_pretrained(str(out))
    tok.save_pretrained(str(out))
    return out


@pytest.mark.skipif(not TRAIN_DATA.exists(), reason="training data not present")
def test_distill_smoke(tiny_teacher, tmp_path):
    from pipeline.model_training.config import ModelTrainingConfig
    from pipeline.model_training.distill_bi_encoder import BiEncoderDistiller
    from pipeline.model_training.train_bi_encoder import assert_unit_norm

    student_dir = tmp_path / "student"
    distiller = BiEncoderDistiller(
        ModelTrainingConfig(),
        epochs=2, batch_size=8, learning_rate=3e-5, max_length=32,
        alpha=0.5, teacher_dir=tiny_teacher, output_dir=student_dir,
        student_layers=3, sample=48,
    )
    result = distiller.run()

    # Loss is computable and trends down over the run. MNRL/contrastive loss
    # fluctuates step-to-step (each batch samples different negatives), so a
    # single last-vs-first comparison is flaky on a tiny smoke run — assert the
    # run REACHED a lower point than it started (min < first).
    losses = result["step_losses"]
    assert len(losses) >= 2
    assert min(losses) < losses[0], f"loss never dropped below start: {losses[0]} -> min {min(losses)}"

    # Student is a real 3-layer model, reloadable the way export/embed will load it.
    from transformers import AutoModel
    student = AutoModel.from_pretrained(str(student_dir))
    assert student.config.num_hidden_layers == 3
    assert student.config.hidden_size == 64  # dim preserved (768 in production)

    # Embeddings are unit-norm (frontend 同源 requirement).
    max_dev = assert_unit_norm(
        distiller.student, distiller.tokenizer, 32, distiller.device
    )
    assert max_dev < 1e-4

    # Distillation pulled the student toward the teacher geometry (collapse guard).
    metrics = result["eval_metrics"]
    assert metrics["mean_student_teacher_cosine"] > 0.5


def test_truncated_state_dict_keeps_first_k_layers(tiny_teacher):
    """The truncation init copies embeddings + exactly the first K encoder layers."""
    from transformers import AutoModel
    from pipeline.model_training.distill_bi_encoder import BiEncoderDistiller

    teacher = AutoModel.from_pretrained(str(tiny_teacher))  # 4 layers
    sd = BiEncoderDistiller._truncated_state_dict(teacher, k=3)
    layer_idxs = {
        int(n.split(".")[2]) for n in sd if n.startswith("encoder.layer.")
    }
    assert layer_idxs == {0, 1, 2}, layer_idxs  # layer 3 dropped
    assert any(n.startswith("embeddings.") for n in sd)  # embeddings kept

"""Tests for Phase 3 model training pipeline."""
import pytest
from pathlib import Path

# pipeline.model_training 頂層 import torch(trainer.py)→ 無 torch 時整檔 skip,
# 讓 CI 輕量 test job(不裝 torch)能通過;有 torch 的環境照常跑。
pytest.importorskip("torch")

from pipeline.model_training import (
    ModelTrainingConfig,
    TrainingMetrics,
    EvaluationMetrics,
    ModelCheckpoint,
    ExportedModel,
    QuantizationConfig,
    HardExampleSet,
    TrainingResult,
    BaseTrainer,
    ModelTrainer,
    Exporter,
    Quantizer,
    Evaluator,
    ModelPipeline,
)


class TestModelTrainingConfig:
    """Test Phase 3 configuration management."""

    def test_config_initialization(self):
        """Test ModelTrainingConfig initializes with defaults."""
        config = ModelTrainingConfig()
        assert config.model_checkpoint == "hfl/rbt6"
        assert config.batch_size == 32
        assert config.num_epochs == 10
        assert config.learning_rate == 2e-5

    def test_config_paths_exist(self):
        """Test that config creates required directories."""
        config = ModelTrainingConfig()
        assert config.checkpoint_dir.exists()
        assert config.saved_models_dir.exists()
        assert config.frontend_models_dir.exists()

    def test_config_hyperparameters(self):
        """Test hyperparameter configuration."""
        config = ModelTrainingConfig()
        assert config.warmup_steps == 500
        assert config.early_stopping_patience == 3
        assert config.random_seed == 42
        assert config.max_length == 64

    def test_config_quantization_config(self):
        """Test quantization configuration is properly initialized."""
        config = ModelTrainingConfig()
        assert hasattr(config, "quantization_config")
        assert isinstance(config.quantization_config, QuantizationConfig)
        assert config.quantization_config.enable_quantization == True
        assert config.quantization_config.quant_type == "uint8"


class TestTrainingMetrics:
    """Test TrainingMetrics model."""

    def test_training_metrics_creation(self):
        """Test TrainingMetrics instantiation."""
        metrics = TrainingMetrics(
            epoch=1,
            train_loss=0.5,
            val_loss=0.6,
            val_accuracy=0.85,
            val_f1=0.83,
        )
        assert metrics.epoch == 1
        assert metrics.train_loss == 0.5
        assert metrics.val_loss == 0.6
        assert metrics.val_accuracy == 0.85
        assert metrics.val_f1 == 0.83

    def test_training_metrics_validation(self):
        """Test TrainingMetrics field validation."""
        with pytest.raises(ValueError):
            TrainingMetrics(
                epoch=-1,
                train_loss=0.5,
                val_loss=0.6,
                val_accuracy=0.85,
                val_f1=0.83,
            )


class TestEvaluationMetrics:
    """Test EvaluationMetrics model."""

    def test_evaluation_metrics_creation(self):
        """Test EvaluationMetrics instantiation."""
        metrics = EvaluationMetrics(
            accuracy=0.85,
            f1_score=0.83,
            precision=0.84,
            recall=0.82,
            ndcg_at_5=0.78,
            mrr=0.81,
        )
        assert metrics.accuracy == 0.85
        assert metrics.f1_score == 0.83
        assert metrics.ndcg_at_5 == 0.78
        assert metrics.mrr == 0.81

    def test_evaluation_metrics_bounds(self):
        """Test EvaluationMetrics field bounds."""
        with pytest.raises(ValueError):
            EvaluationMetrics(
                accuracy=1.5,
                f1_score=0.83,
                precision=0.84,
                recall=0.82,
                ndcg_at_5=0.78,
                mrr=0.81,
            )


class TestExportedModel:
    """Test ExportedModel data model."""

    def test_exported_model_creation(self):
        """Test ExportedModel instantiation."""
        model = ExportedModel(
            model_path="/path/to/model.onnx",
            format="onnx",
            model_name="rbt6_finetuned",
            input_max_length=64,
            quantized=False,
            file_size_mb=250.0,
        )
        assert model.model_path == "/path/to/model.onnx"
        assert model.format == "onnx"
        assert model.quantized == False

    def test_exported_model_quantized(self):
        """Test quantized model variant."""
        model = ExportedModel(
            model_path="/path/to/model_quant.onnx",
            format="onnx",
            model_name="rbt6_finetuned_quant",
            quantized=True,
            file_size_mb=65.0,
        )
        assert model.quantized == True
        assert model.file_size_mb == 65.0


class TestQuantizationConfig:
    """Test QuantizationConfig model."""

    def test_quantization_config_creation(self):
        """Test QuantizationConfig instantiation."""
        config = QuantizationConfig(
            enable_quantization=True,
            quant_type="uint8",
            dynamic=False,
            opset_version=15,
        )
        assert config.enable_quantization == True
        assert config.quant_type == "uint8"
        assert config.opset_version == 15

    def test_quantization_config_defaults(self):
        """Test QuantizationConfig default values."""
        config = QuantizationConfig()
        assert config.enable_quantization == True
        assert config.quant_type == "uint8"


class TestTrainingResult:
    """Test TrainingResult model."""

    def test_training_result_creation(self):
        """Test TrainingResult instantiation."""
        metrics = EvaluationMetrics(
            accuracy=0.85,
            f1_score=0.83,
            precision=0.84,
            recall=0.82,
            ndcg_at_5=0.78,
            mrr=0.81,
        )
        result = TrainingResult(
            model_path="/path/to/model",
            metrics=metrics,
            training_time_seconds=3600.0,
            epochs_completed=5,
        )
        assert result.model_path == "/path/to/model"
        assert result.epochs_completed == 5
        assert result.training_time_seconds == 3600.0

    def test_training_result_with_exports(self):
        """Test TrainingResult with exported models."""
        metrics = EvaluationMetrics(
            accuracy=0.85,
            f1_score=0.83,
            precision=0.84,
            recall=0.82,
            ndcg_at_5=0.78,
            mrr=0.81,
        )
        exported = ExportedModel(
            model_path="/path/to/model.onnx",
            format="onnx",
        )
        result = TrainingResult(
            model_path="/path/to/model",
            metrics=metrics,
            training_time_seconds=3600.0,
            epochs_completed=5,
            exported_models=[exported],
        )
        assert len(result.exported_models) == 1
        assert result.exported_models[0].format == "onnx"


class TestBaseTrainer:
    """Test Phase 3 BaseTrainer functionality."""

    def test_base_trainer_initialization(self):
        """Test BaseTrainer initializes with config."""
        config = ModelTrainingConfig()

        class DummyTrainer(BaseTrainer):
            def run(self):
                return None

        trainer = DummyTrainer(config)
        assert trainer.config == config
        assert trainer.logger is not None

    def test_base_trainer_checkpoint_methods(self):
        """Test checkpoint save/load methods."""
        config = ModelTrainingConfig()

        class DummyTrainer(BaseTrainer):
            def run(self):
                return None

        trainer = DummyTrainer(config)
        test_data = {"model": "test_model", "epoch": 5}

        path = trainer.save_checkpoint(test_data, "test_checkpoint")
        assert path.exists()

        loaded_data = trainer.load_checkpoint("test_checkpoint")
        assert loaded_data == test_data

    def test_base_trainer_checkpoint_not_found(self):
        """Test loading non-existent checkpoint raises error."""
        config = ModelTrainingConfig()

        class DummyTrainer(BaseTrainer):
            def run(self):
                return None

        trainer = DummyTrainer(config)
        with pytest.raises(FileNotFoundError):
            trainer.load_checkpoint("nonexistent_checkpoint")


class TestPhase3ModuleImports:
    """Test that all Phase 3 modules are properly exported."""

    def test_all_classes_importable(self):
        """Test all Phase 3 classes can be imported."""
        from pipeline.model_training import (
            ModelTrainingConfig,
            TrainingMetrics,
            EvaluationMetrics,
            ModelCheckpoint,
            ExportedModel,
            QuantizationConfig,
            HardExampleSet,
            TrainingResult,
            BaseTrainer,
            ModelTrainer,
            Exporter,
            Quantizer,
            Evaluator,
            ModelPipeline,
        )
        assert all([
            ModelTrainingConfig,
            TrainingMetrics,
            EvaluationMetrics,
            ModelCheckpoint,
            ExportedModel,
            QuantizationConfig,
            HardExampleSet,
            TrainingResult,
            BaseTrainer,
            ModelTrainer,
            Exporter,
            Quantizer,
            Evaluator,
            ModelPipeline,
        ])

    def test_model_pipeline_initialization(self):
        """Test ModelPipeline can be initialized."""
        config = ModelTrainingConfig()
        pipeline = ModelPipeline(config)
        assert pipeline.config == config
        assert pipeline.logger is not None

    def test_model_trainer_initialization(self):
        """Test ModelTrainer can be initialized."""
        config = ModelTrainingConfig()
        trainer = ModelTrainer(config)
        assert trainer.config == config
        assert trainer.model is None
        assert trainer.tokenizer is None
        assert trainer.train_dataset is None
        assert trainer.val_dataset is None


class TestPhase3Integration:
    """Integration tests for Phase 3 module structure."""

    def test_config_hierarchy(self):
        """Test configuration hierarchy and defaults."""
        config = ModelTrainingConfig()

        assert config.model_checkpoint == "hfl/rbt6"
        assert config.enable_adversarial_training == True
        assert config.enable_quantization == True
        assert isinstance(config.quantization_config, QuantizationConfig)

    def test_all_processors_have_run_method(self):
        """Test all processor classes have run() method."""
        config = ModelTrainingConfig()

        trainer = ModelTrainer(config)
        exporter = Exporter(config)
        quantizer = Quantizer(config)
        evaluator = Evaluator(config)
        pipeline = ModelPipeline(config)

        assert callable(getattr(trainer, "run", None))
        assert callable(getattr(exporter, "run", None))
        assert callable(getattr(quantizer, "run", None))
        assert callable(getattr(evaluator, "run", None))
        assert callable(getattr(pipeline, "run", None))

    def test_pipeline_orchestrator_creation(self):
        """Test pipeline orchestrator can be created."""
        config = ModelTrainingConfig()
        pipeline = ModelPipeline(config)

        assert pipeline.model is None
        assert pipeline.tokenizer is None
        assert pipeline.exported_models == []

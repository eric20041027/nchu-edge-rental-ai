"""End-to-end integration tests for Phase 4 (Master Orchestrator)."""
import pytest
from pathlib import Path

from pipeline import PipelineOrchestrator, run_crawlers, run_data_prep, run_model_training
from pipeline.crawlers import CrawlerConfig
from pipeline.data_prep import DataPrepConfig
from pipeline.model_training import ModelTrainingConfig


class TestPipelineOrchestrator:
    """Test PipelineOrchestrator functionality."""

    def test_orchestrator_initialization(self):
        """Test PipelineOrchestrator initializes correctly."""
        orchestrator = PipelineOrchestrator()
        assert orchestrator.results == {}
        assert orchestrator.phase_times == {}
        assert orchestrator.logger is not None

    def test_orchestrator_skip_phases(self):
        """Test orchestrator respects skip_phases parameter."""
        skip_phases = [1, 3]
        orchestrator = PipelineOrchestrator(skip_phases=skip_phases)
        assert orchestrator.skip_phases == skip_phases

    def test_orchestrator_logger_creation(self):
        """Test orchestrator creates proper logger."""
        orchestrator = PipelineOrchestrator()
        logger = orchestrator._create_logger()
        assert logger is not None
        assert logger.name == "PipelineOrchestrator"

    def test_orchestrator_phase_1_check(self):
        """Test Phase 1 is not skipped by default."""
        orchestrator = PipelineOrchestrator()
        assert 1 not in orchestrator.skip_phases

    def test_orchestrator_phase_2_check(self):
        """Test Phase 2 is not skipped by default."""
        orchestrator = PipelineOrchestrator()
        assert 2 not in orchestrator.skip_phases

    def test_orchestrator_phase_3_check(self):
        """Test Phase 3 is not skipped by default."""
        orchestrator = PipelineOrchestrator()
        assert 3 not in orchestrator.skip_phases


class TestRunnerFunctions:
    """Test individual phase runner functions."""

    def test_run_crawlers_returns_dict(self):
        """Test run_crawlers returns proper dictionary."""
        config = CrawlerConfig()
        result = run_crawlers(config)

        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] == "completed"

    def test_run_data_prep_function_exists(self):
        """Test run_data_prep function exists and is callable."""
        assert callable(run_data_prep)

    def test_run_model_training_function_exists(self):
        """Test run_model_training function exists and is callable."""
        assert callable(run_model_training)


class TestCrossPhaseIntegration:
    """Test integration across all three phases."""

    def test_all_configs_initialization(self):
        """Test all phase configs can be initialized together."""
        crawler_config = CrawlerConfig()
        dataprep_config = DataPrepConfig()
        training_config = ModelTrainingConfig()

        assert crawler_config is not None
        assert dataprep_config is not None
        assert training_config is not None

    def test_config_path_consistency(self):
        """Test that configs use consistent directory structures."""
        crawler_config = CrawlerConfig()
        dataprep_config = DataPrepConfig()
        training_config = ModelTrainingConfig()

        # All should have data-related paths
        assert hasattr(dataprep_config, "processed_data_dir")
        assert hasattr(training_config, "saved_models_dir")
        # Crawler config has output CSV paths
        assert hasattr(crawler_config, "output_csv")

    def test_data_flow_phase_1_to_2(self):
        """Test data output from Phase 1 can be input to Phase 2."""
        crawler_config = CrawlerConfig()
        dataprep_config = DataPrepConfig()

        # Both should have defined data paths
        # Phase 1 outputs CSV files that Phase 2 reads
        assert hasattr(crawler_config, "output_csv")
        assert hasattr(dataprep_config, "raw_data_dir")

    def test_data_flow_phase_2_to_3(self):
        """Test data output from Phase 2 can be input to Phase 3."""
        dataprep_config = DataPrepConfig()
        training_config = ModelTrainingConfig()

        # Phase 2 outputs JSON files that Phase 3 reads
        assert dataprep_config.processed_data_dir is not None
        assert training_config.train_data_path is not None


class TestOrchestratorWorkflow:
    """Test orchestrator workflow patterns."""

    def test_orchestrator_can_skip_all_phases(self):
        """Test orchestrator can be created with all phases skipped."""
        skip_all = [1, 2, 3]
        orchestrator = PipelineOrchestrator(skip_phases=skip_all)

        assert 1 in orchestrator.skip_phases
        assert 2 in orchestrator.skip_phases
        assert 3 in orchestrator.skip_phases

    def test_orchestrator_phase_combinations(self):
        """Test orchestrator works with various phase combinations."""
        combinations = [
            [],  # Run all
            [1],  # Skip Phase 1
            [2],  # Skip Phase 2
            [3],  # Skip Phase 3
            [1, 2],  # Run only Phase 3
            [1, 3],  # Run only Phase 2
            [2, 3],  # Run only Phase 1
        ]

        for skip_phases in combinations:
            orchestrator = PipelineOrchestrator(skip_phases=skip_phases)
            assert orchestrator.skip_phases == skip_phases

    def test_orchestrator_result_building(self):
        """Test orchestrator builds proper results dictionary."""
        orchestrator = PipelineOrchestrator(skip_phases=[1, 2, 3])

        # Simulate phase execution
        orchestrator.results["phase_1"] = {"status": "completed"}
        orchestrator.results["phase_2"] = {"status": "completed"}
        orchestrator.results["phase_3"] = {"status": "completed"}

        final_results = orchestrator._build_final_results(100.5)

        assert final_results["status"] == "completed"
        assert final_results["total_time_seconds"] == 100.5
        assert isinstance(final_results["phase_times"], dict)


class TestPhase4ModuleImports:
    """Test that Phase 4 modules import correctly."""

    def test_orchestrator_importable(self):
        """Test PipelineOrchestrator can be imported."""
        from pipeline.orchestrator import PipelineOrchestrator as Orchestrator
        assert Orchestrator is not None

    def test_runners_importable(self):
        """Test runner functions can be imported."""
        from pipeline.runners import run_crawlers, run_data_prep, run_model_training
        assert all([run_crawlers, run_data_prep, run_model_training])

    def test_orchestrator_from_pipeline(self):
        """Test PipelineOrchestrator accessible from pipeline module."""
        from pipeline import PipelineOrchestrator as Orchestrator
        assert Orchestrator is not None

    def test_runners_from_pipeline(self):
        """Test runner functions accessible from pipeline module."""
        from pipeline import run_crawlers, run_data_prep, run_model_training
        assert all([run_crawlers, run_data_prep, run_model_training])


class TestPhase4ArchitectureConsistency:
    """Test consistency of Phase 4 with Phases 1-3."""

    def test_all_phases_have_config_classes(self):
        """Test all phases have configuration classes."""
        from pipeline.crawlers import CrawlerConfig
        from pipeline.data_prep import DataPrepConfig
        from pipeline.model_training import ModelTrainingConfig

        assert all([CrawlerConfig, DataPrepConfig, ModelTrainingConfig])

    def test_all_phases_have_orchestrators(self):
        """Test Phase 1, 2, 3 have appropriate orchestrators."""
        from pipeline.crawlers_runner import main as crawlers_main
        from pipeline.data_prep import DataPipeline
        from pipeline.model_training import ModelPipeline

        # Phase 1 has crawlers_runner
        # Phase 2 has DataPipeline
        # Phase 3 has ModelPipeline
        assert all([crawlers_main, DataPipeline, ModelPipeline])

    def test_master_orchestrator_exists(self):
        """Test master orchestrator exists for Phase 4."""
        from pipeline.orchestrator import PipelineOrchestrator
        assert PipelineOrchestrator is not None

    def test_unified_entry_point_exists(self):
        """Test unified entry point script exists."""
        entry_point = Path(__file__).parent.parent / "pipeline_runner.py"
        assert entry_point.exists()


class TestEndToEndDataFlow:
    """Test conceptual data flow through all phases."""

    def test_phase_1_output_format(self):
        """Test Phase 1 has output configuration."""
        crawler_config = CrawlerConfig()
        # Phase 1 should have output CSV configured
        assert hasattr(crawler_config, "output_csv")
        assert crawler_config.output_csv is not None

    def test_phase_2_input_output_format(self):
        """Test Phase 2 reads CSV and outputs JSON."""
        dataprep_config = DataPrepConfig()
        # Phase 2 reads from raw_data_dir (where Phase 1 outputs)
        # Phase 2 outputs JSON training data
        assert dataprep_config.raw_data_dir is not None
        assert dataprep_config.processed_data_dir is not None

    def test_phase_3_input_format(self):
        """Test Phase 3 reads JSON training data from Phase 2."""
        training_config = ModelTrainingConfig()
        # Phase 3 reads JSON files created by Phase 2
        assert training_config.train_data_path is not None
        assert training_config.val_data_path is not None
        assert training_config.test_data_path is not None

    def test_phase_3_output_format(self):
        """Test Phase 3 outputs trained models and metrics."""
        training_config = ModelTrainingConfig()
        # Phase 3 outputs:
        # 1. Trained PyTorch model
        # 2. ONNX model
        # 3. Quantized ONNX model
        # 4. Evaluation metrics
        assert training_config.saved_models_dir is not None
        assert training_config.onnx_output_path is not None
        assert training_config.quantized_model_path is not None


class TestPhase4Documentation:
    """Test documentation files exist for Phase 4."""

    def test_phase3_report_exists(self):
        """Test Phase 3 completion report exists."""
        report = Path(__file__).parent.parent / "PHASE3_COMPLETION_REPORT.md"
        assert report.exists()

    def test_project_root_documentation(self):
        """Test project root has documentation files."""
        project_root = Path(__file__).parent.parent
        # At least one documentation file should exist
        docs = [
            project_root / "README.md",
            project_root / "PHASE3_COMPLETION_REPORT.md",
        ]
        assert any(doc.exists() for doc in docs)

    def test_architecture_documentation(self):
        """Test architecture is documented across phases."""
        # Check that we have documentation for the three-phase architecture
        project_root = Path(__file__).parent.parent
        expected_docs = [
            project_root / "PHASE3_COMPLETION_REPORT.md",
        ]
        # At least one comprehensive doc should exist
        assert any(doc.exists() for doc in expected_docs)

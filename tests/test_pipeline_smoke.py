from pathlib import Path

from src.pipeline import run_experiment


def test_pipeline_smoke_classical_only(tmp_path):
    data_path = Path(__file__).parent / "data" / "sample_nasa93_small.csv"
    result = run_experiment(data_path=data_path, output_dir=tmp_path, skip_neural=True, test_size=0.25)
    assert result["best_model_name"]
    assert (tmp_path / "corrected_log_effort_model_comparison.csv").exists()
    assert (tmp_path / "corrected_best_model_predictions.csv").exists()
    assert (tmp_path / "actual_vs_predicted.png").exists()

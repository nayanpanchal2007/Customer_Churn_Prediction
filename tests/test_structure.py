from pathlib import Path


def test_required_structure():
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "src" / "data_preprocessing.py",
        root / "src" / "feature_engineering.py",
        root / "src" / "train_models.py",
        root / "src" / "evaluate_models.py",
        root / "src" / "hyperparameter_tuning.py",
        root / "src" / "explainability.py",
        root / "src" / "prediction.py",
        root / "app" / "app.py",
        root / "requirements.txt",
        root / "README.md",
        root / "run.py",
    ]
    assert all(p.exists() for p in required)

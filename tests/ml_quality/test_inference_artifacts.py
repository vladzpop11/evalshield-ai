"""Inference initialization must load prebuilt artifacts without runtime writes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.ml import inference, model_generator
from src.ml.inference import EvalShieldEngine


def forbid_model_generation(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("API runtime must not generate model artifacts")


def patch_out_model_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch both the generator module and the old import location so this test
    # also detects accidental reintroduction of the runtime generator import.
    monkeypatch.setattr(model_generator, "generate_shield_model", forbid_model_generation)
    monkeypatch.setattr(inference, "generate_shield_model", forbid_model_generation, raising=False)


@pytest.mark.ml_quality
def test_engine_loads_valid_prebuilt_artifacts_without_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_out_model_generation(monkeypatch)
    artifacts_dir = Path(__file__).resolve().parents[2] / "src" / "ml" / "artifacts"

    engine = EvalShieldEngine(artifacts_dir=artifacts_dir, preferred_backend="sklearn_joblib")

    assert engine.backend == "sklearn_joblib"
    assert engine.joblib_model is not None


@pytest.mark.ml_quality
def test_missing_model_artifacts_fail_without_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_out_model_generation(monkeypatch)

    with pytest.raises(FileNotFoundError, match="No model artifacts found"):
        EvalShieldEngine(artifacts_dir=tmp_path)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.ml_quality
def test_unusable_model_artifacts_fail_safely_without_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_out_model_generation(monkeypatch)
    (tmp_path / "evalshield_model.joblib").write_bytes(b"not a valid model")

    with pytest.raises(RuntimeError, match="No usable model artifact could be loaded") as exc_info:
        EvalShieldEngine(artifacts_dir=tmp_path, preferred_backend="sklearn_joblib")

    assert str(tmp_path) not in str(exc_info.value)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["evalshield_model.joblib"]

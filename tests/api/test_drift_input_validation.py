"""Validation regressions for drift and recalibration sample inputs."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from src.api.main import app
from src.api.schemas import (
    FEATURE_SAMPLE_RANGES,
    MAX_DRIFT_SAMPLES_PER_FEATURE,
    MAX_DRIFT_SAMPLES_TOTAL,
    DriftCheckRequest,
    RecalibrateRequest,
)
from src.ml.model_generator import FEATURE_NAMES


REQUEST_MODELS = [
    pytest.param(DriftCheckRequest, "feature_samples", id="drift-check"),
    pytest.param(RecalibrateRequest, "production_samples", id="recalibrate"),
]


def make_request(model, field_name: str, samples: dict[str, list[float]]):
    return model(**{field_name: samples})


@pytest.mark.parametrize("model,field_name", REQUEST_MODELS)
def test_valid_feature_names_ranges_and_sample_boundaries_are_accepted(model, field_name):
    for feature_name in FEATURE_NAMES:
        low, high = FEATURE_SAMPLE_RANGES[feature_name]
        make_request(model, field_name, {feature_name: [low, high]})

    make_request(
        model,
        field_name,
        {"toxicity_score": [0.25] * MAX_DRIFT_SAMPLES_PER_FEATURE},
    )

    all_features_at_limit = {
        feature_name: [FEATURE_SAMPLE_RANGES[feature_name][0]] * MAX_DRIFT_SAMPLES_PER_FEATURE
        for feature_name in FEATURE_NAMES
    }
    assert sum(map(len, all_features_at_limit.values())) == MAX_DRIFT_SAMPLES_TOTAL
    make_request(model, field_name, all_features_at_limit)


@pytest.mark.parametrize("model,field_name", REQUEST_MODELS)
def test_per_feature_and_aggregate_sample_limits_are_enforced(model, field_name):
    with pytest.raises(ValidationError):
        make_request(
            model,
            field_name,
            {"toxicity_score": [0.5] * (MAX_DRIFT_SAMPLES_PER_FEATURE + 1)},
        )

    aggregate_over_limit = {
        feature_name: [FEATURE_SAMPLE_RANGES[feature_name][0]] * MAX_DRIFT_SAMPLES_PER_FEATURE
        for feature_name in FEATURE_NAMES
    }
    aggregate_over_limit[FEATURE_NAMES[0]].append(FEATURE_SAMPLE_RANGES[FEATURE_NAMES[0]][0])
    with pytest.raises(ValidationError, match="At most 20000 samples"):
        make_request(model, field_name, aggregate_over_limit)


@pytest.mark.parametrize("model,field_name", REQUEST_MODELS)
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_are_rejected(model, field_name, bad_value):
    with pytest.raises(ValidationError):
        make_request(model, field_name, {"toxicity_score": [bad_value]})


@pytest.mark.parametrize("model,field_name", REQUEST_MODELS)
@pytest.mark.parametrize("feature_name", FEATURE_NAMES[:-1])
@pytest.mark.parametrize("bad_value", [-0.001, 1.001])
def test_unit_interval_features_reject_out_of_range_values(
    model, field_name, feature_name, bad_value
):
    with pytest.raises(ValidationError):
        make_request(model, field_name, {feature_name: [bad_value]})


@pytest.mark.parametrize("model,field_name", REQUEST_MODELS)
@pytest.mark.parametrize("bad_value", [-0.001, 10.001])
def test_token_entropy_rejects_out_of_range_values(model, field_name, bad_value):
    with pytest.raises(ValidationError):
        make_request(model, field_name, {"token_entropy": [bad_value]})


@pytest.mark.parametrize("model,field_name", REQUEST_MODELS)
@pytest.mark.parametrize("feature_name", ["target", "unrecognized_feature"])
def test_non_model_feature_names_are_rejected(model, field_name, feature_name):
    with pytest.raises(ValidationError):
        make_request(model, field_name, {feature_name: [0.5]})


@pytest.mark.parametrize("model,field_name", REQUEST_MODELS)
def test_empty_inputs_and_short_sample_arrays_remain_valid(model, field_name):
    make_request(model, field_name, {})
    make_request(model, field_name, {"token_entropy": []})
    make_request(model, field_name, {"token_entropy": [4.0] * 9})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "field_name"),
    [
        ("/drift/check", "feature_samples"),
        ("/drift/recalibrate", "production_samples"),
    ],
)
async def test_empty_feature_maps_preserve_endpoint_behavior(path, field_name):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post(path, json={field_name: {}})

    assert response.status_code == 200
    if path == "/drift/check":
        assert response.json()["features"] == []
    else:
        assert response.json()["samples_evaluated"] == 0
        assert response.json()["recalibration_status"] == "STABLE_NO_RECALIBRATION_NEEDED"


@pytest.mark.asyncio
async def test_drift_endpoint_preserves_existing_minimum_sample_threshold():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/drift/check", json={"feature_samples": {"token_entropy": [4.0] * 9}}
        )

    assert response.status_code == 200
    assert response.json()["features"] == []


@pytest.mark.asyncio
async def test_recalibration_endpoint_preserves_existing_minimum_sample_threshold():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/drift/recalibrate",
            json={"production_samples": {"token_entropy": [4.0] * 14}},
        )

    assert response.status_code == 200
    assert response.json()["samples_evaluated"] == 0
    assert response.json()["recalibration_status"] == "STABLE_NO_RECALIBRATION_NEEDED"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "field_name"),
    [
        ("/drift/check", "feature_samples"),
        ("/drift/recalibrate", "production_samples"),
    ],
)
async def test_rejected_sample_payloads_return_safe_validation_errors(path, field_name):
    secret_marker = "sensitive-marker-do-not-reflect"
    payload = {field_name: {"target": [secret_marker]}}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post(path, json=payload)

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid request payload."}
    assert secret_marker not in response.text

"""Authentication configuration and access-control regression tests."""

from __future__ import annotations

import secrets
from types import SimpleNamespace

import httpx
import pytest
from fastapi.routing import APIRoute
from httpx import ASGITransport, AsyncClient

from src.api import main as api_main
from src.api.security import APIAuthConfig, documentation_urls, require_api_auth
from src.sdk import EvalShieldClient, EvalShieldException, EvalShieldGuard
import src.sdk.client as sdk_client_module


API_KEY = secrets.token_urlsafe(32)
REPLACEMENT_KEY = secrets.token_urlsafe(32)
METRIC_PAYLOAD = {
    "toxicity_score": 0.1,
    "hallucination_risk": 0.1,
    "prompt_injection_score": 0.1,
    "semantic_drift": 0.1,
    "token_entropy": 4.0,
}


@pytest.fixture
async def api_client():
    async with AsyncClient(
        transport=ASGITransport(app=api_main.app),
        base_url="http://testserver",
    ) as client:
        yield client


def set_api_config(monkeypatch, config: APIAuthConfig) -> None:
    monkeypatch.setitem(api_main.app.state._state, "api_auth_config", config)


def test_demo_configuration_defaults_to_anonymous_and_hides_no_key():
    config = APIAuthConfig.from_env({})

    assert config.mode == "demo"
    assert config.authentication_enabled is False
    assert config.validate() is config
    assert API_KEY not in repr(config)


def test_guard_accepts_optional_key_and_preserves_injected_client():
    guard = EvalShieldGuard(api_key=API_KEY)
    assert guard.client._auth_headers() == {"Authorization": f"Bearer {API_KEY}"}

    injected_client = EvalShieldClient(api_key=REPLACEMENT_KEY)
    injected_guard = EvalShieldGuard(client=injected_client, api_key=API_KEY)
    assert injected_guard.client is injected_client
    assert injected_guard.client._auth_headers() == {"Authorization": f"Bearer {REPLACEMENT_KEY}"}


def test_production_configuration_requires_secure_key_and_never_echoes_it():
    with pytest.raises(RuntimeError, match="EVALSHIELD_API_KEY") as exc_info:
        APIAuthConfig.from_env({"EVALSHIELD_MODE": "production"}).validate()

    assert API_KEY not in str(exc_info.value)

    with pytest.raises(RuntimeError, match="at least 32"):
        APIAuthConfig(mode="production", api_key="short").validate()

    configured = APIAuthConfig(mode="production", api_key=API_KEY).validate()
    assert configured.authentication_enabled is True
    assert API_KEY not in repr(configured)


def test_invalid_mode_fails_validation_without_sensitive_details():
    with pytest.raises(RuntimeError, match="EVALSHIELD_MODE"):
        APIAuthConfig.from_env(
            {"EVALSHIELD_MODE": "prod-ish", "EVALSHIELD_API_KEY": API_KEY}
        ).validate()


def test_production_disables_docs_and_openapi_but_demo_keeps_them():
    assert documentation_urls(APIAuthConfig(mode="demo")) == {
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "openapi_url": "/openapi.json",
    }
    assert documentation_urls(APIAuthConfig(mode="production", api_key=API_KEY)) == {
        "docs_url": None,
        "redoc_url": None,
        "openapi_url": None,
    }


@pytest.mark.asyncio
async def test_production_configuration_is_validated_during_startup(monkeypatch):
    set_api_config(monkeypatch, APIAuthConfig(mode="production"))
    with pytest.raises(RuntimeError, match="EVALSHIELD_API_KEY"):
        async with api_main.lifespan(api_main.app):
            pytest.fail("Invalid production configuration must not start the application")

    valid_config = APIAuthConfig(mode="production", api_key=API_KEY)
    set_api_config(monkeypatch, valid_config)
    for name in ("get_engine", "get_router", "get_extractor", "get_pii", "get_baseline_df"):
        monkeypatch.setattr(api_main, name, lambda: None)

    async with api_main.lifespan(api_main.app):
        assert api_main.app.state.api_auth_config is valid_config


def test_every_fastapi_operation_has_central_auth_dependency_except_health_policy():
    operations = [route for route in api_main.app.routes if isinstance(route, APIRoute)]
    assert {route.path for route in operations} == {
        "/",
        "/health",
        "/model/metadata",
        "/model/shadow-stats",
        "/model/compliance-report",
        "/metrics",
        "/evaluate",
        "/evaluate/stream",
        "/evaluate/batch",
        "/drift/check",
        "/drift/recalibrate",
        "/red-team/simulate",
        "/red-team/mutations",
    }
    assert all(
        any(dependency.call is require_api_auth for dependency in route.dependant.dependencies)
        for route in operations
    )


@pytest.mark.asyncio
async def test_demo_mode_keeps_anonymous_sdk_and_health_workflows(api_client, monkeypatch):
    set_api_config(monkeypatch, APIAuthConfig(mode="demo"))
    monkeypatch.setattr(api_main, "get_engine", lambda: SimpleNamespace(backend="demo"))

    health = await api_client.get("/health")
    mutations = await api_client.get("/red-team/mutations")

    assert health.status_code == 200
    assert mutations.status_code == 200
    assert EvalShieldClient()._auth_headers() == {}


@pytest.mark.asyncio
async def test_documentation_routes_follow_runtime_auth_configuration(api_client, monkeypatch):
    # The app's routes are constructed from import-time demo configuration.
    # Startup may receive the production policy through app.state instead.
    set_api_config(monkeypatch, APIAuthConfig(mode="production", api_key=API_KEY))
    for name in ("get_engine", "get_router", "get_extractor", "get_pii", "get_baseline_df"):
        monkeypatch.setattr(api_main, name, lambda: None)

    async with api_main.lifespan(api_main.app):
        for path in ("/docs", "/redoc", "/openapi.json"):
            anonymous = await api_client.get(path)
            assert anonymous.status_code == 404, path
            assert API_KEY not in anonymous.text

            authenticated = await api_client.get(
                path,
                headers={"Authorization": f"Bearer {API_KEY}"},
            )
            assert authenticated.status_code == 404, path

        # The runtime production policy must not make the deliberately public
        # health check private, and operation routes remain protected.
        monkeypatch.setattr(api_main, "get_engine", lambda: SimpleNamespace(backend="test"))
        assert (await api_client.get("/health")).status_code == 200
        assert (await api_client.get("/red-team/mutations")).status_code == 401

    # Demo mode keeps FastAPI's normal documentation and OpenAPI behavior.
    set_api_config(monkeypatch, APIAuthConfig(mode="demo"))
    for path in ("/docs", "/redoc", "/openapi.json"):
        response = await api_client.get(path)
        assert response.status_code == 200, path


@pytest.mark.asyncio
async def test_production_protects_all_sensitive_endpoints_and_keeps_only_health_public(
    api_client, monkeypatch
):
    set_api_config(monkeypatch, APIAuthConfig(mode="production", api_key=API_KEY))
    monkeypatch.setattr(api_main, "get_engine", lambda: SimpleNamespace(backend="test"))
    cases = [
        ("GET", "/", None),
        ("GET", "/model/metadata", None),
        ("GET", "/model/shadow-stats", None),
        ("GET", "/model/compliance-report", None),
        ("GET", "/metrics", None),
        ("GET", "/red-team/mutations", None),
        ("POST", "/evaluate", METRIC_PAYLOAD),
        ("POST", "/evaluate/stream", METRIC_PAYLOAD),
        ("POST", "/evaluate/batch", {"items": [METRIC_PAYLOAD]}),
        ("POST", "/drift/check", {"feature_samples": {}}),
        ("POST", "/drift/recalibrate", {"production_samples": {}}),
        ("POST", "/red-team/simulate", {"seed_prompt": "safe seed"}),
    ]

    for method, path, payload in cases:
        response = await api_client.request(method, path, json=payload)
        assert response.status_code == 401, path
        assert API_KEY not in response.text

    public_health = await api_client.get("/health")
    assert public_health.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "authorization",
    [None, "Basic not-a-bearer", "Bearer", "Bearer token extra", "Bearer " + REPLACEMENT_KEY],
)
async def test_missing_malformed_and_invalid_credentials_are_uniform_and_do_not_echo_input(
    api_client, monkeypatch, authorization
):
    set_api_config(monkeypatch, APIAuthConfig(mode="production", api_key=API_KEY))
    headers = {"Authorization": authorization} if authorization is not None else {}
    supplied_token = authorization or ""
    response = await api_client.post(
        "/evaluate",
        json={"prompt_text": f"private content {supplied_token}"},
        headers=headers,
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Missing or invalid bearer token."}
    assert API_KEY not in response.text
    if supplied_token:
        assert supplied_token not in response.text
    assert "private content" not in response.text
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.asyncio
async def test_query_parameter_never_authenticates_and_valid_header_does(api_client, monkeypatch):
    set_api_config(monkeypatch, APIAuthConfig(mode="production", api_key=API_KEY))
    monkeypatch.setattr(api_main, "get_engine", lambda: SimpleNamespace(backend="test"))

    query_auth = await api_client.get(f"/red-team/mutations?api_key={API_KEY}")
    assert query_auth.status_code == 401
    assert API_KEY not in query_auth.text

    valid = await api_client.get(
        "/red-team/mutations",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    assert valid.status_code == 200
    assert API_KEY not in valid.text


def test_sync_sdk_sends_api_key_for_evaluation_and_red_team(monkeypatch):
    seen_headers = []
    transport = httpx.MockTransport(
        lambda request: (seen_headers.append(request.headers.get("authorization")) or
                         httpx.Response(200, json={"ok": True}, request=request))
    )
    real_client = httpx.Client

    def client_with_transport(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(sdk_client_module.httpx, "Client", client_with_transport)
    client = EvalShieldClient(base_url="https://unit.test", api_key=API_KEY)

    client.evaluate(prompt_text="safe")
    client.simulate_red_team("safe seed", trigger_recalibration=False)

    assert seen_headers == [f"Bearer {API_KEY}", f"Bearer {API_KEY}"]


@pytest.mark.asyncio
async def test_async_sdk_sends_api_key_for_evaluation_and_red_team(monkeypatch):
    seen_headers = []

    def handler(request):
        seen_headers.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"ok": True}, request=request)

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def async_client_with_transport(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(sdk_client_module.httpx, "AsyncClient", async_client_with_transport)
    client = EvalShieldClient(base_url="https://unit.test", api_key=API_KEY)

    await client.async_evaluate(prompt_text="safe")
    await client.async_simulate_red_team("safe seed", trigger_recalibration=False)

    assert seen_headers == [f"Bearer {API_KEY}", f"Bearer {API_KEY}"]


def test_sync_sdk_auth_error_does_not_echo_configured_key(monkeypatch):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(401, text=API_KEY, request=request)
    )
    real_client = httpx.Client

    def client_with_transport(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(sdk_client_module.httpx, "Client", client_with_transport)
    with pytest.raises(EvalShieldException) as exc_info:
        EvalShieldClient(base_url="https://unit.test", api_key=API_KEY).evaluate(prompt_text="safe")

    assert "401" in str(exc_info.value)
    assert API_KEY not in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_sdk_auth_error_does_not_echo_configured_key(monkeypatch):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(401, text=API_KEY, request=request)
    )
    real_client = httpx.AsyncClient

    def async_client_with_transport(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(sdk_client_module.httpx, "AsyncClient", async_client_with_transport)
    with pytest.raises(EvalShieldException) as exc_info:
        await EvalShieldClient(base_url="https://unit.test", api_key=API_KEY).async_evaluate(
            prompt_text="safe"
        )

    assert "401" in str(exc_info.value)
    assert API_KEY not in str(exc_info.value)

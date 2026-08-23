import httpx
from atlas_api.main import create_app
from atlas_core.config import Settings


async def test_health_endpoint_reports_ok() -> None:
    app = create_app(Settings(environment="test"))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["environment"] == "test"


async def test_health_reflects_configured_environment() -> None:
    app = create_app(Settings(environment="staging"))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")

    assert response.json()["environment"] == "staging"

from __future__ import annotations

from app.api.dependencies import get_auth_service
from app.main import app
from app.services.auth_service import AuthenticatedUser


class FakeAuthService:
    async def register_user(self, db_session, *, username: str, password: str, email: str | None = None):
        return AuthenticatedUser(user_id="user_001", username=username, email=email)

    async def authenticate_user(self, db_session, *, username: str, password: str):
        return AuthenticatedUser(user_id="user_001", username=username, email="demo@example.com"), "jwt-token"

    async def get_user_by_id(self, db_session, *, user_id: str):
        return AuthenticatedUser(user_id=user_id, username="demo_user", email="demo@example.com")


def unwrap(payload: dict):
    return payload["data"] if "data" in payload else payload


async def test_register(api_client):
    app.dependency_overrides[get_auth_service] = lambda: FakeAuthService()
    response = await api_client.post(
        "/api/v1/auth/register",
        json={
            "username": "demo_user",
            "password": "demo123456",
            "email": "demo@example.com",
        },
    )
    assert response.status_code == 200
    data = unwrap(response.json())
    assert data["user_id"] == "user_001"
    assert data["username"] == "demo_user"


async def test_login(api_client):
    app.dependency_overrides[get_auth_service] = lambda: FakeAuthService()
    response = await api_client.post(
        "/api/v1/auth/login",
        json={
            "username": "demo_user",
            "password": "demo123456",
        },
    )
    assert response.status_code == 200
    data = unwrap(response.json())
    assert data["access_token"] == "jwt-token"
    assert data["token_type"] == "bearer"


async def test_me(api_client):
    app.dependency_overrides[get_auth_service] = lambda: FakeAuthService()
    response = await api_client.get("/api/v1/auth/me")
    assert response.status_code == 200
    data = unwrap(response.json())
    assert data["user_id"] == "test-user-001"

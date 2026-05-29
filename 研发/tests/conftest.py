from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass

import httpx
import pytest

from app.api.dependencies import get_current_user_id, get_db_session, get_role_service
from app.main import app
from app.services.role_service import RoleRecord


class DummySession:
    async def execute(self, *args, **kwargs):
        return None

    async def commit(self):
        return None


class FakeRoleService:
    async def list_roles(self, db_session, user_id=None):
        return [
            RoleRecord(
                role_id="lawyer_01",
                name="民事律师",
                category="lawyer",
                role_type="preset",
                system_prompt="lawyer prompt",
                knowledge_base_id="kb_lawyer_default",
            ),
            RoleRecord(
                role_id="custom_00001",
                name="我的顾问",
                category="general",
                role_type="custom",
                system_prompt="custom prompt",
                knowledge_base_id=None,
            ),
        ]

    async def resolve_role(self, db_session, *, user_id, role_id=None, role_name=None):
        return RoleRecord(
            role_id=role_id or "lawyer_01",
            name=role_name or "民事律师",
            category="lawyer" if (role_id or "").startswith("lawyer") or role_name == "民事律师" else "general",
            role_type="preset" if role_id == "lawyer_01" else "custom",
            system_prompt="role prompt",
            knowledge_base_id="kb_lawyer_default",
        )

    async def detect_role(self, db_session, *, user_id, query):
        return (
            RoleRecord(
                role_id="lawyer_01",
                name="民事律师",
                category="lawyer",
                role_type="preset",
                system_prompt="role prompt",
                knowledge_base_id="kb_lawyer_default",
            ),
            "matched",
            0.9,
            "matched_by_keyword:合同",
        )

    async def create_custom_role(self, db_session, *, user_id, name, system_prompt, category="general", role_type="custom", knowledge_base_id=None):
        return RoleRecord(
            role_id="custom_00002",
            name=name,
            category=category,
            role_type=role_type,
            system_prompt=system_prompt,
            knowledge_base_id=knowledge_base_id,
        )


@pytest.fixture()
def fake_role_service():
    return FakeRoleService()


@pytest.fixture()
async def api_client(fake_role_service) -> AsyncGenerator[httpx.AsyncClient, None]:
    async def override_db():
        yield DummySession()

    async def override_current_user():
        return "test-user-001"

    app.dependency_overrides[get_db_session] = override_db
    app.dependency_overrides[get_current_user_id] = override_current_user
    app.dependency_overrides[get_role_service] = lambda: fake_role_service

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client

    app.dependency_overrides.clear()

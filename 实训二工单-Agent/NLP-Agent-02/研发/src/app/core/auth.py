from typing import Annotated

from fastapi import Header, Request
from structlog import get_logger

from app.core.config import settings
from app.core.database import async_session_factory
from app.core.exceptions import UnauthorizedError
from app.repositories.admin_audit_repository import AdminAccessAuditRepository
from app.schemas.admin_audit import AdminAccessAuditLogCreate

logger = get_logger()


def _build_admin_audit_fields(request: Request) -> dict[str, str | None]:
    client_host = request.client.host if request.client else None
    request_id = getattr(request.state, "request_id", None)
    return {
        "path": request.url.path,
        "method": request.method,
        "client_host": client_host,
        "request_id": request_id,
    }


async def _persist_admin_access_audit(
    request: Request,
    *,
    access_granted: bool,
    auth_mode: str,
    failure_reason: str | None,
) -> None:
    async with async_session_factory() as session:
        repository = AdminAccessAuditRepository(session)
        await repository.create(
            AdminAccessAuditLogCreate(
                **_build_admin_audit_fields(request),
                access_granted=access_granted,
                auth_mode=auth_mode,
                failure_reason=failure_reason,
            )
        )


async def require_admin_access(
    request: Request,
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
) -> None:
    expected_token = settings.admin_access_token
    if not expected_token:
        logger.info("admin_access_granted_dev_mode", **_build_admin_audit_fields(request))
        await _persist_admin_access_audit(
            request,
            access_granted=True,
            auth_mode="dev_mode",
            failure_reason=None,
        )
        return
    if x_admin_token != expected_token:
        logger.warning("admin_access_denied", **_build_admin_audit_fields(request))
        await _persist_admin_access_audit(
            request,
            access_granted=False,
            auth_mode="token",
            failure_reason="invalid_admin_token",
        )
        raise UnauthorizedError("Invalid admin token")
    logger.info("admin_access_granted", **_build_admin_audit_fields(request))
    await _persist_admin_access_audit(
        request,
        access_granted=True,
        auth_mode="token",
        failure_reason=None,
    )

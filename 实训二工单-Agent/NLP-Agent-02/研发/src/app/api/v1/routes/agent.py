from typing import Annotated
# FastAPI 路由文件，专门处理 Agent（智能体）相关的 API 请求。
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.orchestrator import AgentOrchestrator
from app.core.database import get_db_session
from app.schemas.agent import (
    AgentExecuteResponse,
    AgentNaturalLanguageRequest,
    AgentStateResponse,
    AgentToolRequest,
)
from app.schemas.common import ApiResponse
from app.schemas.conversation import ConversationSessionList, ConversationSessionView
from app.schemas.conversation_history import ConversationHistoryList, ConversationHistoryRead
from app.schemas.llm_audit import LLMAuditLogList, LLMAuditLogRead
from app.services.conversation_history_service import (
    ConversationHistoryService,
    get_conversation_history_service,
)
from app.services.conversation_service import ConversationService, get_conversation_service
from app.services.llm_audit_service import LLMAuditLogService, get_llm_audit_log_service

router = APIRouter()


@router.post("/execute", response_model=ApiResponse[AgentExecuteResponse], status_code=status.HTTP_200_OK)
async def execute_agent_tool(
    payload: AgentToolRequest,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> ApiResponse[AgentExecuteResponse]:
    orchestrator = AgentOrchestrator(session)
    result = await orchestrator.execute(payload)
    return ApiResponse(code=200, message="success", data=result)


@router.post(
    "/chat",
    response_model=ApiResponse[AgentStateResponse],
    status_code=status.HTTP_200_OK,
)
async def execute_agent_chat(
    payload: AgentNaturalLanguageRequest,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> ApiResponse[AgentStateResponse]:
    orchestrator = AgentOrchestrator(session)
    result = await orchestrator.execute_natural_language(payload)
    return ApiResponse(code=200, message="success", data=result)


@router.get(
    "/sessions",
    response_model=ApiResponse[ConversationSessionList],
    status_code=status.HTTP_200_OK,
)
async def list_agent_sessions(
    service: Annotated[ConversationService, Depends(get_conversation_service)],
    include_expired: Annotated[bool, Query()] = False,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ApiResponse[ConversationSessionList]:
    result = await service.list_sessions(
        include_expired=include_expired,
        limit=limit,
        offset=offset,
    )
    return ApiResponse(code=200, message="success", data=result)


@router.get(
    "/sessions/history",
    response_model=ApiResponse[ConversationHistoryList],
    status_code=status.HTTP_200_OK,
)
async def list_agent_session_history(
    service: Annotated[ConversationHistoryService, Depends(get_conversation_history_service)],
    session_id: Annotated[str | None, Query()] = None,
    parser_source: Annotated[str | None, Query()] = None,
    agent_state: Annotated[str | None, Query()] = None,
    intent: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ApiResponse[ConversationHistoryList]:
    result = await service.list_logs(
        session_id=session_id,
        parser_source=parser_source,
        agent_state=agent_state,
        intent=intent,
        limit=limit,
        offset=offset,
    )
    return ApiResponse(code=200, message="success", data=result)


@router.get(
    "/sessions/{session_id}/history",
    response_model=ApiResponse[list[ConversationHistoryRead]],
    status_code=status.HTTP_200_OK,
)
async def get_agent_session_history(
    session_id: str,
    service: Annotated[ConversationHistoryService, Depends(get_conversation_history_service)],
) -> ApiResponse[list[ConversationHistoryRead]]:
    result = await service.list_by_session_id(session_id)
    return ApiResponse(code=200, message="success", data=result)


@router.get(
    "/sessions/{session_id}",
    response_model=ApiResponse[ConversationSessionView],
    status_code=status.HTTP_200_OK,
)
async def get_agent_session(
    session_id: str,
    service: Annotated[ConversationService, Depends(get_conversation_service)],
) -> ApiResponse[ConversationSessionView]:
    result = await service.get_session(session_id)
    return ApiResponse(code=200, message="success", data=result)


@router.delete(
    "/sessions/{session_id}",
    response_model=ApiResponse[dict[str, str]],
    status_code=status.HTTP_200_OK,
)
async def delete_agent_session(
    session_id: str,
    service: Annotated[ConversationService, Depends(get_conversation_service)],
) -> ApiResponse[dict[str, str]]:
    await service.clear_confirmation(session_id)
    return ApiResponse(code=200, message="deleted", data={"session_id": session_id})


@router.get(
    "/llm-audit",
    response_model=ApiResponse[LLMAuditLogList],
    status_code=status.HTTP_200_OK,
)
async def list_llm_audit_logs(
    service: Annotated[LLMAuditLogService, Depends(get_llm_audit_log_service)],
    session_id: Annotated[str | None, Query()] = None,
    parser_stage: Annotated[str | None, Query()] = None,
    success: Annotated[bool | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ApiResponse[LLMAuditLogList]:
    result = await service.list_logs(
        session_id=session_id,
        parser_stage=parser_stage,
        success=success,
        limit=limit,
        offset=offset,
    )
    return ApiResponse(code=200, message="success", data=result)


@router.get(
    "/llm-audit/{session_id}",
    response_model=ApiResponse[list[LLMAuditLogRead]],
    status_code=status.HTTP_200_OK,
)
async def list_llm_audit_logs_by_session(
    session_id: str,
    service: Annotated[LLMAuditLogService, Depends(get_llm_audit_log_service)],
) -> ApiResponse[list[LLMAuditLogRead]]:
    result = await service.list_by_session_id(session_id)
    return ApiResponse(code=200, message="success", data=result)

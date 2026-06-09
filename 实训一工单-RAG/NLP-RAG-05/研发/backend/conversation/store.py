from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
import json
from threading import Lock

from backend.config import settings
from backend.conversation.models import ConversationState

try:
    import redis
except Exception:  # pragma: no cover
    redis = None


class ConversationStore(ABC):
    @abstractmethod
    def get(self, session_id: str) -> ConversationState | None:
        raise NotImplementedError

    @abstractmethod
    def save(self, state: ConversationState) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, session_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def touch(self, session_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_recent_turns(self, session_id: str, limit: int) -> list:
        raise NotImplementedError


class InMemoryConversationStore(ConversationStore):
    def __init__(self, ttl_seconds: int, history_limit: int) -> None:
        self.ttl_seconds = max(60, ttl_seconds)
        self.history_limit = max(1, history_limit)
        self._states: dict[str, tuple[ConversationState, datetime]] = {}
        self._lock = Lock()

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _cleanup(self) -> None:
        now = self._now()
        expired = [session_id for session_id, (_, expiry) in self._states.items() if expiry <= now]
        for session_id in expired:
            self._states.pop(session_id, None)

    def get(self, session_id: str) -> ConversationState | None:
        with self._lock:
            self._cleanup()
            payload = self._states.get(session_id)
            if not payload:
                return None
            state, expiry = payload
            if expiry <= self._now():
                self._states.pop(session_id, None)
                return None
            return state.model_copy(deep=True)

    def save(self, state: ConversationState) -> None:
        with self._lock:
            state.history_turns = state.history_turns[-self.history_limit :]
            self._states[state.session_id] = (state.model_copy(deep=True), self._now() + timedelta(seconds=self.ttl_seconds))

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._states.pop(session_id, None)

    def touch(self, session_id: str) -> None:
        with self._lock:
            payload = self._states.get(session_id)
            if not payload:
                return
            state, _ = payload
            self._states[session_id] = (state, self._now() + timedelta(seconds=self.ttl_seconds))

    def list_recent_turns(self, session_id: str, limit: int) -> list:
        state = self.get(session_id)
        if not state:
            return []
        return state.history_turns[-max(1, limit) :]


class RedisConversationStore(ConversationStore):
    def __init__(self, redis_uri: str, ttl_seconds: int, history_limit: int) -> None:
        if redis is None:
            raise RuntimeError("redis package is not installed")
        self.client = redis.from_url(redis_uri, decode_responses=True)
        self.ttl_seconds = max(60, ttl_seconds)
        self.history_limit = max(1, history_limit)

    def _key(self, session_id: str) -> str:
        return f"rag:conv:{session_id}"

    def get(self, session_id: str) -> ConversationState | None:
        raw = self.client.get(self._key(session_id))
        if not raw:
            return None
        return ConversationState.model_validate_json(raw)

    def save(self, state: ConversationState) -> None:
        state.history_turns = state.history_turns[-self.history_limit :]
        self.client.set(self._key(state.session_id), state.model_dump_json(), ex=self.ttl_seconds)

    def delete(self, session_id: str) -> None:
        self.client.delete(self._key(session_id))

    def touch(self, session_id: str) -> None:
        self.client.expire(self._key(session_id), self.ttl_seconds)

    def list_recent_turns(self, session_id: str, limit: int) -> list:
        state = self.get(session_id)
        if not state:
            return []
        return state.history_turns[-max(1, limit) :]


def build_conversation_store() -> ConversationStore:
    backend = str(settings.conversation_store_backend or "memory").strip().lower()
    if backend == "redis":
        return RedisConversationStore(
            redis_uri=settings.redis_uri,
            ttl_seconds=settings.conversation_ttl_seconds,
            history_limit=settings.conversation_history_limit,
        )
    return InMemoryConversationStore(
        ttl_seconds=settings.conversation_ttl_seconds,
        history_limit=settings.conversation_history_limit,
    )

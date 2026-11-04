"""Session management: persist conversation events/state across multiple run() calls."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from agents_bundle.data_types import Event


class Session(BaseModel):
    """Container for persistent conversation state across multiple run() calls."""

    session_id: str
    user_id: str | None = None
    events: list[Event] = Field(default_factory=list)
    state: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class BaseSessionManager(ABC):
    """Abstract base class for session storage backends."""

    @abstractmethod
    async def create(self, session_id: str, user_id: str | None = None) -> Session:
        """Create a new session."""

    @abstractmethod
    async def get(self, session_id: str) -> Session | None:
        """Retrieve a session by ID. Returns None if not found."""

    @abstractmethod
    async def save(self, session: Session) -> None:
        """Persist session changes to storage."""

    async def get_or_create(self, session_id: str, user_id: str | None = None) -> Session:
        """Get existing session or create a new one."""
        session = await self.get(session_id)
        if session is None:
            session = await self.create(session_id, user_id)
        return session


class InMemorySessionManager(BaseSessionManager):
    """In-memory session storage for development and testing (lost on restart)."""

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    async def create(self, session_id: str, user_id: str | None = None) -> Session:
        if session_id in self._sessions:
            raise ValueError(f"Session {session_id} already exists")
        session = Session(session_id=session_id, user_id=user_id)
        self._sessions[session_id] = session
        return session

    async def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    async def save(self, session: Session) -> None:
        session.updated_at = datetime.now()
        self._sessions[session.session_id] = session

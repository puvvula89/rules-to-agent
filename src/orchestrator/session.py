from typing import Dict, Any, List
from pydantic import BaseModel, Field

class SessionLedger(BaseModel):
    account_context: Dict[str, Any] = Field(default_factory=dict)
    line_context: Dict[str, Any] = Field(default_factory=dict)
    trade_in_context: Dict[str, Any] = Field(default_factory=dict)
    new_device_context: Dict[str, Any] = Field(default_factory=dict)
    order_context: Dict[str, Any] = Field(default_factory=dict)

class SessionData(BaseModel):
    session_id: str
    ledger: SessionLedger = Field(default_factory=SessionLedger)
    transcript: List[Dict[str, str]] = Field(default_factory=list)
    current_state: str = "Auth"
    current_objective: str = ""

class SessionManager:
    """
    Abstracted State Management using the Repository Pattern.
    Starts with in-memory dict for POC. Can swap to Redis later.
    """
    def __init__(self):
        # In-memory store for POC (Day 1-3 strategy)
        self._store: Dict[str, SessionData] = {}

    def get_session(self, session_id: str) -> SessionData:
        """Retrieves or creates a session."""
        if session_id not in self._store:
            self._store[session_id] = SessionData(session_id=session_id)
        return self._store[session_id]

    def save_session(self, session: SessionData):
        """Saves the session state."""
        # Enforce sliding window transcript (last 5 turns)
        if len(session.transcript) > 5:
            session.transcript = session.transcript[-5:]
        self._store[session.session_id] = session

    def clear_memory_keys(self, session_id: str, keys_to_clear: List[str]):
        """Wipes specific ledger contexts based on global transitions."""
        session = self.get_session(session_id)
        for key in keys_to_clear:
            if hasattr(session.ledger, key):
                setattr(session.ledger, key, {})
        self.save_session(session)

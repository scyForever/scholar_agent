from __future__ import annotations

from collections import defaultdict
from typing import Dict

from src.core.models import DialogueState


class DialogueManager:
    def __init__(self) -> None:
        self._states: Dict[str, DialogueState] = defaultdict(DialogueState)

    def get_state(self, session_id: str) -> DialogueState:
        return self._states[session_id]

    def update_state(self, session_id: str, **kwargs: object) -> DialogueState:
        state = self._states[session_id]
        for key, value in kwargs.items():
            setattr(state, key, value)
        return state

    def add_user_message(self, session_id: str, message: str) -> None:
        self._states[session_id].history.append({"role": "user", "content": message})

    def add_assistant_message(self, session_id: str, message: str) -> None:
        self._states[session_id].history.append({"role": "assistant", "content": message})

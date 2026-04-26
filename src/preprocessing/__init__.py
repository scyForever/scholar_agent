from importlib import import_module
from typing import Any


_EXPORTS = {
    "DialogueManager": ".dialogue_manager",
    "IntentClassifier": ".intent_classifier",
    "QueryRewriter": ".query_rewriter",
    "SlotFiller": ".slot_filler",
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value

__all__ = ["DialogueManager", "IntentClassifier", "QueryRewriter", "SlotFiller"]

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass(slots=True)
class ToolParameter:
    name: str
    type_name: str
    description: str
    required: bool = True


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Callable[..., Any]] = {}
        self._definitions: Dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition, func: Callable[..., Any]) -> Callable[..., Any]:
        self._tools[definition.name] = func
        self._definitions[definition.name] = definition
        return func

    def call(self, name: str, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise KeyError(f"Tool not registered: {name}")
        return self._tools[name](**kwargs)

    def get_definition(self, name: str) -> ToolDefinition:
        return self._definitions[name]

    def list_tools(self) -> List[ToolDefinition]:
        return list(self._definitions.values())


TOOL_REGISTRY = ToolRegistry()


def register_tool(definition: ToolDefinition) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return TOOL_REGISTRY.register(definition, func)

    return decorator

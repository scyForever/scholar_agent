from __future__ import annotations

from dataclasses import dataclass, field
from inspect import signature
from typing import Any, Callable, Dict, List, Sequence, Type

from pydantic import Field, create_model

try:
    from langchain_core.tools import BaseTool, StructuredTool
except ImportError:  # pragma: no cover
    BaseTool = None
    StructuredTool = None


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
        self._langchain_tools: Dict[str, BaseTool] = {}

    def register(self, definition: ToolDefinition, func: Callable[..., Any]) -> Callable[..., Any]:
        self._tools[definition.name] = func
        self._definitions[definition.name] = definition
        langchain_tool = self._build_langchain_tool(definition, func)
        if langchain_tool is not None:
            self._langchain_tools[definition.name] = langchain_tool
        return func

    def call(self, name: str, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise KeyError(f"Tool not registered: {name}")
        return self._tools[name](**kwargs)

    def get_definition(self, name: str) -> ToolDefinition:
        return self._definitions[name]

    def list_tools(self) -> List[ToolDefinition]:
        return list(self._definitions.values())

    def get_langchain_tool(self, name: str) -> BaseTool:
        if name not in self._langchain_tools:
            raise KeyError(f"LangChain tool not registered: {name}")
        return self._langchain_tools[name]

    def list_langchain_tools(
        self,
        *,
        names: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
    ) -> List[BaseTool]:
        selected_names = list(names) if names is not None else list(self._langchain_tools)
        required_tags = set(tags or [])
        tools: List[BaseTool] = []
        for name in selected_names:
            tool = self._langchain_tools.get(name)
            definition = self._definitions.get(name)
            if tool is None or definition is None:
                continue
            if required_tags and not required_tags.intersection(definition.tags):
                continue
            tools.append(tool)
        return tools

    def _build_langchain_tool(self, definition: ToolDefinition, func: Callable[..., Any]) -> BaseTool | None:
        if StructuredTool is None:
            return None
        args_schema = self._build_args_schema(definition, func)
        return StructuredTool.from_function(
            func=func,
            name=definition.name,
            description=self._langchain_description(definition),
            args_schema=args_schema,
            infer_schema=False,
        )

    def _build_args_schema(self, definition: ToolDefinition, func: Callable[..., Any]) -> Type[Any]:
        func_signature = signature(func)
        fields: Dict[str, tuple[Any, Any]] = {}
        for parameter in definition.parameters:
            annotation = self._resolve_type(parameter.type_name)
            fields[parameter.name] = (
                annotation,
                self._field_for_parameter(parameter, func_signature.parameters.get(parameter.name)),
            )
        model_name = "".join(part.capitalize() for part in definition.name.split("_")) + "Input"
        return create_model(model_name, **fields)

    def _resolve_type(self, type_name: str) -> Any:
        normalized = type_name.strip().lower()
        mapping: Dict[str, Any] = {
            "str": str,
            "string": str,
            "int": int,
            "integer": int,
            "float": float,
            "bool": bool,
            "boolean": bool,
            "dict": dict[str, Any],
            "list": list[Any],
        }
        return mapping.get(normalized, str)

    def _default_value(self, type_name: str) -> Any:
        normalized = type_name.strip().lower()
        defaults: Dict[str, Any] = {
            "str": "",
            "string": "",
            "int": 0,
            "integer": 0,
            "float": 0.0,
            "bool": False,
            "boolean": False,
            "dict": {},
            "list": [],
        }
        return defaults.get(normalized, "")

    def _field_for_parameter(self, parameter: ToolParameter, signature_parameter: Any) -> Any:
        if parameter.required:
            return Field(default=..., description=parameter.description)
        if signature_parameter is not None and signature_parameter.default is not signature_parameter.empty:
            return Field(default=signature_parameter.default, description=parameter.description)
        normalized = parameter.type_name.strip().lower()
        if normalized == "dict":
            return Field(default_factory=dict, description=parameter.description)
        if normalized == "list":
            return Field(default_factory=list, description=parameter.description)
        return Field(default=self._default_value(parameter.type_name), description=parameter.description)

    def _langchain_description(self, definition: ToolDefinition) -> str:
        if not definition.parameters:
            return definition.description
        parameter_lines = [definition.description, "", "参数说明："]
        for parameter in definition.parameters:
            required = "必填" if parameter.required else "可选"
            parameter_lines.append(f"- {parameter.name}: {parameter.description}（{required}）")
        return "\n".join(parameter_lines)


TOOL_REGISTRY = ToolRegistry()


def register_tool(definition: ToolDefinition) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return TOOL_REGISTRY.register(definition, func)

    return decorator

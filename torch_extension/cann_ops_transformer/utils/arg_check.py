# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Python-side argument type checks with clear parameter names in errors.

Ops normally need no direct use: ``OpBuilder.load()`` already applies
``wrap_op_module``. ``@check_args`` / ``require_*`` are for pure-Python APIs.

Policy: only reject clear type mismatches; uncertain cases are left to pybind
(prefer false pass over false reject).
"""

from __future__ import annotations

import functools
import inspect
import numbers
import re
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import torch


class ArgTypeError(TypeError):
    """Type mismatch for a named argument; carries fields for rich reformatting."""

    def __init__(self, name: str, expected: str, value: Any):
        self.param_name = name
        self.expected = expected
        self.value = value
        super().__init__(self.short_message())

    def short_message(self) -> str:
        actual = _type_name(self.value)
        value_repr = _safe_value_repr(self.value)
        if value_repr is not None:
            return (
                f"{self.param_name} must be {self.expected}, "
                f"got {actual} (value={value_repr})"
            )
        return f"{self.param_name} must be {self.expected}, got {actual}"


def _type_name(value) -> str:
    return type(value).__name__


def _safe_value_repr(value, *, max_len: int = 64) -> Optional[str]:
    """Repr for scalar-like illegal values; skip Tensor / huge objects."""
    if isinstance(value, torch.Tensor):
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return None
    if isinstance(value, str):
        text = repr(value)
        if len(text) <= max_len:
            return text
        inner = value[: max(0, (max_len - 5) // 2)]
        return repr(inner)[:-1] + "...'"
    if isinstance(value, (int, float, bool, type(None))):
        return repr(value)
    if isinstance(value, (list, tuple)) and len(value) <= 4:
        if all(isinstance(x, (int, float, bool, str, type(None))) for x in value):
            text = repr(value)
            return text if len(text) <= max_len else None
    return None


def _invoked_value_repr(value, *, max_len: int = 64) -> str:
    """Compact value for the Invoked-with line (pybind-like positional list)."""
    if isinstance(value, torch.Tensor):
        return f"<Tensor {tuple(value.shape)} {value.dtype}>"
    scalar = _safe_value_repr(value, max_len=max_len)
    return scalar if scalar is not None else "..."


def _annotation_label(annotation) -> str:
    """Human-readable expected type for error messages."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return "Any"
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(args) == len(non_none) + 1 and len(non_none) == 1:
            return f"Optional[{_annotation_label(non_none[0])}]"
        return " | ".join(_annotation_label(a) for a in args)
    if origin in (list, List):
        inner = _annotation_label(args[0]) if args else "Any"
        return f"list[{inner}]"
    if origin in (tuple, Tuple):
        if args:
            return "tuple[" + ", ".join(_annotation_label(a) for a in args) + "]"
        return "tuple"
    if origin in (dict, Dict):
        if len(args) >= 2:
            return f"dict[{_annotation_label(args[0])}, {_annotation_label(args[1])}]"
        return "dict"
    if annotation is torch.Tensor:
        return "Tensor"
    if annotation is torch.dtype:
        return "dtype"
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).replace("torch.", "")


def _format_type_error(
    api_name: str,
    param_names: Sequence[str],
    annotations: Dict[str, Any],
    bound_values: Dict[str, Any],
    err: ArgTypeError,
) -> str:
    """Build a pybind-shaped TypeError message (supported + Invoked with + mismatch)."""
    supported = ", ".join(
        f"{name}: {_annotation_label(annotations.get(name, Any))}"
        for name in param_names
    )
    invoked = ", ".join(
        _invoked_value_repr(bound_values[name])
        for name in param_names
        if name in bound_values
    )
    return (
        f"{api_name}(): incompatible function arguments. "
        f"The following argument types are supported:\n"
        f"    1. ({supported})\n"
        f"\n"
        f"Invoked with: {invoked}\n"
        f"Mismatched argument: {err.short_message()}"
    )


def _raise_type_error(name: str, expected: str, value) -> None:
    raise ArgTypeError(name, expected, value)


def require_int(name: str, value, *, allow_bool: bool = False) -> int:
    """Accept int-like values; do not reject ambiguous cases (e.g. bool, numpy)."""
    if allow_bool and isinstance(value, bool):
        return int(value)
    # Includes bool (int subclass); ambiguous cases are left to the callee.
    if isinstance(value, int):
        return value
    if isinstance(value, numbers.Integral):
        return value  # type: ignore[return-value]
    _raise_type_error(name, "int", value)


def require_optional_int(
    name: str, value, *, allow_bool: bool = False
) -> Optional[int]:
    if value is None:
        return None
    return require_int(name, value, allow_bool=allow_bool)


def require_float(name: str, value, *, allow_int: bool = False) -> float:
    """Accept float-like values; do not reject ambiguous cases (e.g. bool, numpy)."""
    if isinstance(value, float):
        return value
    if allow_int and isinstance(value, int):  # includes bool
        return float(value)
    if isinstance(value, numbers.Real) and not isinstance(value, (int, bool)):
        return value  # type: ignore[return-value]
    # bool/int when allow_int=False: ambiguous — do not reject.
    if isinstance(value, (int, bool)):
        return value  # type: ignore[return-value]
    _raise_type_error(name, "float", value)


def require_optional_float(
    name: str, value, *, allow_int: bool = False
) -> Optional[float]:
    if value is None:
        return None
    return require_float(name, value, allow_int=allow_int)


def require_str(name: str, value) -> str:
    if not isinstance(value, str):
        _raise_type_error(name, "str", value)
    return value


def require_optional_str(name: str, value) -> Optional[str]:
    if value is None:
        return None
    return require_str(name, value)


def require_bool(name: str, value) -> bool:
    if not isinstance(value, bool):
        _raise_type_error(name, "bool", value)
    return value


def require_tensor(name: str, value) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        _raise_type_error(name, "Tensor", value)
    return value


def require_optional_tensor(name: str, value) -> Optional[torch.Tensor]:
    if value is None:
        return None
    return require_tensor(name, value)


def require_list_tensor(name: str, value) -> List[torch.Tensor]:
    if not isinstance(value, (list, tuple)):
        _raise_type_error(name, "list[Tensor]", value)
    for i, item in enumerate(value):
        if not isinstance(item, torch.Tensor):
            _raise_type_error(f"{name}[{i}]", "Tensor", item)
    return list(value)


def require_optional_list_tensor(name: str, value) -> Optional[List[torch.Tensor]]:
    if value is None:
        return None
    return require_list_tensor(name, value)


def require_dtype(name: str, value) -> torch.dtype:
    """Require a torch.dtype object (not a Tensor's .dtype field)."""
    if not isinstance(value, torch.dtype):
        _raise_type_error(name, "dtype", value)
    return value


def require_optional_dtype(name: str, value) -> Optional[torch.dtype]:
    if value is None:
        return None
    return require_dtype(name, value)


def check_value(name: str, value, annotation) -> None:
    """Check ``value`` against a typing annotation; raise ``ArgTypeError`` with ``name``."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if type(None) in args:
            if value is None:
                return
            if len(non_none) == 1:
                check_value(name, value, non_none[0])
                return
            last_error = None
            for alt in non_none:
                try:
                    check_value(name, value, alt)
                    return
                except TypeError as e:
                    last_error = e
            if last_error is not None:
                raise last_error
            _raise_type_error(name, _annotation_label(annotation), value)
            return
        last_error = None
        for alt in args:
            try:
                check_value(name, value, alt)
                return
            except TypeError as e:
                last_error = e
        if last_error is not None:
            raise last_error
        _raise_type_error(name, _annotation_label(annotation), value)
        return

    if origin in (list, List):
        if not isinstance(value, (list, tuple)):
            _raise_type_error(name, _annotation_label(annotation), value)
        elem_ann = args[0] if args else Any
        for i, item in enumerate(value):
            check_value(f"{name}[{i}]", item, elem_ann)
        return

    if origin in (dict, Dict):
        if not isinstance(value, dict):
            _raise_type_error(name, _annotation_label(annotation), value)
        key_ann = args[0] if len(args) >= 1 else Any
        val_ann = args[1] if len(args) >= 2 else Any
        for k, v in value.items():
            check_value(f"{name} key", k, key_ann)
            check_value(f"{name}[{k!r}]", v, val_ann)
        return

    if annotation is int:
        require_int(name, value)
        return
    if annotation is float:
        require_float(name, value, allow_int=True)
        return
    if annotation is str:
        require_str(name, value)
        return
    if annotation is bool:
        require_bool(name, value)
        return
    if annotation is torch.Tensor:
        require_tensor(name, value)
        return
    if annotation is torch.dtype:
        require_dtype(name, value)
        return

    if isinstance(annotation, type):
        if not isinstance(value, annotation):
            _raise_type_error(name, _annotation_label(annotation), value)
        return


def check_args(
    func: Optional[Callable] = None,
    *,
    skip: frozenset | set | tuple = (),
) -> Callable:
    """Decorator: validate bound arguments against the function's type hints.

    Unannotated parameters and ``Any`` are skipped. ``self`` / ``cls`` are skipped.
    Names in ``skip`` are skipped.
    """
    skip_names = frozenset(skip) | frozenset({"self", "cls"})

    def decorator(fn: Callable) -> Callable:
        sig = inspect.signature(fn)
        hints_holder: Dict[str, Any] = {}

        @functools.wraps(fn)
        def wrapper(*a, **kw):
            if not hints_holder:
                hints_holder.update(get_type_hints(fn))
            bound = sig.bind(*a, **kw)
            bound.apply_defaults()
            param_names = [
                name
                for name, param in sig.parameters.items()
                if name not in skip_names
                and param.kind
                not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            ]
            for name, value in bound.arguments.items():
                if name in skip_names:
                    continue
                annotation = hints_holder.get(name)
                if annotation is None:
                    continue
                try:
                    check_value(name, value, annotation)
                except ArgTypeError as err:
                    raise TypeError(
                        _format_type_error(
                            fn.__qualname__,
                            param_names,
                            hints_holder,
                            bound.arguments,
                            err,
                        )
                    ) from None
            return fn(*a, **kw)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


# ----- wrap_op_module -----

_PYBIND_TYPE_MAP = {
    "int": int,
    "str": str,
    "float": float,
    "bool": bool,
    "torch.Tensor": torch.Tensor,
    "Tensor": torch.Tensor,
    "torch.dtype": torch.dtype,
    "dtype": torch.dtype,
}


def _resolve_pybind_type(type_str: str):
    text = type_str.strip()
    if text.startswith("Optional[") and text.endswith("]"):
        return Optional[_resolve_pybind_type(text[len("Optional[") : -1])]
    if text.startswith("list[") and text.endswith("]"):
        return List[_resolve_pybind_type(text[len("list[") : -1])]
    if text.startswith("List[") and text.endswith("]"):
        return List[_resolve_pybind_type(text[len("List[") : -1])]
    return _PYBIND_TYPE_MAP.get(text, Any)


def _split_top_level(text: str, sep: str = ",") -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    for ch in text:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth = max(0, depth - 1)
        if ch == sep and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            continue
        buf.append(ch)
    part = "".join(buf).strip()
    if part:
        parts.append(part)
    return parts


@dataclass(frozen=True)
class _PybindParam:
    name: str
    annotation: Any


def _parse_pybind_doc(
    doc: str, fallback_name: str
) -> Optional[Tuple[str, Tuple[_PybindParam, ...]]]:
    """Parse ``name(arg: type, ...) -> ret`` from a pybind ``__doc__`` first line."""
    if not doc:
        return None
    first = doc.strip().splitlines()[0].strip()
    matched = re.match(r"^([A-Za-z_]\w*)\((.*)\)\s*(?:->\s*.+)?\s*$", first)
    if not matched:
        return None
    api_name = matched.group(1) or fallback_name
    params_blob = matched.group(2).strip()
    if not params_blob:
        return api_name, tuple()

    params: List[_PybindParam] = []
    for part in _split_top_level(params_blob):
        pm = re.match(r"^(\*{0,2})([A-Za-z_]\w*)\s*:\s*(.+)$", part)
        if not pm:
            return None
        name = pm.group(2)
        type_and_default = pm.group(3).strip()
        type_str = _split_top_level(type_and_default, "=")[0].strip()
        params.append(_PybindParam(name, _resolve_pybind_type(type_str)))
    return api_name, tuple(params)


def _bind_pybind_args(
    params: Tuple[_PybindParam, ...], args: tuple, kwargs: dict
) -> Dict[str, Any]:
    bound: Dict[str, Any] = {}
    names = [p.name for p in params]
    for i, value in enumerate(args):
        if i < len(names):
            bound[names[i]] = value
    bound.update(kwargs)
    return bound


def _wrap_op_callable(fn: Callable, fallback_name: str) -> Callable:
    # Unparseable / missing __doc__ -> forward unchanged (no type check).
    parsed = _parse_pybind_doc(getattr(fn, "__doc__", "") or "", fallback_name)
    if parsed is None:
        return fn
    api_name, params = parsed
    param_names = [p.name for p in params]
    annotations = {p.name: p.annotation for p in params}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        bound = _bind_pybind_args(params, args, kwargs)
        for p in params:
            # Unknown doc types resolve to Any and are skipped.
            if p.name not in bound or p.annotation is Any:
                continue
            try:
                check_value(p.name, bound[p.name], p.annotation)
            except ArgTypeError as err:
                raise TypeError(
                    _format_type_error(api_name, param_names, annotations, bound, err)
                ) from None
        return fn(*args, **kwargs)

    return wrapper


class _CheckedOpModule:
    """Lazy proxy: wrap callables with arg type checks, forward other attrs."""

    def __init__(self, module):
        object.__setattr__(self, "_module", module)
        object.__setattr__(self, "_cache", {})

    def __getattr__(self, name: str):
        cache = object.__getattribute__(self, "_cache")
        if name in cache:
            return cache[name]
        module = object.__getattribute__(self, "_module")
        attr = getattr(module, name)
        if callable(attr):
            attr = _wrap_op_callable(attr, name)
        cache[name] = attr
        return attr


def wrap_op_module(module):
    """Wrap a loaded extension module with Python arg type checks.

    Signature source is each callable's pybind ``__doc__`` (names + types).
    Only clear type mismatches are rejected; ambiguous values are forwarded so
    pybind remains the authority. Missing/unparseable docs (or ``argN`` names)
    pass through unchanged. Called from ``OpBuilder.load()``; ops should just
    use ``_builder.load()``.
    """
    existing = getattr(module, "__arg_check_proxy__", None)
    if existing is not None:
        return existing
    proxy = _CheckedOpModule(module)
    try:
        # Extension modules may reject arbitrary attributes.
        setattr(module, "__arg_check_proxy__", proxy)
    except Exception:
        pass
    return proxy

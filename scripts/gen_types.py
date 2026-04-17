"""
Generate TypeScript types from the Pydantic request/response models.

Eliminates hand-maintained duplicates between ``src/api/routers/*.py`` and
``frontend/src/lib/api/client.ts``. Call this any time a request/response
model changes and re-run the frontend type-check:

    python scripts/gen_types.py
    cd frontend && npm run check

Output lands at ``frontend/src/lib/types/generated.ts``. That file is
tracked in git so CI picks up drift via a diff check.

The generator deliberately handles a small, predictable subset of Pydantic
field types — it's a contract tool, not a general pydantic-to-ts converter.
Extend ``_type_to_ts`` when you introduce a new kind of field.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
import typing
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

# Ensure repo root is importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# ── Which pydantic models to export ─────────────────────────────────────────


@dataclass(frozen=True)
class Export:
    """One pydantic class to emit as a TypeScript interface."""

    module: str           # dotted import path, e.g. "src.api.routers.chat"
    name: str             # class name inside that module
    ts_name: str | None = None  # override the generated TS name if needed

    @property
    def interface_name(self) -> str:
        return self.ts_name or self.name


EXPORTS: list[Export] = [
    Export("src.api.routers.auth", "LoginRequest"),
    Export("src.api.routers.sessions", "CreateSessionRequest"),
    Export("src.api.routers.sessions", "RenameRequest"),
    Export("src.api.routers.chat", "ChatRequest"),
    Export("src.api.routers.documents", "SelectionUpdate"),
    Export("src.api.routers.documents", "DeleteDocsRequest"),
    Export("src.api.routers.settings", "SettingsUpdate"),
    Export("src.api.routers.settings", "OCRSettings"),
    Export("src.api.routers.settings", "OllamaKeyUpdate"),
    Export("src.api.routers.scores", "SaveScoresRequest"),
    Export("src.api.routers.scores", "SlopeToggle"),
    Export("src.api.routers.scores", "SlopeParams"),
]


# ── Python → TypeScript mapping ─────────────────────────────────────────────


def _type_to_ts(tp: object) -> str:
    """Convert a single Python type annotation into a TypeScript type string.

    Handles: primitives, Optional, list/dict, Union, Any. Anything else
    falls through to ``unknown`` with a comment so the output is valid but
    the maintainer can tell the generator needs extending.
    """
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if tp in (str,):
        return "string"
    if tp in (int, float):
        return "number"
    if tp is bool:
        return "boolean"
    if tp is type(None):
        return "null"
    if tp is typing.Any:
        return "any"

    # Unions: both typing.Union[...] and PEP 604 ``int | None`` syntax.
    # The pipe syntax has origin = types.UnionType, not typing.Union.
    if origin is typing.Union or origin is types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and len(args) == 2:
            return f"{_type_to_ts(non_none[0])} | null"
        return " | ".join(_type_to_ts(a) for a in args)

    if origin in (list, typing.List):
        inner = args[0] if args else typing.Any
        return f"Array<{_type_to_ts(inner)}>"

    if origin in (dict, typing.Dict):
        if len(args) == 2:
            return f"Record<{_type_to_ts(args[0])}, {_type_to_ts(args[1])}>"
        return "Record<string, unknown>"

    if isinstance(tp, type) and issubclass(tp, BaseModel):
        return tp.__name__

    # Fallback — keeps the emitted file valid TS even if we see something new.
    return f"unknown /* unmapped: {tp!r} */"


def _render_model(model: type[BaseModel], name: str) -> str:
    """Emit one ``export interface ... { ... }`` block for ``model``."""
    lines = [f"export interface {name} {{"]
    for field_name, field_info in model.model_fields.items():
        ts_type = _type_to_ts(field_info.annotation)
        optional = "?" if not field_info.is_required() else ""
        lines.append(f"\t{field_name}{optional}: {ts_type};")
    lines.append("}")
    return "\n".join(lines)


HEADER = """\
// ──────────────────────────────────────────────────────────────────────────
// GENERATED FILE — DO NOT EDIT
// Regenerate with:  python scripts/gen_types.py
//
// Source: Pydantic request/response models in src/api/routers/*.py
// ──────────────────────────────────────────────────────────────────────────
"""


def _render_all() -> str:
    out = [HEADER]
    for exp in EXPORTS:
        mod = importlib.import_module(exp.module)
        cls = getattr(mod, exp.name)
        if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
            raise TypeError(f"{exp.module}.{exp.name} is not a pydantic BaseModel")
        out.append(_render_model(cls, exp.interface_name))
        out.append("")  # blank line between blocks
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    target = _REPO_ROOT / "frontend" / "src" / "lib" / "types" / "generated.ts"
    target.parent.mkdir(parents=True, exist_ok=True)
    text = _render_all()
    # Don't touch the file if the output is identical — preserves mtime for
    # tooling that tracks "last modified" on generated files.
    if target.exists() and target.read_text() == text:
        print(f"{target}: unchanged")
        return 0
    target.write_text(text)
    print(f"Wrote {target}  ({len(text.splitlines())} lines, {len(EXPORTS)} interfaces)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

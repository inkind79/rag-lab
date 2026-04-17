"""Generator contract tests.

Two axes:

1. ``_type_to_ts`` produces the expected TS for every Python type we
   actually use across the request models. Extending the generator
   (new field type → new branch) means extending this test.

2. Committed ``frontend/src/lib/types/generated.ts`` matches what the
   generator would emit *right now*. This is the drift check — if a
   backend model changes and the generated file isn't regenerated,
   CI fails here.
"""

from __future__ import annotations

import sys
import typing
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from pydantic import BaseModel

import gen_types  # noqa: E402


# ── _type_to_ts ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "annotation,expected",
    [
        (str, "string"),
        (int, "number"),
        (float, "number"),
        (bool, "boolean"),
        (typing.Any, "any"),
        (typing.Optional[str], "string | null"),
        (typing.Optional[int], "number | null"),
        (typing.List[str], "Array<string>"),
        (typing.List[int], "Array<number>"),
        (typing.Optional[typing.List[str]], "Array<string> | null"),
        (typing.Dict[str, typing.Any], "Record<string, any>"),
        (typing.Dict[str, str], "Record<string, string>"),
    ],
)
def test_type_to_ts(annotation, expected):
    assert gen_types._type_to_ts(annotation) == expected


def test_type_to_ts_nested_optional_list():
    """Optional[List[Optional[int]]] → Array<number | null> | null."""
    annotation = typing.Optional[typing.List[typing.Optional[int]]]
    got = gen_types._type_to_ts(annotation)
    # The generator produces Array<X> | null; inner Optional becomes X | null
    assert got == "Array<number | null> | null"


def test_type_to_ts_unmapped_falls_through():
    """Unknown types produce valid TS (unknown) with a hint, not a crash."""

    class WeirdThing:
        pass

    got = gen_types._type_to_ts(WeirdThing)
    assert got.startswith("unknown")


# ── render_model ─────────────────────────────────────────────────────────────


def test_render_model_required_and_optional():
    class Sample(BaseModel):
        req_name: str
        opt_count: int | None = None
        flag: bool = False

    out = gen_types._render_model(Sample, "Sample")
    assert "export interface Sample {" in out
    assert "\treq_name: string;" in out
    # `opt_count: int | None = None` is optional in pydantic (has default).
    assert "\topt_count?: number | null;" in out
    # `flag: bool = False` is optional in pydantic (has default).
    assert "\tflag?: boolean;" in out


# ── Drift: committed file matches generator output ───────────────────────────


def test_generated_file_up_to_date():
    """If this fails, run `python scripts/gen_types.py` and commit the diff."""
    target = _REPO_ROOT / "frontend" / "src" / "lib" / "types" / "generated.ts"
    assert target.exists(), (
        f"Generated types file is missing at {target}. "
        "Run `python scripts/gen_types.py` to create it."
    )

    expected = gen_types._render_all()
    actual = target.read_text()

    if actual != expected:
        pytest.fail(
            "frontend/src/lib/types/generated.ts is stale.\n"
            "A pydantic request/response model changed without regenerating.\n"
            "Fix:  python scripts/gen_types.py   (then commit the diff)"
        )

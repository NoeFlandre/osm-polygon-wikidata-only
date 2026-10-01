"""Tests build objects through their constructors, never around them."""

from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any

TESTS = Path(__file__).resolve().parents[1]
SEAM_ALLOWLIST = Path(__file__).with_name("private_test_seams.json")


def test_no_test_builds_an_object_with_dunder_new() -> None:
    """``Cls.__new__(Cls)`` skips ``__init__``, so the object may not be constructible for real."""
    offenders = [
        f"{path.relative_to(TESTS).as_posix()}:{node.lineno}"
        for path in sorted(TESTS.rglob("*.py"))
        if path != Path(__file__).resolve()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "__new__"
    ]

    assert offenders == []


def _private_patch_target(node: ast.Call) -> str | None:
    """Return an explicitly named private target from a patch/setter call."""
    function = node.func
    name = function.attr if isinstance(function, ast.Attribute) else None
    private_name: str | None = None

    if name in {"setattr", "delattr"} and len(node.args) >= 2:
        attribute = node.args[1]
        if isinstance(attribute, ast.Constant) and isinstance(attribute.value, str):
            private_name = attribute.value
    elif name == "object" and len(node.args) >= 2:
        # unittest.mock.patch.object, monkeypatch/mocker.patch.object
        parent = function.value if isinstance(function, ast.Attribute) else None
        if isinstance(parent, ast.Attribute) and parent.attr == "patch":
            attribute = node.args[1]
            if isinstance(attribute, ast.Constant) and isinstance(attribute.value, str):
                private_name = attribute.value
    elif name == "patch" and node.args:
        # Dotted target strings can name a private module-level function.
        target = node.args[0]
        if isinstance(target, ast.Constant) and isinstance(target.value, str):
            private_name = target.value.rsplit(".", maxsplit=1)[-1]

    if (
        private_name
        and private_name.startswith("_")
        and not (private_name.startswith("__") and private_name.endswith("__"))
    ):
        return private_name
    return None


def _patch_counts() -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = {}
    for path in sorted(TESTS.rglob("*.py")):
        relative = path.relative_to(TESTS).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                target = _private_patch_target(node)
                if target is not None:
                    counts.setdefault(relative, Counter())[target] += 1
    return counts


def _is_fixture_owned_attribute(node: ast.Attribute, parents: dict[ast.AST, ast.AST]) -> bool:
    """Allow ``self._state`` only inside a class method that owns that fake state."""
    if not isinstance(node.value, ast.Name) or node.value.id != "self":
        return False

    ancestor = parents.get(node)
    method: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    while ancestor is not None:
        if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method = ancestor
            break
        if isinstance(ancestor, ast.ClassDef):
            return False
        ancestor = parents.get(ancestor)

    if method is None or not method.args.args or method.args.args[0].arg != "self":
        return False

    ancestor = parents.get(method)
    while ancestor is not None:
        if isinstance(ancestor, ast.ClassDef):
            return True
        if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
        ancestor = parents.get(ancestor)
    return False


def test_private_test_seams_stay_within_documented_allowlist() -> None:
    """The private-patch allowlist matches current use; new coverage uses public seams."""
    policy: dict[str, Any] = json.loads(SEAM_ALLOWLIST.read_text(encoding="utf-8"))
    allowed: dict[str, dict[str, int]] = policy["targets"]
    actual = _patch_counts()
    offenders = [
        f"{path}:{name} ({count} calls; allowed {allowed.get(path, {}).get(name, 0)})"
        for path, names in actual.items()
        for name, count in names.items()
        if count > allowed.get(path, {}).get(name, 0)
    ]

    assert policy["policy"]
    assert offenders == []
    assert allowed == {
        path: dict(sorted(counts.items())) for path, counts in sorted(actual.items())
    }, "private patch allowlist must contain exactly the current seams and call counts"


def test_tests_do_not_write_private_attributes_on_other_objects() -> None:
    """Fake classes may own their internal state; tests cannot assign another object's private state."""
    offenders: list[str] = []
    for path in sorted(TESTS.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = {
            child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for child in ast.walk(target):
                    if not isinstance(child, ast.Attribute) or not child.attr.startswith("_"):
                        continue
                    if child.attr.startswith("__") and child.attr.endswith("__"):
                        continue
                    if _is_fixture_owned_attribute(child, parents):
                        continue
                    offenders.append(f"{path.relative_to(TESTS).as_posix()}:{node.lineno}")

    assert offenders == []

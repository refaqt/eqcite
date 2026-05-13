"""
Restricted expression evaluation for experiment variables.

Expressions may use +, -, *, /, unary +/-, parentheses, and sin(), cos().
Names must refer to existing mapping variables (v1: no references to other
expression variables). Evaluation is element-wise on NumPy arrays.
"""

from __future__ import annotations

import ast
import math
from typing import Dict, FrozenSet, Set

import numpy as np


def _validate_expr_ast(node: ast.AST) -> None:
    if isinstance(node, ast.Expression):
        _validate_expr_ast(node.body)
        return
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            raise ValueError(f"Operator not allowed: {type(node.op).__name__}")
        _validate_expr_ast(node.left)
        _validate_expr_ast(node.right)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise ValueError(f"Unary operator not allowed: {type(node.op).__name__}")
        _validate_expr_ast(node.operand)
        return
    if isinstance(node, ast.Name):
        if not isinstance(node.ctx, ast.Load):
            raise ValueError("Only load context for names is allowed")
        return
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric constants are allowed")
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple sin/cos calls are allowed")
        fn = node.func.id
        if fn not in ("sin", "cos"):
            raise ValueError(f"Function not allowed: {fn}")
        if len(node.args) != 1 or node.keywords:
            raise ValueError("sin/cos require exactly one positional argument")
        _validate_expr_ast(node.args[0])
        return
    raise ValueError(f"Syntax not allowed in expression: {type(node).__name__}")


def referenced_mapping_names(expr: str) -> FrozenSet[str]:
    """Return mapping variable names referenced in expr (after validation)."""
    expr = (expr or "").strip()
    if not expr:
        return frozenset()
    tree = ast.parse(expr, mode="eval")
    _validate_expr_ast(tree)

    out: Set[str] = set()

    def walk(n: ast.AST) -> None:
        if isinstance(n, ast.Expression):
            walk(n.body)
        elif isinstance(n, ast.BinOp):
            walk(n.left)
            walk(n.right)
        elif isinstance(n, ast.UnaryOp):
            walk(n.operand)
        elif isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Call):
            for a in n.args:
                walk(a)

    walk(tree)
    return frozenset(out)


def evaluate_expression(expr: str, env: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Evaluate expr using env name -> 1D ndarray (same length).

    Raises:
        ValueError: invalid syntax or missing names.
    """
    expr = (expr or "").strip()
    if not expr:
        raise ValueError("Empty expression")
    names = referenced_mapping_names(expr)
    for n in names:
        if n not in env:
            raise ValueError(f"Unknown variable in expression: {n!r}")
    if not env:
        raise ValueError("env is empty")
    ref = next(iter(env.values()))
    length = len(np.asarray(ref))

    safe_dict = {
        "__builtins__": {},
        "sin": np.sin,
        "cos": np.cos,
        "pi": math.pi,
    }
    for k, v in env.items():
        if k in safe_dict:
            raise ValueError(f"Reserved name in env: {k!r}")
        arr = np.asarray(v, dtype=float)
        if arr.shape != (length,):
            raise ValueError(f"Variable {k!r} must be 1D of length {length}, got {arr.shape}")
        safe_dict[k] = arr

    tree = ast.parse(expr, mode="eval")
    _validate_expr_ast(tree)
    code = compile(tree, "<expr>", "eval")
    return np.asarray(eval(code, safe_dict), dtype=float)

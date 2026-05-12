"""Tests for restricted expression evaluation."""

from __future__ import annotations

import unittest

import numpy as np

from src import expression_eval


class TestExpressionEval(unittest.TestCase):
    def test_basic_arithmetic(self) -> None:
        t = np.linspace(0, 1, 16, endpoint=False)
        env = {"a": np.sin(t), "b": np.ones_like(t)}
        y = expression_eval.evaluate_expression("a + 2 * b", env)
        self.assertEqual(y.shape, (16,))
        np.testing.assert_allclose(y, np.sin(t) + 2.0)

    def test_sin_cos(self) -> None:
        t = np.linspace(0, 1, 32, endpoint=False)
        env = {"x": t}
        y = expression_eval.evaluate_expression("sin(x) + cos(x)", env)
        self.assertEqual(y.shape, (32,))

    def test_reject_unsafe(self) -> None:
        t = np.linspace(0, 1, 8, endpoint=False)
        env = {"a": t}
        with self.assertRaises(ValueError):
            expression_eval.evaluate_expression("__import__('os').system('x')", env)

    def test_referenced_names(self) -> None:
        names = expression_eval.referenced_mapping_names("a + b / 2")
        self.assertEqual(set(names), {"a", "b"})


if __name__ == "__main__":
    unittest.main()

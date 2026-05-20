from __future__ import annotations

import unittest

from PySide6.QtWidgets import QApplication

from src.gui.pages.frf import FrfFilteringPage


class FrfFilteringTrendSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_build_filtering_cfg_includes_force_and_response_trend_times(self) -> None:
        page = FrfFilteringPage()
        cfg = page.build_filtering_cfg()
        self.assertIn("trend_estimation", cfg)
        trend = cfg["trend_estimation"]
        self.assertIn("force", trend)
        self.assertIn("response", trend)
        self.assertIn("impact_start_s", trend["force"])
        self.assertIn("ringdown_end_s", trend["force"])
        self.assertIn("impact_start_s", trend["response"])
        self.assertIn("ringdown_end_s", trend["response"])

    def test_apply_filtering_cfg_restores_trend_times(self) -> None:
        page = FrfFilteringPage()
        page.apply_filtering_cfg(
            {
                "trend_estimation": {
                    "force": {"impact_start_s": 0.01, "ringdown_end_s": 0.12},
                    "response": {"impact_start_s": 0.015, "ringdown_end_s": 0.2},
                },
                "u_chain": [{"name": "wiener", "enabled": True}],
                "resp_chain": [{"name": "wiener", "enabled": False}],
                "params": {"wiener": {"mysize": 51}},
            }
        )
        cfg = page.build_filtering_cfg()
        self.assertEqual(cfg["trend_estimation"]["force"]["impact_start_s"], 0.01)
        self.assertEqual(cfg["trend_estimation"]["force"]["ringdown_end_s"], 0.12)
        self.assertEqual(cfg["trend_estimation"]["response"]["impact_start_s"], 0.015)
        self.assertEqual(cfg["trend_estimation"]["response"]["ringdown_end_s"], 0.2)


if __name__ == "__main__":
    unittest.main()

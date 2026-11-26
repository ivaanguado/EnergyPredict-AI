import unittest
import pandas as pd

from energypredict.evaluation.metrics import (
    calculate_metrics,
)


class TestMetrics(unittest.TestCase):

    def test_known_errors(self):

        predictions = pd.DataFrame({
            "actual_demand_mw": [30000, 32000, 34000],
            "predicted_demand_mw": [30300, 31600, 34500],
        })

        metrics = calculate_metrics(predictions)

        self.assertEqual(metrics["count"], 3)
        self.assertAlmostEqual(metrics["mae"], 400.0)
        self.assertAlmostEqual(metrics["rmse"], 408.24829046)
        self.assertAlmostEqual(metrics["mape"], 1.24019608)

    def test_perfect_prediction(self):

        predictions = pd.DataFrame({
            "actual_demand_mw": [30000, 32000],
            "predicted_demand_mw": [30000, 32000],
        })

        metrics = calculate_metrics(predictions)

        self.assertEqual(metrics["count"], 2)
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["mape"], 0.0)

    def test_pending_predictions(self):

        predictions = pd.DataFrame({
            "actual_demand_mw": [None, None],
            "predicted_demand_mw": [30000, 32000],
        })

        metrics = calculate_metrics(predictions)

        self.assertEqual(metrics["count"], 0)
        self.assertIsNone(metrics["mae"])
        self.assertIsNone(metrics["rmse"])
        self.assertIsNone(metrics["mape"])

    def test_mixed_predictions(self):

        predictions = pd.DataFrame({
            "actual_demand_mw": [30000, None],
            "predicted_demand_mw": [30300, 32000],
        })

        metrics = calculate_metrics(predictions)

        self.assertEqual(metrics["count"], 1)
        self.assertAlmostEqual(metrics["mae"], 300.0)
        self.assertAlmostEqual(metrics["rmse"], 300.0)
        self.assertAlmostEqual(metrics["mape"], 1.0)


if __name__ == "__main__":
    unittest.main()
import unittest

import numpy as np
import pandas as pd

from predict import (
    get_recent_history,
    get_horizon_limits,
    predict_next_hour,
)


METHODS = [
    "recursive",
    "direct",
    "hybrid",
]


class TestForecastConsistency(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.history = get_recent_history()

        cls.generation_hour = (
            pd.Timestamp.now(tz="UTC").floor("h")
        )

    def check_method(self, method):

        limits = get_horizon_limits(
            self.history,
            method,
        )

        max_horizon = limits["max_horizon"]

        if max_horizon < 2:
            self.skipTest(
                "Horizontes insuficientes para comparar."
            )

        full_result = predict_next_hour(
            method=method,
            horizon=max_horizon,
            return_data=True,
            history=self.history,
        )

        full_forecast = full_result["forecast"]

        for horizon in range(1, max_horizon):

            with self.subTest(
                method=method,
                horizon=horizon,
            ):

                if (
                    pd.Timestamp.now(tz="UTC").floor("h")
                    != self.generation_hour
                ):
                    self.skipTest(
                        "La hora ha cambiado durante la prueba."
                    )

                individual_result = predict_next_hour(
                    method=method,
                    horizon=horizon,
                    return_data=True,
                    history=self.history,
                )

                target = individual_result["target_hour"]

                expected_target = (
                    self.generation_hour
                    + pd.Timedelta(hours=horizon)
                )

                self.assertEqual(
                    target,
                    expected_target,
                )

                full_prediction = full_forecast.loc[
                    full_forecast["datetime_utc"] == target,
                    "predicted_demand_mw",
                ]

                self.assertEqual(
                    len(full_prediction),
                    1,
                )

                individual_prediction = (
                    individual_result["forecast"]
                    .iloc[-1]["predicted_demand_mw"]
                )

                self.assertTrue(
                    np.isclose(
                        full_prediction.iloc[0],
                        individual_prediction,
                        rtol=0,
                        atol=0.000001,
                    ),
                    msg=(
                        f"Predicciones diferentes: "
                        f"{method}, horizonte {horizon}"
                    ),
                )

                print(
                    f"\nVALIDADO: {method} | "
                    f"+{horizon} h | "
                    f"{individual_prediction:.2f} MW"
                )

    def test_recursive(self):
        self.check_method("recursive")

    def test_direct(self):
        self.check_method("direct")

    def test_hybrid(self):
        self.check_method("hybrid")


if __name__ == "__main__":
    unittest.main(verbosity=2)
import numpy as np
import pandas as pd

from energypredict.storage.prediction_repository import (
    get_predictions,
)


def calculate_metrics(predictions):
    """Calcula las métricas de las predicciones evaluadas."""

    evaluated = predictions.dropna(
        subset=[
            "actual_demand_mw",
            "predicted_demand_mw",
        ]
    ).copy()

    if evaluated.empty:
        return {
            "count": 0,
            "mae": None,
            "rmse": None,
            "mape": None,
        }

    actual = evaluated["actual_demand_mw"].to_numpy(
        dtype=float
    )

    predicted = evaluated["predicted_demand_mw"].to_numpy(
        dtype=float
    )

    errors = actual - predicted

    mae = np.mean(np.abs(errors))

    rmse = np.sqrt(
        np.mean(np.square(errors))
    )

    valid = actual != 0

    mape = (
        np.mean(
            np.abs(errors[valid] / actual[valid])
        ) * 100
        if valid.any()
        else None
    )

    return {
        "count": len(evaluated),
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape) if mape is not None else None,
    }


def get_global_metrics():
    """Calcula el rendimiento global del sistema."""

    predictions = get_predictions()

    return calculate_metrics(predictions)


def get_metrics_by_method():
    """Calcula el rendimiento de cada método."""

    predictions = get_predictions()

    results = []

    for method in ["recursive", "direct", "hybrid"]:

        subset = predictions[
            predictions["method"] == method
        ]

        metrics = calculate_metrics(subset)

        results.append({
            "method": method,
            **metrics,
        })

    return pd.DataFrame(results)


def get_metrics_by_horizon():
    """Calcula el rendimiento según el horizonte."""

    predictions = get_predictions()

    results = []

    for horizon in range(1, 7):

        subset = predictions[
            predictions["horizon_hours"] == horizon
        ]

        metrics = calculate_metrics(subset)

        results.append({
            "horizon_hours": horizon,
            **metrics,
        })

    return pd.DataFrame(results)


def get_metrics_by_method_and_horizon():
    """Calcula el rendimiento por método y horizonte."""

    predictions = get_predictions()

    results = []

    for method in ["recursive", "direct", "hybrid"]:

        for horizon in range(1, 7):

            subset = predictions[
                (predictions["method"] == method)
                & (predictions["horizon_hours"] == horizon)
            ]

            metrics = calculate_metrics(subset)

            results.append({
                "method": method,
                "horizon_hours": horizon,
                **metrics,
            })

    return pd.DataFrame(results)


if __name__ == "__main__":

    print("\nMÉTRICAS GLOBALES")
    print("-" * 50)
    print(get_global_metrics())

    print("\nMÉTRICAS POR MÉTODO")
    print("-" * 50)
    print(get_metrics_by_method().to_string(index=False))

    print("\nMÉTRICAS POR HORIZONTE")
    print("-" * 50)
    print(get_metrics_by_horizon().to_string(index=False))

    print("\nMÉTRICAS POR MÉTODO Y HORIZONTE")
    print("-" * 50)
    print(
        get_metrics_by_method_and_horizon()
        .to_string(index=False)
    )
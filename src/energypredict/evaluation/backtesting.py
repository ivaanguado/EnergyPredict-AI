
import pandas as pd

from energypredict.prediction.recursive_forecast import (
    forecast_recursive,
)


def evaluate_recursive_forecast(
    model,
    feature_columns,
    hourly_data,
    max_horizon=6,
    min_history=169,
):
    """
    Evalúa predicciones recursivas realizadas
    desde diferentes momentos históricos.

    No utiliza valores futuros para construir
    las características de cada predicción.
    """

    data = hourly_data.sort_values(
        "datetime_utc"
    ).reset_index(drop=True)

    if max_horizon < 1:
        raise ValueError("max_horizon debe ser positivo.")

    if min_history < 169:
        raise ValueError(
            "Se necesitan al menos 169 horas de historial."
        )

    if len(data) < min_history + max_horizon:
        raise ValueError(
            "No hay suficientes datos para el backtesting."
        )

    if data["datetime_utc"].duplicated().any():
        raise ValueError("Hay fechas duplicadas.")

    expected = pd.date_range(
        start=data["datetime_utc"].iloc[0],
        periods=len(data),
        freq="h",
    )

    if not pd.DatetimeIndex(
        data["datetime_utc"]
    ).equals(expected):
        raise ValueError(
            "Los datos horarios no son consecutivos."
        )

    results = []

    # Cada iteración simula un momento del pasado
    for end_idx in range(
        min_history,
        len(data) - max_horizon + 1,
    ):
        history = data.iloc[:end_idx][
            ["datetime_utc", "demand_mw"]
        ].copy()

        forecast = forecast_recursive(
            model=model,
            feature_columns=feature_columns,
            history=history,
            steps=max_horizon,
        )

        actual = data.iloc[
            end_idx:end_idx + max_horizon
        ].reset_index(drop=True)

        for horizon in range(1, max_horizon + 1):
            predicted = forecast.iloc[horizon - 1]
            observed = actual.iloc[horizon - 1]

            if (
                predicted["datetime_utc"]
                != observed["datetime_utc"]
            ):
                raise ValueError(
                    "Las fechas de predicción "
                    "y observación no coinciden."
                )

            # No evaluar contra valores interpolados
            if observed["contains_interpolation"]:
                continue

            real_value = float(observed["demand_mw"])
            predicted_value = float(
                predicted["predicted_demand_mw"]
            )

            absolute_error = abs(
                real_value - predicted_value
            )

            percentage_error = (
                absolute_error / real_value * 100
            )

            results.append({
                "origin_datetime_utc": (
                    history["datetime_utc"].iloc[-1]
                ),
                "target_datetime_utc": (
                    predicted["datetime_utc"]
                ),
                "horizon": horizon,
                "actual_mw": real_value,
                "predicted_mw": predicted_value,
                "absolute_error_mw": absolute_error,
                "percentage_error": percentage_error,
            })

    return pd.DataFrame(results)

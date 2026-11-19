
import pandas as pd

from energypredict.features.feature_builder import (
    build_features,
)

from energypredict.training.multihorizon_dataset import (
    get_horizon_features,
)


def evaluate_direct_forecast(
    models,
    feature_columns,
    hourly_data,
    max_horizon=6,
    min_history=169,
):
    """
    Evalúa modelos directos usando únicamente
    la información disponible en cada origen.

    models: diccionario {horizonte: modelo}
    """

    data = hourly_data.sort_values(
        "datetime_utc"
    ).reset_index(drop=True)

    if data["datetime_utc"].duplicated().any():
        raise ValueError("Existen fechas duplicadas.")

    if not data["datetime_utc"].diff().dropna().eq(
        pd.Timedelta(hours=1)
    ).all():
        raise ValueError("La serie debe ser horaria y consecutiva.")

    results = []

    for origin_idx in range(
        min_history - 1,
        len(data) - max_horizon,
    ):
        history = data.iloc[:origin_idx + 1][
            ["datetime_utc", "demand_mw"]
        ]

        features = build_features(history)
        last_features = features.iloc[[-1]].copy()

        origin_date = data.loc[
            origin_idx, "datetime_utc"
        ]

        last_features["origin_datetime_utc"] = (
            origin_date
        )

        for horizon in range(1, max_horizon + 1):
            target_idx = origin_idx + horizon

            if (
                "contains_interpolation" in data.columns
                and data.loc[
                    target_idx, "contains_interpolation"
                ]
            ):
                continue

            X = get_horizon_features(
                last_features,
                horizon,
                feature_columns,
            )

            if X.isna().any().any():
                raise ValueError(
                    "Existen características nulas."
                )

            prediction = float(
                models[horizon].predict(X)[0]
            )

            actual = float(
                data.loc[target_idx, "demand_mw"]
            )

            absolute_error = abs(
                actual - prediction
            )

            percentage_error = (
                absolute_error / actual * 100
            )

            results.append({
                "origin_datetime_utc": origin_date,
                "target_datetime_utc": data.loc[
                    target_idx, "datetime_utc"
                ],
                "horizon": horizon,
                "actual_mw": actual,
                "predicted_mw": prediction,
                "absolute_error_mw": absolute_error,
                "percentage_error": percentage_error,
            })

    return pd.DataFrame(results)

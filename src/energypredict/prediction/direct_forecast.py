
import pandas as pd

from energypredict.features.feature_builder import (
    build_features,
)

from energypredict.training.multihorizon_dataset import (
    get_horizon_features,
)


def forecast_direct(
    models,
    feature_columns,
    history,
    max_horizon=6,
):
    """
    Genera predicciones directas para los próximos
    horizontes sin utilizar predicciones anteriores.
    """

    if max_horizon < 1:
        raise ValueError("max_horizon debe ser positivo.")

    data = history[
        ["datetime_utc", "demand_mw"]
    ].copy()

    data = data.sort_values(
        "datetime_utc"
    ).reset_index(drop=True)

    if data["datetime_utc"].duplicated().any():
        raise ValueError("Hay fechas duplicadas.")

    if not data["datetime_utc"].diff().dropna().eq(
        pd.Timedelta(hours=1)
    ).all():
        raise ValueError(
            "El historial debe contener horas consecutivas."
        )

    features = build_features(data)

    last_features = features.iloc[[-1]].copy()

    origin_date = data["datetime_utc"].iloc[-1]

    last_features["origin_datetime_utc"] = origin_date

    results = []

    for horizon in range(1, max_horizon + 1):
        X = get_horizon_features(
            last_features,
            horizon,
            feature_columns,
        )

        if X.isna().any().any():
            raise ValueError(
                f"Características nulas en +{horizon} h."
            )

        prediction = float(
            models[horizon].predict(X)[0]
        )

        target_date = (
            origin_date
            + pd.Timedelta(hours=horizon)
        )

        results.append({
            "datetime_utc": target_date,
            "horizon": horizon,
            "predicted_demand_mw": prediction,
        })

    return pd.DataFrame(results)

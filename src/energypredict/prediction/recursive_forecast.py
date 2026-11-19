
import pandas as pd

from energypredict.features.feature_builder import build_features


def forecast_recursive(model, feature_columns, history, steps=4):
    """Predice varias horas utilizando predicciones anteriores."""

    if steps < 1:
        raise ValueError("steps debe ser mayor que cero.")

    current_history = history[
        ["datetime_utc", "demand_mw"]
    ].copy()

    current_history = current_history.sort_values(
        "datetime_utc"
    ).reset_index(drop=True)

    predictions = []

    for step in range(steps):
        features = build_features(current_history)

        next_hour = features.iloc[[-1]]
        X = next_hour[feature_columns]

        if X.isna().any().any():
            raise ValueError(
                "Hay características nulas en la predicción."
            )

        predicted_demand = float(model.predict(X)[0])

        target_date = next_hour[
            "target_datetime_utc"
        ].iloc[0]

        predictions.append({
            "datetime_utc": target_date,
            "predicted_demand_mw": predicted_demand,
        })

        new_row = pd.DataFrame({
            "datetime_utc": [target_date],
            "demand_mw": [predicted_demand],
        })

        current_history = pd.concat(
            [current_history, new_row],
            ignore_index=True,
        )

    return pd.DataFrame(predictions)

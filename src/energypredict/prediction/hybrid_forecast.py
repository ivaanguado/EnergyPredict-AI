
import pandas as pd

from energypredict.prediction.recursive_forecast import (
    forecast_recursive,
)
from energypredict.prediction.direct_forecast import (
    forecast_direct,
)


# Selección experimental basada en el backtesting.
HORIZON_METHODS = {
    1: "direct",
    2: "recursive",
    3: "direct",
    4: "direct",
    5: "direct",
    6: "direct",
}


def forecast_hybrid(
    recursive_model,
    direct_models,
    feature_columns,
    history,
    steps=6,
):
    if not 1 <= steps <= 6:
        raise ValueError(
            "El método híbrido admite entre 1 y 6 horas."
        )

    recursive = forecast_recursive(
        model=recursive_model,
        feature_columns=feature_columns,
        history=history,
        steps=steps,
    )

    direct = forecast_direct(
        models=direct_models,
        feature_columns=feature_columns,
        history=history,
        max_horizon=steps,
    )

    results = []

    for horizon in range(1, steps + 1):
        method = HORIZON_METHODS[horizon]

        source = (
            direct if method == "direct"
            else recursive
        )

        row = source.iloc[horizon - 1]

        results.append({
            "datetime_utc": row["datetime_utc"],
            "horizon": horizon,
            "predicted_demand_mw": (
                row["predicted_demand_mw"]
            ),
            "method": method,
        })

    return pd.DataFrame(results)

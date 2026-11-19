
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from energypredict.prediction.recursive_forecast import (
    forecast_recursive,
)
from energypredict.prediction.direct_forecast import (
    forecast_direct,
)
from energypredict.prediction.hybrid_forecast import (
    HORIZON_METHODS,
)


ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    ROOT / "data/processed/demand_spain_october_2026.parquet"
)

MAX_HORIZON = 6
MIN_HISTORY = 169

OCTOBER_START = pd.Timestamp(
    "2026-10-01 00:00:00",
    tz="UTC",
)

OCTOBER_END = pd.Timestamp(
    "2026-11-01 00:00:00",
    tz="UTC",
)


def load_models():
    recursive_artifact = joblib.load(
        ROOT / "models/random_forest_v1.joblib"
    )

    recursive_model = recursive_artifact["model"]
    columns = recursive_artifact["feature_columns"]

    direct_models = {}

    for horizon in range(1, MAX_HORIZON + 1):
        artifact = joblib.load(
            ROOT
            / "models/direct_final"
            / f"random_forest_h{horizon}.joblib"
        )

        if artifact["feature_columns"] != columns:
            raise ValueError(
                "Las características de los modelos "
                "no coinciden."
            )

        direct_models[horizon] = artifact["model"]

    return recursive_model, direct_models, columns


def main():
    data = pd.read_parquet(DATA_PATH)

    data["datetime_utc"] = pd.to_datetime(
        data["datetime_utc"],
        utc=True,
    )

    data = data.sort_values(
        "datetime_utc"
    ).reset_index(drop=True)

    if data["datetime_utc"].duplicated().any():
        raise ValueError("Hay fechas duplicadas.")

    if not data["datetime_utc"].diff().dropna().eq(
        pd.Timedelta(hours=1)
    ).all():
        raise ValueError(
            "El historial no tiene frecuencia horaria continua."
        )

    recursive_model, direct_models, columns = load_models()

    results = []
    skipped_history = 0
    skipped_targets = 0

    for i in range(MIN_HISTORY - 1, len(data) - MAX_HORIZON):
        origin = data.iloc[i]["datetime_utc"]

        # Evaluamos exclusivamente objetivos de octubre.
        first_target = origin + pd.Timedelta(hours=1)
        last_target = origin + pd.Timedelta(
            hours=MAX_HORIZON
        )

        if (
            first_target < OCTOBER_START
            or last_target >= OCTOBER_END
        ):
            continue

        history = data.iloc[
            i - MIN_HISTORY + 1:i + 1
        ].copy()

        # Evitar características basadas en datos ausentes
        # o interpolados.
        if (
            history["demand_mw"].isna().any()
            or not history["is_valid"].all()
        ):
            skipped_history += 1
            continue

        targets = data.iloc[
            i + 1:i + 1 + MAX_HORIZON
        ].copy()

        valid_targets = (
            targets["is_valid"]
            & ~targets["contains_interpolation"]
            & targets["demand_mw"].notna()
        )

        if not valid_targets.all():
            skipped_targets += 1
            continue

        recursive = forecast_recursive(
            model=recursive_model,
            feature_columns=columns,
            history=history,
            steps=MAX_HORIZON,
        )

        direct = forecast_direct(
            models=direct_models,
            feature_columns=columns,
            history=history,
            max_horizon=MAX_HORIZON,
        )

        for horizon in range(1, MAX_HORIZON + 1):
            actual = float(
                targets.iloc[horizon - 1]["demand_mw"]
            )

            rec_prediction = float(
                recursive.iloc[horizon - 1][
                    "predicted_demand_mw"
                ]
            )

            dir_prediction = float(
                direct.iloc[horizon - 1][
                    "predicted_demand_mw"
                ]
            )

            hybrid_prediction = (
                dir_prediction
                if HORIZON_METHODS[horizon] == "direct"
                else rec_prediction
            )

            for method, prediction in [
                ("Recursivo", rec_prediction),
                ("Directo", dir_prediction),
                ("Híbrido", hybrid_prediction),
            ]:
                absolute_error = abs(actual - prediction)

                results.append({
                    "origin": origin,
                    "target": targets.iloc[
                        horizon - 1
                    ]["datetime_utc"],
                    "horizon": horizon,
                    "method": method,
                    "actual_mw": actual,
                    "predicted_mw": prediction,
                    "absolute_error_mw": absolute_error,
                    "percentage_error": (
                        absolute_error / actual * 100
                    ),
                })

    if not results:
        raise ValueError(
            "No se han encontrado predicciones evaluables."
        )

    results = pd.DataFrame(results)

    summary = (
        results.groupby(["horizon", "method"])
        .agg(
            predictions=("absolute_error_mw", "size"),
            MAE_MW=("absolute_error_mw", "mean"),
            MAPE_percent=("percentage_error", "mean"),
        )
        .round(2)
    )

    print("\nVALIDACIÓN INDEPENDIENTE - OCTUBRE 2026")
    print("----------------------------------------")
    print("Orígenes descartados por historial:", skipped_history)
    print("Orígenes descartados por objetivos:", skipped_targets)
    print("\nRESULTADOS")
    print(summary.to_string())

    output = ROOT / "data/processed/october_validation_results.parquet"
    results.to_parquet(output, index=False)

    print("\nResultados guardados en:", output)


if __name__ == "__main__":
    main()

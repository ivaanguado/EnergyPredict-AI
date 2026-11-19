
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from energypredict.training.multihorizon_dataset import (
    build_multihorizon_dataset,
    get_horizon_features,
)


ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    ROOT / "data/processed/features_spain_2024.parquet"
)

OUTPUT_DIR = ROOT / "models/direct_final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "lag_1",
    "lag_2",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
    "rolling_std_24",
    "rolling_mean_168",
]

MAX_HORIZON = 6


def main():
    df = pd.read_parquet(DATA_PATH)

    multi = build_multihorizon_dataset(
        df,
        max_horizon=MAX_HORIZON,
    )

    # Límite cronológico de entrenamiento
    # según el dataset original de 2024.
    original_test_start = int(len(df) * 0.85)

    test_start_date = df.iloc[
        original_test_start
    ]["target_datetime_utc"]

    print("\nENTRENAMIENTO FINAL MULTIHORIZONTE")
    print("----------------------------------")
    print("Inicio del periodo de prueba:", test_start_date)

    for horizon in range(1, MAX_HORIZON + 1):
        # Cada modelo solo puede utilizar objetivos
        # anteriores al comienzo del periodo de prueba.
        target_dates = (
            multi["origin_datetime_utc"]
            + pd.Timedelta(hours=horizon)
        )

        train = multi.loc[
            target_dates < test_start_date
        ].copy()

        X_train = get_horizon_features(
            train,
            horizon,
            FEATURE_COLUMNS,
        )

        y_train = train[f"target_h{horizon}"]

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        artifact = {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "prediction_horizon_hours": horizon,
            "model_type": "direct_random_forest",
            "model_version": "2.0",
            "training_samples": len(train),
            "training_target_end_utc": target_dates.loc[
                train.index
            ].max(),
        }

        output_path = (
            OUTPUT_DIR
            / f"random_forest_h{horizon}.joblib"
        )

        joblib.dump(artifact, output_path)

        print(
            f"Horizonte +{horizon} h | "
            f"Registros: {len(train)} | "
            f"Guardado correctamente"
        )

    print("\nModelos finales guardados en:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

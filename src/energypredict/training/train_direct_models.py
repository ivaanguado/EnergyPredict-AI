
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
)

from energypredict.training.multihorizon_dataset import (
    build_multihorizon_dataset,
    get_horizon_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    PROJECT_ROOT
    / "data/processed/features_spain_2024.parquet"
)

MODELS_DIR = PROJECT_ROOT / "models/direct"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

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


def train_direct_models():
    # 1. Cargar dataset
    df = pd.read_parquet(DATA_PATH)

    multi = build_multihorizon_dataset(
        df,
        max_horizon=MAX_HORIZON,
    )

    # 2. División cronológica
    n = len(multi)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    # Separación entre conjuntos para evitar
    # solapamiento de objetivos futuros
    gap = MAX_HORIZON - 1

    train = multi.iloc[:train_end - gap]
    val = multi.iloc[train_end:val_end - gap]
    test = multi.iloc[val_end:]

    print("\nENTRENAMIENTO MULTIHORIZONTE")
    print("-----------------------------")
    print("Entrenamiento:", len(train))
    print("Validación:", len(val))
    print("Prueba:", len(test))

    results = []

    # 3. Entrenar un modelo por horizonte
    for horizon in range(1, MAX_HORIZON + 1):
        print(f"\nEntrenando horizonte +{horizon} h...")

        X_train = get_horizon_features(
            train, horizon, FEATURE_COLUMNS
        )
        y_train = train[f"target_h{horizon}"]

        X_val = get_horizon_features(
            val, horizon, FEATURE_COLUMNS
        )
        y_val = val[f"target_h{horizon}"]

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        # 4. Evaluar en validación
        predictions = model.predict(X_val)

        mae = mean_absolute_error(
            y_val, predictions
        )

        mape = (
            mean_absolute_percentage_error(
                y_val, predictions
            ) * 100
        )

        # 5. Guardar modelo
        artifact = {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "prediction_horizon_hours": horizon,
            "model_type": "direct_random_forest",
            "model_version": "1.0",
        }

        model_path = (
            MODELS_DIR
            / f"random_forest_h{horizon}.joblib"
        )

        joblib.dump(artifact, model_path)

        results.append({
            "horizon": horizon,
            "MAE_MW": mae,
            "MAPE_percent": mape,
        })

        print(
            f"MAE: {mae:.2f} MW | "
            f"MAPE: {mape:.2f} %"
        )

    # 6. Mostrar resultados
    summary = pd.DataFrame(results)

    print("\nRESULTADOS DE VALIDACIÓN")
    print("-----------------------------")
    print(
        summary.round(2).to_string(index=False)
    )

    print("\nModelos guardados en:", MODELS_DIR)


if __name__ == "__main__":
    train_direct_models()

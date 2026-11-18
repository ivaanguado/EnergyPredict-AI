from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "models/random_forest_v1.joblib"
DATA_PATH = PROJECT_ROOT / "data/processed/features_spain_2024.parquet"


def predict_next_hour():
    # Cargar el modelo entrenado
    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    # Cargar los datos históricos
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values("target_datetime_utc")

    # Seleccionar la última observación
    last_row = df.iloc[[-1]]

    # Preparar las características
    X = last_row[feature_columns]

    # Realizar predicción
    prediction = model.predict(X)[0]

    # Obtener fecha objetivo
    prediction_date = last_row["target_datetime_utc"].iloc[0]

    # Mostrar resultado
    print("\nENERGYPREDICT AI")
    print("--------------------------")
    print("Modelo: Random Forest")
    print(f"Fecha objetivo: {prediction_date}")
    print(f"Demanda estimada: {prediction:,.2f} MW")


if __name__ == "__main__":
    predict_next_hour()

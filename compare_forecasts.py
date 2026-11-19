
from pathlib import Path

import joblib
import pandas as pd

from energypredict.evaluation.backtesting import (
    evaluate_recursive_forecast,
)

from energypredict.evaluation.direct_backtesting import (
    evaluate_direct_forecast,
)


ROOT = Path(__file__).resolve().parent

# Cargar modelo recursivo
recursive_artifact = joblib.load(
    ROOT / "models/random_forest_v1.joblib"
)

recursive_model = recursive_artifact["model"]

feature_columns = recursive_artifact[
    "feature_columns"
]

# Cargar modelos directos
direct_models = {}

for horizon in range(1, 7):
    artifact = joblib.load(
        ROOT
        / f"models/direct_final/random_forest_h{horizon}.joblib"
    )

    direct_models[horizon] = artifact["model"]

# Cargar los datos de noviembre de 2026
# desde el dataset horario ya procesado
data_path = ROOT / "data/processed/demand_spain_2026.parquet"

if not data_path.exists():
    raise FileNotFoundError(
        f"No existe el dataset: {data_path}"
    )

hourly = pd.read_parquet(data_path)

hourly = hourly[
    (hourly["datetime_utc"] >= "2026-11-01")
    & (hourly["datetime_utc"] < "2026-11-19")
].copy()

print("Horas disponibles:", len(hourly))

# Evaluación recursiva
recursive_results = evaluate_recursive_forecast(
    recursive_model,
    feature_columns,
    hourly,
    max_horizon=6,
    min_history=169,
)

# Evaluación directa
direct_results = evaluate_direct_forecast(
    direct_models,
    feature_columns,
    hourly,
    max_horizon=6,
    min_history=169,
)

# Comparar solamente observaciones comunes
keys = [
    "origin_datetime_utc",
    "target_datetime_utc",
    "horizon",
]

comparison = recursive_results.merge(
    direct_results,
    on=keys,
    suffixes=("_recursive", "_direct"),
    validate="one_to_one",
)

summary = comparison.groupby("horizon").agg(
    predictions=("absolute_error_mw_direct", "size"),
    recursive_mae=("absolute_error_mw_recursive", "mean"),
    direct_mae=("absolute_error_mw_direct", "mean"),
    recursive_mape=("percentage_error_recursive", "mean"),
    direct_mape=("percentage_error_direct", "mean"),
).round(2)

print("\nCOMPARACIÓN SOBRE LAS MISMAS OBSERVACIONES")
print("-----------------------------------------")
print(summary.to_string())


# Comparación diaria de errores
comparison["date"] = (
    comparison["target_datetime_utc"]
    .dt.date
)

daily = comparison.groupby(
    ["date", "horizon"]
).agg(
    recursive_mae=(
        "absolute_error_mw_recursive", "mean"
    ),
    direct_mae=(
        "absolute_error_mw_direct", "mean"
    ),
).reset_index()

# Contar cuántos días gana cada arquitectura
daily["winner"] = daily.apply(
    lambda row: (
        "Directo"
        if row["direct_mae"] < row["recursive_mae"]
        else "Recursivo"
    ),
    axis=1,
)

wins = pd.crosstab(
    daily["horizon"],
    daily["winner"],
)

print("\nDÍAS GANADOS POR CADA MODELO")
print("--------------------------------")
print(wins.to_string())


# Comparación por bloques temporales
dates = sorted(comparison["target_datetime_utc"].dt.date.unique())

midpoint = len(dates) // 2

first_dates = set(dates[:midpoint])

comparison["period"] = (
    comparison["target_datetime_utc"]
    .dt.date
    .apply(
        lambda date: (
            "Primera mitad"
            if date in first_dates
            else "Segunda mitad"
        )
    )
)

period_results = comparison.groupby(
    ["period", "horizon"]
).agg(
    predictions=("absolute_error_mw_direct", "size"),
    recursive_mae=("absolute_error_mw_recursive", "mean"),
    direct_mae=("absolute_error_mw_direct", "mean"),
).round(2)

print("\nCOMPARACIÓN POR PERIODOS")
print("-----------------------------")
print(period_results.to_string())

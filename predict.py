
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd

from energypredict.ingestion.entsoe_client import (
    get_entsoe_demand,
    parse_entsoe_demand,
)
from energypredict.processing.entsoe_transform import (
    transform_entsoe_to_hourly,
)
from energypredict.prediction.recursive_forecast import (
    forecast_recursive,
)
from energypredict.prediction.direct_forecast import (
    forecast_direct,
)

from energypredict.prediction.hybrid_forecast import (
    forecast_hybrid,
)


PROJECT_ROOT = Path(__file__).resolve().parent

RECURSIVE_MODEL_PATH = (
    PROJECT_ROOT / "models/random_forest_v1.joblib"
)

DIRECT_MODELS_DIR = (
    PROJECT_ROOT / "models/direct_final"
)

SPAIN_TIMEZONE = ZoneInfo("Europe/Madrid")

MODEL_MAX_STEPS = {
    "recursive": 12,
    "direct": 6,
    "hybrid": 6,
}

MAX_USER_HORIZON = 6

def load_recursive_model():
    artifact = joblib.load(RECURSIVE_MODEL_PATH)

    return (
        artifact["model"],
        artifact["feature_columns"],
    )


def load_direct_models():
    models = {}
    feature_columns = None

    for horizon in range(1, 7):
        path = (
            DIRECT_MODELS_DIR
            / f"random_forest_h{horizon}.joblib"
        )

        artifact = joblib.load(path)

        columns = artifact["feature_columns"]

        if feature_columns is None:
            feature_columns = columns
        elif feature_columns != columns:
            raise ValueError(
                "Los modelos directos tienen "
                "características incompatibles."
            )

        models[horizon] = artifact["model"]

    return models, feature_columns


def get_recent_history():
    now = datetime.now(timezone.utc)

    end = (
        now.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        + timedelta(days=1)
    )

    start = end - timedelta(days=10)

    print("\nConsultando datos recientes de ENTSO-E...")

    xml = get_entsoe_demand(
        start_date=start.strftime("%Y%m%d%H%M"),
        end_date=end.strftime("%Y%m%d%H%M"),
    )

    raw = parse_entsoe_demand(xml)

    hourly = transform_entsoe_to_hourly(raw)

    if hourly.empty:
        raise ValueError(
            "No se han obtenido observaciones horarias."
        )

    return hourly

def get_horizon_limits(hourly, method="recursive"):
    if method not in MODEL_MAX_STEPS:
        raise ValueError(f"Método no reconocido: {method}")

    last_observation = hourly["datetime_utc"].max()

    current_hour = pd.Timestamp.now(tz="UTC").floor("h")

    delay_hours = int(
        (current_hour - last_observation)
        / pd.Timedelta(hours=1)
    )

    max_steps = MODEL_MAX_STEPS[method]

    max_horizon = max(
        0,
        min(
            MAX_USER_HORIZON,
            max_steps - delay_hours,
        ),
    )

    return {
        "last_observation": last_observation,
        "current_hour": current_hour,
        "delay_hours": delay_hours,
        "max_steps": max_steps,
        "max_horizon": max_horizon,
    }

def predict_next_hour(
    method="recursive",
    horizon=None,
    return_data=False,
    history=None,
):
    # 1. Obtener datos recientes
    hourly = (
        get_recent_history()
        if history is None
        else history.copy()
    )

    # 2. Calcular los límites del horizonte
    limits = get_horizon_limits(hourly, method)

    last_observation = limits["last_observation"]
    max_steps = limits["max_steps"]
    max_horizon = limits["max_horizon"]

    if horizon is None:
        horizon = 1

    if not isinstance(horizon, int) or not 1 <= horizon <= 6:
        raise ValueError(
            "El horizonte debe estar entre 1 y 6."
        )

    if horizon > max_horizon:
        raise ValueError(
            f"El método {method} permite actualmente "
            f"un horizonte máximo de {max_horizon} horas "
            "debido al retraso de ENTSO-E."
        )

    target_hour = (
        limits["current_hour"]
        + pd.Timedelta(hours=horizon)
    )

    steps = int(
        (target_hour - last_observation)
        / pd.Timedelta(hours=1)
    )

    if not 1 <= steps <= max_steps:
        raise ValueError(
            "El horizonte supera los límites del modelo."
        )

    print(f"\nMétodo seleccionado: {method}")
    print(f"Retraso de ENTSO-E: {limits['delay_hours']} horas")
    print(f"Horizonte máximo disponible: {max_horizon} horas")
    print(f"Horizonte necesario: {steps} horas")
    print(f"Hora objetivo UTC: {target_hour}")
    

    # 3. Generar predicciones
    if method == "recursive":
        model, feature_columns = (
            load_recursive_model()
        )

        forecast = forecast_recursive(
            model=model,
            feature_columns=feature_columns,
            history=hourly,
            steps=steps,
        )

    elif method == "direct":
        models, feature_columns = (
            load_direct_models()
        )

        forecast = forecast_direct(
            models=models,
            feature_columns=feature_columns,
            history=hourly,
            max_horizon=steps,
        )

    elif method == "hybrid":
        recursive_model, recursive_columns = (
            load_recursive_model()
        )

        direct_models, direct_columns = (
            load_direct_models()
        )

        if recursive_columns != direct_columns:
            raise ValueError(
                "Las características de los modelos "
                "recursivo y directo no coinciden."
            )

        forecast = forecast_hybrid(
            recursive_model=recursive_model,
            direct_models=direct_models,
            feature_columns=recursive_columns,
            history=hourly,
            steps=steps,
        )

    else:
        raise ValueError(
            f"Método no reconocido: {method}"
        )

    
    # Devolver datos estructurados para la interfaz web
    if return_data:
        return {
            "history": hourly.copy(),
            "forecast": forecast.copy(),
            "method": method,
            "steps": steps,
            "last_observation": last_observation,
            "target_hour": target_hour,
        }


    # 4. Mostrar resultados
    print("\nENERGYPREDICT AI")
    print("--------------------------")
    print("Modelo: Random Forest")
    print(f"Método: {method}")
    print(f"Última observación: {last_observation}")

    print("\nPREDICCIONES")
    print("--------------------------")

    for i, row in forecast.iterrows():
        date_spain = row[
            "datetime_utc"
        ].tz_convert(SPAIN_TIMEZONE)

        prediction = row["predicted_demand_mw"]

        is_target = i == len(forecast) - 1

        
        label = (
            "OBJETIVO"
            if is_target
            else "Intermedia"
        )

        # Identificar el modelo utilizado
        if method == "hybrid":
            model_used = row["method"].capitalize()
        else:
            model_used = method.capitalize()

        print(
            f"+{i + 1} h | "
            f"{date_spain:%d/%m/%Y %H:%M} | "
            f"{prediction:,.2f} MW | "
            f"{model_used} | "
            f"{label}"
        )


    # 5. Resumen final
    final_prediction = forecast.iloc[-1]

    final_date = final_prediction[
        "datetime_utc"
    ].tz_convert(SPAIN_TIMEZONE)

    print("\nPREDICCIÓN OBJETIVO")
    print("--------------------------")
    print(f"Fecha (España): {final_date:%d/%m/%Y %H:%M}")
    print(
        "Demanda estimada:",
        f"{final_prediction['predicted_demand_mw']:,.2f} MW"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "EnergyPredict AI - Predicción "
            "de demanda eléctrica en España"
        )
    )

    parser.add_argument(
        "--method",
        choices=["recursive", "direct", "hybrid"],
        default="recursive",
        help="Método de predicción (por defecto: recursive)",
    )

    args = parser.parse_args()

    predict_next_hour(method=args.method)

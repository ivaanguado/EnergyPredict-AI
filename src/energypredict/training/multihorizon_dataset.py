
import pandas as pd


def build_multihorizon_dataset(
    features: pd.DataFrame,
    max_horizon: int = 6,
) -> pd.DataFrame:
    """
    Prepara objetivos para predicciones directas
    desde 1 hasta max_horizon horas.

    La fila original corresponde a una predicción
    de +1 hora desde la última demanda conocida.
    """

    if max_horizon < 1:
        raise ValueError("max_horizon debe ser positivo.")

    data = features.copy()

    data = data.sort_values(
        "target_datetime_utc"
    ).reset_index(drop=True)

    dates = pd.to_datetime(
        data["target_datetime_utc"],
        utc=True,
    )

    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("Hay fechas inválidas o duplicadas.")

    if not dates.diff().dropna().eq(
        pd.Timedelta(hours=1)
    ).all():
        raise ValueError(
            "Las fechas deben ser horarias y consecutivas."
        )

    # Guardar la fecha de origen:
    # última hora con demanda real conocida
    data["origin_datetime_utc"] = (
        dates - pd.Timedelta(hours=1)
    )

    # Crear el objetivo correspondiente a cada horizonte
    for horizon in range(1, max_horizon + 1):
        data[f"target_h{horizon}"] = (
            data["target_next_hour"].shift(
                -(horizon - 1)
            )
        )

    # Eliminar las últimas filas sin objetivos futuros
    data = data.dropna().reset_index(drop=True)

    return data


def get_horizon_features(
    data: pd.DataFrame,
    horizon: int,
    feature_columns: list,
) -> pd.DataFrame:
    """
    Prepara las características para un horizonte concreto.

    Las variables históricas se mantienen en el origen.
    Las variables de calendario corresponden a la
    hora futura que queremos predecir.
    """

    if horizon < 1:
        raise ValueError("El horizonte debe ser positivo.")

    target_dates = (
        pd.to_datetime(
            data["origin_datetime_utc"],
            utc=True,
        )
        + pd.Timedelta(hours=horizon)
    )

    X = data[feature_columns].copy()

    X["hour"] = target_dates.dt.hour
    X["day_of_week"] = target_dates.dt.dayofweek
    X["month"] = target_dates.dt.month
    X["is_weekend"] = (
        target_dates.dt.dayofweek >= 5
    ).astype(int)

    return X[feature_columns]

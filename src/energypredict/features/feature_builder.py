
import pandas as pd


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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera las características para predecir la demanda
    eléctrica de la siguiente hora.

    Recibe un DataFrame con:
    - datetime_utc
    - demand_mw

    Devuelve características, fecha objetivo y, cuando
    existe, la demanda real de la siguiente hora.
    """
    df = df.copy()
    df["datetime_utc"] = pd.to_datetime(
        df["datetime_utc"], utc=True
    )
    df = df.sort_values("datetime_utc").reset_index(drop=True)

    if df["datetime_utc"].duplicated().any():
        raise ValueError("Hay horas duplicadas.")

    if not df["datetime_utc"].diff().dropna().eq(
        pd.Timedelta(hours=1)
    ).all():
        raise ValueError("La serie contiene saltos horarios.")

    # Hora que queremos predecir
    df["target_datetime_utc"] = (
        df["datetime_utc"] + pd.Timedelta(hours=1)
    )

    # Calendario de la hora objetivo en España
    target_local = df["target_datetime_utc"].dt.tz_convert(
        "Europe/Madrid"
    )

    df["hour"] = target_local.dt.hour
    df["day_of_week"] = target_local.dt.day_of_week
    df["month"] = target_local.dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Demanda de horas anteriores a la hora objetivo
    df["lag_1"] = df["demand_mw"]
    df["lag_2"] = df["demand_mw"].shift(1)
    df["lag_24"] = df["demand_mw"].shift(23)
    df["lag_168"] = df["demand_mw"].shift(167)

    # Estadísticas de las últimas horas conocidas
    df["rolling_mean_24"] = (
        df["demand_mw"].rolling(24).mean()
    )
    df["rolling_std_24"] = (
        df["demand_mw"].rolling(24).std()
    )
    df["rolling_mean_168"] = (
        df["demand_mw"].rolling(168).mean()
    )

    # Objetivo conocido para evaluación histórica
    df["target_next_hour"] = df["demand_mw"].shift(-1)

    return df[
        ["target_datetime_utc", *FEATURE_COLUMNS, "target_next_hour"]
    ].dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

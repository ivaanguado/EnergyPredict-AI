
import pandas as pd


def demand_to_dataframe(data, indicator_id=2037):
    """Extrae la demanda real de la respuesta de REData."""

    indicators = data.get("included", [])

    matches = [
        item for item in indicators
        if str(item.get("id")) == str(indicator_id)
    ]

    if len(matches) != 1:
        raise ValueError(
            f"No se encontró un único indicador {indicator_id}."
        )

    values = matches[0]["attributes"].get("values", [])

    if not values:
        raise ValueError("El indicador no contiene registros.")

    df = pd.DataFrame(values)

    df = df[["datetime", "value"]].copy()

    df["datetime_utc"] = pd.to_datetime(
        df["datetime"],
        utc=True,
        errors="coerce",
    )

    df["demand_mw"] = pd.to_numeric(
        df["value"],
        errors="coerce",
    )

    df = df[["datetime_utc", "demand_mw"]]

    return df.sort_values("datetime_utc").reset_index(drop=True)


def aggregate_hourly(df):
    """Agrega la demanda de cinco minutos a valores horarios."""

    if df["datetime_utc"].isna().any():
        raise ValueError("Existen fechas inválidas.")

    if df["demand_mw"].isna().any():
        raise ValueError("Existen valores de demanda inválidos.")

    if df["datetime_utc"].duplicated().any():
        raise ValueError("Existen fechas duplicadas.")

    hourly = (
        df.set_index("datetime_utc")
        .resample("h")
        .agg(
            demand_mw=("demand_mw", "mean"),
            observations=("demand_mw", "count"),
        )
        .reset_index()
    )

    if not hourly["observations"].eq(12).all():
        raise ValueError(
            "Hay horas incompletas: se esperaban 12 registros por hora."
        )

    return hourly.drop(columns=["observations"])

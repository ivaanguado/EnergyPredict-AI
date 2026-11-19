
import pandas as pd


def transform_historical_to_hourly(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Transforma demanda de 15 minutos a frecuencia horaria
    conservando información sobre la calidad de los datos.

    Solo interpola huecos aislados de 15 minutos.
    Las horas con huecos mayores quedan marcadas como inválidas.
    """

    data = df[["datetime_utc", "demand_mw"]].copy()

    data["datetime_utc"] = pd.to_datetime(
        data["datetime_utc"],
        utc=True,
    )

    data["demand_mw"] = pd.to_numeric(
        data["demand_mw"],
        errors="coerce",
    )

    if data.empty:
        raise ValueError("No se recibieron datos.")

    if data["datetime_utc"].isna().any():
        raise ValueError("Hay fechas inválidas.")

    if data["demand_mw"].isna().any():
        raise ValueError("Hay demandas nulas.")

    if (data["demand_mw"] <= 0).any():
        raise ValueError("Hay demandas no positivas.")

    if data["datetime_utc"].duplicated().any():
        raise ValueError("Hay fechas duplicadas.")

    data = data.sort_values(
        "datetime_utc"
    ).reset_index(drop=True)

    dates = data["datetime_utc"]

    aligned = (
        dates.dt.minute.isin([0, 15, 30, 45])
        & dates.dt.second.eq(0)
        & dates.dt.microsecond.eq(0)
        & dates.dt.nanosecond.eq(0)
    )

    if not aligned.all():
        raise ValueError(
            "Hay registros no alineados a 15 minutos."
        )

    start = dates.min().floor("h")
    end = dates.max().floor("h")

    expected = pd.date_range(
        start=start,
        end=end + pd.Timedelta(minutes=45),
        freq="15min",
        tz="UTC",
    )

    data = data.set_index(
        "datetime_utc"
    ).reindex(expected)

    data.index.name = "datetime_utc"

    missing = data["demand_mw"].isna()

    # Identificar la longitud de cada grupo de ausencias.
    groups = (~missing).cumsum()

    gap_lengths = missing.groupby(groups).transform("sum")

    # Únicamente se permiten interpolaciones aisladas.
    isolated = missing & gap_lengths.eq(1)

    # Interpolación temporal provisional.
    interpolated = data["demand_mw"].interpolate(
        method="time",
        limit_area="inside",
    )

    # Solo sustituir ausencias aisladas.
    data.loc[isolated, "demand_mw"] = (
        interpolated.loc[isolated]
    )

    data["was_interpolated"] = isolated

    # Agregación horaria.
    hourly = (
        data.resample("h")
        .agg(
            demand_mw=("demand_mw", "mean"),
            valid_intervals=("demand_mw", "count"),
            contains_interpolation=(
                "was_interpolated",
                "max",
            ),
        )
        .reset_index()
    )

    # Una hora solo es válida si tiene cuatro intervalos.
    hourly["is_valid"] = (
        hourly["valid_intervals"] == 4
    )

    # Evitar considerar como observación real una media
    # calculada con menos de cuatro intervalos.
    hourly.loc[
        ~hourly["is_valid"],
        "demand_mw",
    ] = float("nan")

    print("\nTRANSFORMACIÓN HISTÓRICA")
    print("------------------------------")
    print("Horas totales:", len(hourly))
    print(
        "Horas válidas:",
        int(hourly["is_valid"].sum()),
    )
    print(
        "Horas inválidas:",
        int((~hourly["is_valid"]).sum()),
    )
    print(
        "Intervalos interpolados:",
        int(data["was_interpolated"].sum()),
    )

    return hourly

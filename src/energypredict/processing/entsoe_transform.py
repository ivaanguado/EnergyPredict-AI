
import pandas as pd


def transform_entsoe_to_hourly(
    df: pd.DataFrame,
    max_consecutive_missing: int = 1,
) -> pd.DataFrame:
    """
    Convierte datos de demanda de ENTSO-E de 15 minutos
    a demanda media horaria.

    - Valida fechas y valores de demanda.
    - Descarta la última hora si está incompleta.
    - Interpola únicamente huecos aislados.
    - Rechaza huecos consecutivos superiores al límite.
    - No extrapola valores fuera del intervalo disponible.
    - Identifica las horas que contienen interpolaciones.
    """

    # 1. Validar y preparar los datos
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
        raise ValueError("Hay valores de demanda nulos.")

    if (data["demand_mw"] <= 0).any():
        raise ValueError("Hay demandas no positivas.")

    if data["datetime_utc"].duplicated().any():
        raise ValueError("Hay fechas duplicadas.")

    if max_consecutive_missing < 0:
        raise ValueError(
            "max_consecutive_missing no puede ser negativo."
        )

    data = data.sort_values(
        "datetime_utc"
    ).reset_index(drop=True)

    # 2. Comprobar frecuencia de 15 minutos
    timestamps = data["datetime_utc"]

    aligned = (
        timestamps.dt.minute.isin([0, 15, 30, 45])
        & timestamps.dt.second.eq(0)
        & timestamps.dt.microsecond.eq(0)
        & timestamps.dt.nanosecond.eq(0)
    )

    if not aligned.all():
        raise ValueError(
            "Hay registros que no están alineados "
            "a intervalos de 15 minutos."
        )

    # 3. Descartar la última hora si está incompleta
    data["hour"] = data["datetime_utc"].dt.floor("h")

    last_hour = data["hour"].max()

    last_hour_count = (
        data["hour"] == last_hour
    ).sum()

    if last_hour_count < 4:
        print(
            "Última hora incompleta descartada:",
            last_hour,
        )

        data = data[
            data["hour"] < last_hour
        ].copy()

    data = data.drop(columns=["hour"])

    if data.empty:
        raise ValueError(
            "No quedan datos tras descartar "
            "la última hora incompleta."
        )

    # 4. Construir la secuencia temporal esperada
    start = data["datetime_utc"].min().floor("h")

    end = data["datetime_utc"].max().floor("h")

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

    # 5. Identificar registros ausentes
    data["was_interpolated"] = (
        data["demand_mw"].isna()
    )

    missing_count = int(
        data["was_interpolated"].sum()
    )

    # 6. Detectar huecos consecutivos
    missing = data["was_interpolated"]

    missing_groups = (
        missing.groupby(
            (~missing).cumsum()
        ).sum()
    )

    longest_gap = int(missing_groups.max())

    if longest_gap > max_consecutive_missing:
        raise ValueError(
            "Se ha detectado un hueco de "
            f"{longest_gap} intervalos consecutivos. "
            f"Máximo permitido: {max_consecutive_missing}."
        )

    # 7. Interpolar únicamente huecos permitidos
    data["demand_mw"] = (
        data["demand_mw"].interpolate(
            method="time",
            limit_area="inside",
        )
    )

    if data["demand_mw"].isna().any():
        raise ValueError(
            "No se pudieron recuperar todos "
            "los registros ausentes."
        )

    # 8. Transformar a frecuencia horaria
    hourly = (
        data.resample("h")
        .agg(
            demand_mw=("demand_mw", "mean"),
            contains_interpolation=(
                "was_interpolated",
                "max",
            ),
        )
        .reset_index()
    )

    # 9. Mostrar resumen de calidad
    print(
        f"Transformación completada: "
        f"{len(hourly)} horas."
    )

    print(
        f"Registros interpolados: {missing_count}"
    )

    return hourly

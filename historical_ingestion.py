
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.energypredict.ingestion.ree_client import get_demand_data
from src.energypredict.processing.transform import (
    demand_to_dataframe,
    aggregate_hourly,
)
from src.energypredict.processing.storage import save_parquet


START_DATE = date(2024, 1, 1)
END_DATE = date(2024, 1, 14)  # Primero probamos dos semanas

BATCH_DAYS = 7
OUTPUT_DIR = Path("data/processed/batches")
TIMEZONE = "Europe/Madrid"


def download_batch(start, end):
    """Descarga y valida un intervalo de días completos."""

    print(f"\nDescargando: {start} -> {end}", flush=True)

    data = get_demand_data(
        start_date=f"{start}T00:00",
        end_date=f"{end}T23:59",
    )

    df = demand_to_dataframe(data, indicator_id=2037)

    # Límites esperados del intervalo en hora peninsular.
    start_utc = pd.Timestamp(
        start
    ).tz_localize(TIMEZONE).tz_convert("UTC")

    end_utc = pd.Timestamp(
        end + timedelta(days=1)
    ).tz_localize(TIMEZONE).tz_convert("UTC")

    expected = pd.date_range(
        start=start_utc,
        end=end_utc,
        freq="5min",
        inclusive="left",
    )

    actual = pd.DatetimeIndex(df["datetime_utc"])

    if actual.hasnans:
        raise ValueError("Existen fechas inválidas.")

    if actual.has_duplicates:
        raise ValueError("Existen fechas duplicadas.")

    if df["demand_mw"].isna().any():
        raise ValueError("Existen valores nulos.")

    if (df["demand_mw"] <= 0).any():
        raise ValueError("Existen valores no positivos.")

    
    if not actual.equals(expected):
        missing = expected.difference(actual)
        unexpected = actual.difference(expected)

        print("\nDIAGNÓSTICO TEMPORAL")
        print("-" * 50)

        print(f"Primer registro: {actual.min()}")
        print(f"Último registro: {actual.max()}")

        print(f"Intervalos ausentes: {len(missing)}")
        print(f"Intervalos inesperados: {len(unexpected)}")

        print("\nPrimeros 10 intervalos ausentes:")
        print(missing[:10])

        print("\nÚltimos 10 intervalos ausentes:")
        print(missing[-10:])

        print("\nFrecuencias observadas:")
        print(
            df["datetime_utc"]
            .diff()
            .value_counts()
            .head(10)
        )

        raise ValueError(
            "La respuesta no contiene todos los "
            "intervalos esperados."
        )


    hourly = aggregate_hourly(df)

    print(
        f"Validación correcta: {len(df)} registros "
        f"-> {len(hourly)} horas."
    )

    return hourly


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    current = START_DATE
    files = []

    while current <= END_DATE:
        batch_end = min(
            current + timedelta(days=BATCH_DAYS - 1),
            END_DATE,
        )

        filename = (
            f"demand_{current}_{batch_end}.parquet"
        )

        path = OUTPUT_DIR / filename

        if path.exists():
            print(f"Lote existente: {filename}")
        else:
            hourly = download_batch(current, batch_end)
            save_parquet(hourly, path)

            # Evitamos lanzar peticiones consecutivas.
            time.sleep(2)

        files.append(path)
        current = batch_end + timedelta(days=1)

    # Consolidamos los lotes descargados.
    historical = pd.concat(
        [pd.read_parquet(path) for path in files],
        ignore_index=True,
    )

    historical = historical.sort_values(
        "datetime_utc"
    ).reset_index(drop=True)

    if historical["datetime_utc"].duplicated().any():
        raise ValueError("Hay fechas duplicadas entre lotes.")

    expected_start = pd.Timestamp(
        START_DATE
    ).tz_localize(TIMEZONE).tz_convert("UTC")

    expected_end = pd.Timestamp(
        END_DATE + timedelta(days=1)
    ).tz_localize(TIMEZONE).tz_convert("UTC")

    expected_hours = pd.date_range(
        expected_start,
        expected_end,
        freq="h",
        inclusive="left",
    )

    actual_hours = pd.DatetimeIndex(
        historical["datetime_utc"]
    )

    if not actual_hours.equals(expected_hours):
        raise ValueError(
            "El histórico consolidado tiene horas "
            "ausentes, adicionales o desordenadas."
        )

    output = (
        f"data/processed/"
        f"demand_hourly_{START_DATE}_{END_DATE}.parquet"
    )

    save_parquet(historical, output)

    print("\nHISTÓRICO COMPLETADO")
    print(f"Total de horas: {len(historical)}")
    print(f"Desde: {historical['datetime_utc'].min()}")
    print(f"Hasta: {historical['datetime_utc'].max()}")


if __name__ == "__main__":
    main()

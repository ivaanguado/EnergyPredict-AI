
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw/entsoe_2024")
OUTPUT = Path("data/processed/demand_spain_2024.parquet")

SPAIN_CODE = "10YES-REE------0"


def main():
    files = sorted(
        RAW_DIR.glob("2024_*_ActualTotalLoad_6.1.A_r3.csv")
    )

    print("Archivos encontrados:", len(files))

    if len(files) != 12:
        raise ValueError("Se esperaban los 12 CSV de 2024.")

    frames = []

    for file in files:
        df = pd.read_csv(file, sep="\t")

        df = df[df["AreaCode"] == SPAIN_CODE].copy()

        print(f"{file.name}: {len(df)} registros")

        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    data["datetime_utc"] = pd.to_datetime(
        data["DateTime(UTC)"],
        utc=True,
        errors="coerce",
    )

    data["demand_mw"] = pd.to_numeric(
        data["TotalLoad[MW]"],
        errors="coerce",
    )

    data = data[["datetime_utc", "demand_mw"]]

    data = data.sort_values(
        "datetime_utc"
    ).reset_index(drop=True)

    print("\nRegistros originales:", len(data))

    # Validaciones de calidad
    if data["datetime_utc"].isna().any():
        raise ValueError("Hay fechas inválidas.")

    if data["demand_mw"].isna().any():
        raise ValueError("Hay valores de demanda nulos.")

    if (data["demand_mw"] <= 0).any():
        raise ValueError(
            "Hay valores de demanda no positivos."
        )

    if data["datetime_utc"].duplicated().any():
        raise ValueError("Hay fechas duplicadas.")

    # Generar todos los intervalos esperados de 2024
    expected = pd.date_range(
        "2024-01-01",
        "2025-01-01",
        freq="15min",
        tz="UTC",
        inclusive="left",
    )

    actual = pd.DatetimeIndex(data["datetime_utc"])

    missing = expected.difference(actual)
    unexpected = actual.difference(expected)

    print("Intervalos ausentes:", len(missing))
    print("Intervalos inesperados:", len(unexpected))
    print("Fechas ausentes:", missing.tolist())

    if len(unexpected) > 0:
        raise ValueError(
            "Existen registros fuera del intervalo esperado."
        )

    if len(missing) > 1:
        raise ValueError(
            "Hay demasiados registros ausentes."
        )

    # Reconstruir la serie temporal completa
    data = data.set_index("datetime_utc").reindex(expected)
    data.index.name = "datetime_utc"

    # Marcar los registros que necesitan interpolación
    data["was_interpolated"] = data["demand_mw"].isna()

    # Interpolación temporal de valores ausentes
    data["demand_mw"] = data["demand_mw"].interpolate(
        method="time",
        limit_area="inside",
    )

    if data["demand_mw"].isna().any():
        raise ValueError(
            "Quedan valores sin recuperar."
        )

    print(
        "Registros interpolados:",
        int(data["was_interpolated"].sum()),
    )

    # Agregar los datos a frecuencia horaria
    hourly = (
        data.resample("h")
        .agg(
            demand_mw=("demand_mw", "mean"),
            contains_interpolation=("was_interpolated", "max"),
        )
        .reset_index()
    )

    if len(hourly) != 8784:
        raise ValueError(
            f"Se esperaban 8784 horas, recibidas {len(hourly)}."
        )

    print(
        "Horas con datos interpolados:",
        int(hourly["contains_interpolation"].sum()),
    )

    # Guardar el dataset procesado
    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    hourly.to_parquet(OUTPUT, index=False)

    print("\nPROCESAMIENTO COMPLETADO")
    print("Registros horarios:", len(hourly))
    print("Primer registro:", hourly["datetime_utc"].min())
    print("Último registro:", hourly["datetime_utc"].max())
    print("Archivo guardado:", OUTPUT)


if __name__ == "__main__":
    main()

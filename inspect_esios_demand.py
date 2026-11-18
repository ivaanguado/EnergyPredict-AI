
import pandas as pd

from src.energypredict.ingestion.esios_client import (
    get_esios_demand,
)


def main():
    print("ENERGYPREDICT AI - HISTÓRICO HORARIO")
    print("-" * 50)

    data = get_esios_demand(
        indicator_id=2037,
        start_date="2024-01-01T00:00:00",
        end_date="2024-01-07T23:59:59",
        time_trunc="hour",
    )

    indicator = data.get("indicator", {})

    print("\nIndicador:", indicator.get("name"))

    values = indicator.get("values", [])

    print("Registros recibidos:", len(values))

    if not values:
        raise ValueError("La API no devolvió registros.")

    print("\nPrimer registro:")
    print(values[0])

    print("\nÚltimo registro:")
    print(values[-1])

    df = pd.DataFrame(values)

    print("\nColumnas:")
    print(df.columns.tolist())

    print("\nPrimeras filas:")
    print(df.head())

    if "datetime_utc" in df.columns:
        dates = pd.to_datetime(
            df["datetime_utc"],
            utc=True,
        )

        print("\nFechas duplicadas:", dates.duplicated().sum())

        print("\nFrecuencias:")
        print(dates.sort_values().diff().value_counts().head())


if __name__ == "__main__":
    main()

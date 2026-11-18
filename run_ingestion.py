
from src.energypredict.ingestion.ree_client import get_demand_data
from src.energypredict.processing.transform import (
    demand_to_dataframe,
    aggregate_hourly,
)
from src.energypredict.processing.validation import validate_demand
from src.energypredict.processing.storage import save_parquet


def main():
    print("ENERGYPREDICT AI - DATA INGESTION")
    print("-" * 50)

    # 1. Descargar datos de Red Eléctrica
    data = get_demand_data(
        start_date="2024-01-07T00:00",
        end_date="2024-01-07T23:59",
    )

    # 2. Seleccionar demanda real (ID 2037)
    df = demand_to_dataframe(data, indicator_id=2037)

    print(f"\nRegistros originales: {len(df)}")

    # 3. Validar los datos originales
    report = validate_demand(
        df.rename(columns={"demand_mw": "demand_value"}),
        frequency="5min",
    )

    print("\nInforme de calidad:")
    for key, value in report.items():
        print(f"{key}: {value}")

    if not report["passed_basic_checks"]:
        raise ValueError("Los datos originales no son válidos.")

    # 4. Agregar los datos a frecuencia horaria
    hourly_df = aggregate_hourly(df)

    print(f"\nRegistros horarios: {len(hourly_df)}")

    print("\nPrimeras filas:")
    print(hourly_df.head())

    # 5. Guardar el dataset procesado
    save_parquet(
        hourly_df,
        "data/processed/demand_hourly_2024_01_07.parquet",
    )

    print("\nPipeline completado correctamente.")


if __name__ == "__main__":
    main()

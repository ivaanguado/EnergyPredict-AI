
import json

from src.energypredict.ingestion.ree_client import get_demand_data


def main():
    data = get_demand_data(
        start_date="2024-01-15T00:00",
        end_date="2024-01-15T23:59",
    )

    print("\nINDICADORES DISPONIBLES")
    print("-" * 50)

    for indicator in data["included"]:
        attributes = indicator["attributes"]
        values = attributes.get("values", [])

        print(f"ID: {indicator.get('id')}")
        print(f"Nombre: {attributes.get('title')}")
        print(f"Magnitud: {attributes.get('magnitude')}")
        print(f"Registros: {len(values)}")

        if values:
            print("Primer registro:")
            print(json.dumps(
                values[0],
                indent=2,
                ensure_ascii=False,
            ))

        print("-" * 50)


        import pandas as pd

        real_indicator = next(
            item for item in data["included"]
            if str(item["id"]) == "2037"
        )

        df = pd.DataFrame(
            real_indicator["attributes"]["values"]
        )

        df["datetime"] = pd.to_datetime(
            df["datetime"],
            utc=True
        )

        df = df.sort_values("datetime")

        print("\nANÁLISIS TEMPORAL")
        print("-" * 50)

        print("Primeras observaciones:")
        print(df[["datetime", "value"]].head(10))

        print("\nÚltimas observaciones:")
        print(df[["datetime", "value"]].tail(5))

        print("\nDiferencias entre registros:")
        print(df["datetime"].diff().value_counts().head())

        print("\nRango temporal:")
        print(df["datetime"].min())
        print(df["datetime"].max())


    with open(
        "data/raw/ree_sample.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    print("Respuesta original guardada correctamente.")


if __name__ == "__main__":
    main()

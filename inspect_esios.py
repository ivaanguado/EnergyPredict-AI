
from src.energypredict.ingestion.esios_client import (
    get_esios_indicators,
)


def main():
    print("ENERGYPREDICT AI - ESIOS")
    print("-" * 50)

    data = get_esios_indicators()

    indicators = data.get("indicators", [])

    print("Total de indicadores:", len(indicators))

    keywords = ["demanda real", "demanda peninsular"]

    matches = [
        indicator
        for indicator in indicators
        if any(
            keyword in indicator.get("name", "").lower()
            for keyword in keywords
        )
    ]

    print("\nINDICADORES DE DEMANDA")
    print("-" * 50)

    for indicator in matches:
        print(
            f"ID: {indicator.get('id')} | "
            f"Nombre: {indicator.get('name')}"
        )


if __name__ == "__main__":
    main()

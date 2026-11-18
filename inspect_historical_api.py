
import requests

BASE_URL = "https://apidatos.ree.es/es/datos/demanda/evolucion"

params = {
    "start_date": "2024-01-01T00:00",
    "end_date": "2024-01-07T23:59",
    "time_trunc": "hour",
}

response = requests.get(
    BASE_URL,
    params=params,
    timeout=30,
)

print("URL:", response.url)
print("Estado HTTP:", response.status_code)

if response.status_code != 200:
    print("Detalle del error:")
    print(response.text[:2000])

response.raise_for_status()

data = response.json()

for indicator in data.get("included", []):
    print("\nIndicador:", indicator.get("type"))
    print("ID:", indicator.get("id"))

    values = indicator.get("attributes", {}).get("values", [])

    print("Registros:", len(values))
    print("Primer registro:", values[0] if values else None)
    print("Último registro:", values[-1] if values else None)

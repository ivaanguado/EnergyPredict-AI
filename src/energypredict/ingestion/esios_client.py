
import os
import requests


BASE_URL = "https://api.esios.ree.es"


def get_esios_indicators():
    """Obtiene el catálogo de indicadores de e·sios."""

    token = os.getenv("ESIOS_TOKEN")

    if not token:
        raise ValueError(
            "No se ha configurado la variable ESIOS_TOKEN."
        )

    headers = {
        "Accept": "application/json; application/vnd.esios-api-v1+json",
        "Content-Type": "application/json",
        "Host": "api.esios.ree.es",
        "Authorization": f"Token token={token}",
        "x-api-key": token,
    }

    response = requests.get(
        f"{BASE_URL}/indicators",
        headers=headers,
        timeout=60,
    )

    print("Estado HTTP:", response.status_code)

    if not response.ok:
        print("Detalle:", response.text[:500])

    response.raise_for_status()

    return response.json()


def get_esios_demand(
    indicator_id,
    start_date,
    end_date,
    time_trunc="hour",
):
    """Consulta los valores históricos de un indicador."""

    token = os.getenv("ESIOS_TOKEN")

    if not token:
        raise ValueError("No se ha configurado ESIOS_TOKEN.")

    headers = {
        "Accept": "application/json; application/vnd.esios-api-v1+json",
        "Authorization": f"Token token={token}",
    }

    params = {
        "start_date": start_date,
        "end_date": end_date,
        "time_trunc": time_trunc,
    }

    response = requests.get(
        f"{BASE_URL}/indicators/{indicator_id}",
        headers=headers,
        params=None,
        timeout=60,
    )

    print("Estado HTTP:", response.status_code)

    if not response.ok:
        print("Detalle:", response.text[:1000])

    response.raise_for_status()

    return response.json()

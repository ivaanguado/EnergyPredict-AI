
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://apidatos.ree.es/es/datos"


def create_session():
    """Crea una sesión HTTP con reintentos automáticos."""

    session = requests.Session()

    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retries)

    session.mount("https://", adapter)

    session.headers.update({
        "Accept": "application/json",
        "User-Agent": "EnergyPredict-AI/0.1",
    })

    return session


def get_demand_data(start_date, end_date):
    """Consulta la demanda eléctrica peninsular."""

    url = f"{BASE_URL}/demanda/demanda-tiempo-real"

    params = {
        "start_date": start_date,
        "end_date": end_date,
        "time_trunc": "hour",
    }

    with create_session() as session:
        response = session.get(
            url,
            params=params,
            timeout=30,
        )

        print(f"URL: {response.url}")
        print(f"Estado HTTP: {response.status_code}")

        response.raise_for_status()

        data = response.json()

    if "errors" in data:
        raise ValueError(f"Error de REData: {data['errors']}")

    if "included" not in data:
        raise ValueError(
            "La respuesta no contiene indicadores 'included'."
        )

    return data

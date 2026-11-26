import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = PROJECT_ROOT / ".env"

BASE_URL = "https://web-api.tp.entsoe.eu/api"

SPAIN_EIC = "10YES-REE------0"

MAX_ATTEMPTS = 3

RETRY_DELAYS = [5, 10]

REQUEST_TIMEOUT = 60

RETRYABLE_STATUS_CODES = {
    408,
    429,
    500,
    502,
    503,
    504,
    599,
}


def get_entsoe_demand(start_date, end_date):
    """Consulta la demanda eléctrica real con reintentos."""

    load_dotenv(dotenv_path=ENV_PATH)

    token = os.getenv("ENTSOE_API_TOKEN")

    if not token:
        raise ValueError(
            "No se ha configurado ENTSOE_API_TOKEN."
        )

    params = {
        "securityToken": token,
        "documentType": "A65",
        "processType": "A16",
        "outBiddingZone_Domain": SPAIN_EIC,
        "periodStart": start_date,
        "periodEnd": end_date,
    }

    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):

        print(
            f"Consulta ENTSO-E: "
            f"intento {attempt}/{MAX_ATTEMPTS}"
        )

        try:

            response = requests.get(
                BASE_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

        except requests.exceptions.Timeout:

            last_error = "Tiempo de espera agotado."

        except requests.exceptions.SSLError:

            raise RuntimeError(
                "Error de seguridad SSL al conectar con ENTSO-E."
            ) from None

        except requests.exceptions.ConnectionError:

            last_error = "Error de conexión con ENTSO-E."

        except requests.exceptions.RequestException:

            raise RuntimeError(
                "Error inesperado en la petición a ENTSO-E."
            ) from None

        else:

            print("Estado HTTP:", response.status_code)

            if response.ok:
                return response.text

            last_error = (
                f"Estado HTTP: {response.status_code}."
            )

            # No reintentar errores permanentes
            if response.status_code not in RETRYABLE_STATUS_CODES:

                raise RuntimeError(
                    f"Error al consultar ENTSO-E. "
                    f"{last_error} "
                    "Comprueba la configuración de la API."
                )

        # Reintentar únicamente si quedan intentos
        if attempt < MAX_ATTEMPTS:

            delay = RETRY_DELAYS[attempt - 1]

            print(f"Error temporal: {last_error}")

            print(
                f"Reintentando en {delay} segundos..."
            )

            time.sleep(delay)

    raise RuntimeError(
        f"No se pudo consultar ENTSO-E después de "
        f"{MAX_ATTEMPTS} intentos. {last_error}"
    ) from None


def parse_entsoe_demand(xml_text):
    """Convierte el XML de ENTSO-E en una serie temporal."""

    root = ET.fromstring(xml_text)

    # Obtener el namespace del documento XML
    namespace = root.tag.split("}")[0].strip("{")
    ns = {"ns": namespace}

    records = []

    for period in root.findall(".//ns:TimeSeries/ns:Period", ns):

        start_text = period.findtext(
            "ns:timeInterval/ns:start",
            namespaces=ns,
        )

        resolution = period.findtext(
            "ns:resolution",
            namespaces=ns,
        )

        if not start_text or not resolution:
            continue

        start = pd.Timestamp(start_text)
        step = pd.Timedelta(resolution)

        for point in period.findall("ns:Point", ns):

            position = point.findtext(
                "ns:position",
                namespaces=ns,
            )

            quantity = point.findtext(
                "ns:quantity",
                namespaces=ns,
            )

            if position is None or quantity is None:
                continue

            timestamp = start + (int(position) - 1) * step

            records.append({
                "datetime_utc": timestamp,
                "demand_mw": float(quantity),
            })

    if not records:
        raise ValueError(
            "No se encontraron datos de demanda en el XML."
        )

    df = pd.DataFrame(records)

    df["datetime_utc"] = pd.to_datetime(
        df["datetime_utc"],
        utc=True,
    )

    return df.sort_values(
        "datetime_utc"
    ).reset_index(drop=True)
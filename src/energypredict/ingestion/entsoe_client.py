
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = PROJECT_ROOT / ".env"

BASE_URL = "https://web-api.tp.entsoe.eu/api"

SPAIN_EIC = "10YES-REE------0"


def get_entsoe_demand(start_date, end_date):
    """Consulta la demanda eléctrica real de España."""

    load_dotenv(dotenv_path=ENV_PATH)

    token = os.getenv("ENTSOE_API_TOKEN")

    if not token:
        raise ValueError("No se ha configurado ENTSOE_API_TOKEN.")

    params = {
        "securityToken": token,
        "documentType": "A65",
        "processType": "A16",
        "outBiddingZone_Domain": SPAIN_EIC,
        "periodStart": start_date,
        "periodEnd": end_date,
    }

    response = requests.get(
        BASE_URL,
        params=params,
        timeout=60,
    )

    print("Estado HTTP:", response.status_code)

    if not response.ok:
        # Evitar mostrar información sensible de la petición.
        raise RuntimeError(
            f"Error al consultar ENTSO-E. "
            f"Estado HTTP: {response.status_code}. "
            "Comprueba la conexión o inténtalo de nuevo más tarde."
        )

    return response.text


import xml.etree.ElementTree as ET

import pandas as pd


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
        df["datetime_utc"], utc=True
    )

    return df.sort_values("datetime_utc").reset_index(drop=True)

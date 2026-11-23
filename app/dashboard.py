
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# CONFIGURACIÓN DEL PROYECTO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predict import predict_next_hour


st.set_page_config(
    page_title="EnergyPredict AI | Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Cargar estilos CSS
css = (APP_DIR / "styles" / "main.css").read_text(
    encoding="utf-8"
)

st.markdown(
    f"<style>{css}</style>",
    unsafe_allow_html=True,
)


# ============================================================
# COMPONENTES
# ============================================================

def metric_card(label, value, unit=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">
                {value} <span>{unit}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ESTADO DE LA APLICACIÓN
# ============================================================

METHOD_MAP = {
    "Recursivo": "recursive",
    "Directo": "direct",
    "Híbrido": "hybrid",
}

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# ============================================================
# BARRA LATERAL
# ============================================================

with st.sidebar:
    st.markdown(
        '<div class="brand">⚡ Energy<span>Predict</span></div>',
        unsafe_allow_html=True,
    )

    st.caption("PANEL DE CONTROL")
    st.divider()

    method = st.selectbox(
        "Modelo de predicción",
        ["Recursivo", "Directo", "Híbrido"],
    )

    horizon = st.slider(
        "Horizonte de predicción",
        min_value=1,
        max_value=6,
        value=3,
        format="%d h",
    )

    generate = st.button(
        "Generar predicción",
        type="primary",
        use_container_width=True,
    )

    st.divider()
    st.caption("EnergyPredict AI · Versión 2.0")


# ============================================================
# GENERAR PREDICCIÓN
# ============================================================

if generate:
    with st.spinner(
        "Consultando ENTSO-E y generando predicciones..."
    ):
        try:
            result = predict_next_hour(
                method=METHOD_MAP[method],
                horizon=horizon,
                return_data=True,
            )

            st.session_state.prediction_result = result

        except Exception as exc:
            st.session_state.prediction_result = None
            st.error(
                "No se pudo generar la predicción. "
                "Comprueba la conexión con ENTSO-E "
                f"y la disponibilidad de datos. Detalle: {exc}"
            )

result = st.session_state.prediction_result

st.write("DEBUG - Resultado disponible:", result is not None)

if result is not None:
    st.write(
        "DEBUG - Última demanda:",
        result["history"].iloc[-1]["demand_mw"],
    )
    st.write(
        "DEBUG - Predicción final:",
        result["forecast"].iloc[-1]["predicted_demand_mw"],
    )


# ============================================================
# ENCABEZADO PRINCIPAL
# ============================================================


st.markdown(
    """
<div class="hero">
    <div class="eyebrow">ENERGY INTELLIGENCE PLATFORM</div>
    <h1>Predicción energética inteligente.</h1>
    <p>
        Monitoriza la demanda eléctrica en España y explora
        predicciones basadas en inteligencia artificial.
    </p>
</div>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# TARJETAS DE MÉTRICAS
# ============================================================

col1, col2, col3, col4 = st.columns(
    4,
    gap="medium",
)

if result is not None:
    history = result["history"]
    forecast = result["forecast"]

    actual_demand = history.iloc[-1]["demand_mw"]
    predicted_demand = forecast.iloc[-1]["predicted_demand_mw"]

    actual_value = f"{actual_demand:,.0f}".replace(",", ".")
    predicted_value = f"{predicted_demand:,.0f}".replace(",", ".")

    displayed_method = result["method"].capitalize()
    displayed_horizon = str(result["steps"])

else:
    actual_value = "—"
    predicted_value = "—"
    displayed_method = method
    displayed_horizon = str(horizon)


with col1:
    metric_card(
        "Última demanda observada",
        actual_value,
        "MW",
    )

with col2:
    metric_card(
        "Demanda prevista",
        predicted_value,
        "MW",
    )

with col3:
    metric_card(
        "Modelo utilizado",
        displayed_method,
    )

with col4:
    metric_card(
        "Pasos desde última observación",
        displayed_horizon,
        "h",
    )


# ============================================================
# GRÁFICA DE DEMANDA
# ============================================================

st.markdown(
    '<div class="section-heading">Evolución de la demanda</div>',
    unsafe_allow_html=True,
)

if result is not None:

    history = result["history"].tail(24).copy()
    forecast = result["forecast"].copy()

    # Convertir fechas a horario español
    history["datetime_spain"] = (
        pd.to_datetime(
            history["datetime_utc"],
            utc=True,
        ).dt.tz_convert("Europe/Madrid")
    )

    forecast["datetime_spain"] = (
        pd.to_datetime(
            forecast["datetime_utc"],
            utc=True,
        ).dt.tz_convert("Europe/Madrid")
    )

    fig = go.Figure()

    # Demanda real
    fig.add_trace(
        go.Scatter(
            x=history["datetime_spain"],
            y=history["demand_mw"],
            mode="lines",
            name="Demanda real",
            line=dict(
                color="#58a6ff",
                width=3,
            ),
        )
    )

    # Predicción
    fig.add_trace(
        go.Scatter(
            x=[
                history["datetime_spain"].iloc[-1],
                *forecast["datetime_spain"].tolist(),
            ],
            y=[
                history["demand_mw"].iloc[-1],
                *forecast["predicted_demand_mw"].tolist(),
            ],
            mode="lines+markers",
            name="Predicción IA",
            line=dict(
                color="#b4f75a",
                width=3,
                dash="dash",
            ),
            marker=dict(size=7),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111923",
        plot_bgcolor="#111923",
        font=dict(color="#91a0af"),
        height=430,
        margin=dict(
            l=20,
            r=20,
            t=35,
            b=20,
        ),
        xaxis=dict(
            title="Fecha y hora (España)",
            showgrid=False,
        ),
        yaxis=dict(
            title="Demanda (MW)",
            gridcolor="#263341",
        ),
        legend=dict(
            orientation="h",
            y=1.15,
        ),
        hovermode="x unified",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )

    st.caption(
        "Datos observados: ENTSO-E · "
        "Predicciones: Random Forest · "
        "Zona horaria: Europe/Madrid"
    )

    # ========================================================
    # TABLA DE PREDICCIONES
    # ========================================================

    st.markdown(
        '<div class="section-heading">Predicciones por hora</div>',
        unsafe_allow_html=True,
    )

    
    table = forecast[
        ["datetime_spain", "predicted_demand_mw"]
    ].copy()

    table["datetime_spain"] = table[
        "datetime_spain"
    ].dt.strftime("%d/%m/%Y %H:%M")

    table.columns = [
        "Fecha y hora (España)",
        "Demanda prevista (MW)",
    ]

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Demanda prevista (MW)": st.column_config.NumberColumn(
                "Demanda prevista (MW)",
                format="%.2f",
            ),
        },
    )


    # Información adicional
    last_observation = pd.Timestamp(
        result["last_observation"]
    ).tz_convert("Europe/Madrid")

    target_date = forecast[
        "datetime_spain"
    ].iloc[-1]

    st.caption(
        f"Última observación: "
        f"{last_observation:%d/%m/%Y %H:%M} · "
        f"Objetivo de predicción: "
        f"{target_date:%d/%m/%Y %H:%M}"
    )

else:
    st.info(
        "Selecciona un modelo y un horizonte "
        "en el panel lateral y pulsa "
        "«Generar predicción» para visualizar "
        "los datos reales."
    )

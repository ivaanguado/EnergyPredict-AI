import sys
from pathlib import Path
from html import escape
from energypredict.storage.prediction_repository import (
    save_prediction,
)
from app.views.performance import render_performance
from app.views.auto_update import render_auto_update

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

from app.views.comparison import (
    render_comparison,
    get_comparison_max_horizon,
)

from predict import (
    predict_next_hour,
    get_recent_history,
    get_horizon_limits,
)


st.set_page_config(
    page_title="EnergyPredict AI | Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

css = (APP_DIR / "styles" / "main.css").read_text(
    encoding="utf-8"
)

st.markdown(
    f"<style>{css}</style>",
    unsafe_allow_html=True,
)


# ============================================================
# ICONOS
# ============================================================

ICONS = {
    "energy": '<path d="m13 2-9 12h7l-1 8 10-12h-7l1-8z"/>',

    "trend": '<path d="m3 17 6-6 4 4 8-8"/>'
             '<path d="M15 7h6v6"/>',

    "model": '<rect x="5" y="5" width="14" height="14" rx="2"/>'
             '<path d="M9 9h6v6H9z M9 2v3 M15 2v3 '
             'M9 19v3 M15 19v3 M2 9h3 M2 15h3 '
             'M19 9h3 M19 15h3"/>',

    "clock": '<circle cx="12" cy="12" r="9"/>'
             '<path d="M12 7v5l3 2"/>',
}


def svg_icon(name):
    return (
        '<svg width="22" height="22" viewBox="0 0 24 24" '
        'fill="none" stroke="currentColor" stroke-width="1.7" '
        'stroke-linecap="round" stroke-linejoin="round">'
        f'{ICONS.get(name, "")}</svg>'
    )


# ============================================================
# COMPONENTES
# ============================================================


def metric_card(
    label,
    value,
    unit="",
    icon="energy",
    description="",
    variant="neutral",
):
    html = (
        f'<div class="metric-card">'
        f'<div class="metric-top">'
        f'<div class="metric-label">{escape(str(label))}</div>'
        f'<div class="metric-icon {variant}">{svg_icon(icon)}</div>'
        f'</div>'
        f'<div class="metric-value">'
        f'{escape(str(value))} '
        f'<span>{escape(str(unit))}</span>'
        f'</div>'
        f'<div class="metric-description {variant}">'
        f'{escape(str(description))}'
        f'</div>'
        f'</div>'
    )

    st.html(html)


def format_number(value):
    return f"{value:,.0f}".replace(",", ".")


def format_percentage(value):
    return f"{value:+.2f}".replace(".", ",") + " %"


# ============================================================
# ESTADO DE LA APLICACIÓN
# ============================================================

METHOD_MAP = {
    "Recursivo": "recursive",
    "Directo": "direct",
    "Híbrido": "hybrid",
}

METHOD_LABELS = {
    value: key
    for key, value in METHOD_MAP.items()
}

DEFAULT_STATE = {
    "prediction_result": None,
    "last_update": None,
    "executed_horizon": None,
    "prediction_error": False,
    "recent_history": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Descargar y almacenar los datos recientes
@st.cache_data(ttl=300, show_spinner=False)
def load_recent_history():
    return get_recent_history()


if st.session_state.recent_history is None:
    with st.spinner("Consultando datos de ENTSO-E..."):
        try:
            st.session_state.recent_history = (
                load_recent_history()
            )
        except Exception:
            st.session_state.recent_history = None

# ============================================================
# BARRA LATERAL
# ============================================================

section = st.radio(
    "Sección",
    ["Predicción", "Comparativa", "Rendimiento"],
    horizontal=True,
    label_visibility="collapsed",
    key="active_section",
)

with st.sidebar:

    st.markdown(
        """
<div class="brand">
    ⚡ Energy<span>Predict</span>
    <div class="brand-subtitle">INTELLIGENCE PLATFORM</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.caption(section.upper())
    st.divider()

    refresh_data = st.button(
        "Actualizar datos de ENTSO-E",
        use_container_width=True,
    )

    if refresh_data:
        with st.spinner("Actualizando datos..."):
            try:
                load_recent_history.clear()

                st.session_state.recent_history = (
                    load_recent_history()
                )

            except Exception:
                st.warning(
                    "No se pudieron actualizar los datos."
                )

    recent_history = st.session_state.recent_history

    if section == "Predicción":
        method = st.selectbox(
            "Método de predicción",
            ["Recursivo", "Directo", "Híbrido"],
        )

        max_horizon = 0
        delay_hours = None

        if recent_history is not None and not recent_history.empty:
            limits = get_horizon_limits(
                recent_history,
                METHOD_MAP[method],
            )

            max_horizon = limits["max_horizon"]
            delay_hours = limits["delay_hours"]

        if max_horizon > 1:
            horizon = st.slider(
                "Horizonte de predicción",
                min_value=1,
                max_value=max_horizon,
                value=min(3, max_horizon),
                format="%d h",
            )

        elif max_horizon == 1:
            horizon = 1

            st.info(
                "Horizonte disponible: 1 hora."
            )

        else:
            horizon = None

            st.warning(
                "No hay un horizonte válido disponible "
                "para este método."
            )

        if delay_hours is not None:
            st.caption(
                f"Retraso de ENTSO-E: {delay_hours} h"
            )

            st.caption(
                f"Horizonte máximo disponible: "
                f"{max_horizon} h"
            )

        st.caption(
            "Horizonte contado desde la hora actual "
            "redondeada hacia abajo."
        )

        st.divider()

        history_period = st.selectbox(
            "Periodo histórico",
            ["24 horas", "48 horas", "7 días"],
            index=0,
        )

        HISTORY_PERIODS = {
            "24 horas": 24,
            "48 horas": 48,
            "7 días": 168,
        }

        history_hours = HISTORY_PERIODS[history_period]

        generate = st.button(
            "Generar predicción",
            type="primary",
            use_container_width=True,
            disabled=(max_horizon <= 0),
        )

    elif section == "Comparativa":
        if recent_history is not None and not recent_history.empty:
            limits = get_horizon_limits(recent_history, "direct")
            st.caption(f"Retraso de ENTSO-E: {limits['delay_hours']} h")
            common = get_comparison_max_horizon(recent_history)
            st.caption(f"Horizonte común máximo: {max(0, common)} h")
        else:
            st.warning("No hay datos disponibles de ENTSO-E.")
        st.caption("Configura el horizonte en el panel principal.")

    
    st.divider()

    st.markdown(
        """
<div class="sidebar-info">
<div class="sidebar-info-label">FUENTE DE DATOS</div>
<div class="sidebar-info-value">ENTSO-E Transparency Platform</div>
<div class="sidebar-info-label">TECNOLOGÍA</div>
<div class="sidebar-info-value">Random Forest · Machine Learning</div>
<div class="sidebar-info-label">REGIÓN</div>
<div class="sidebar-info-value">España</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.divider()
    render_auto_update()

    st.divider()
    st.caption("EnergyPredict AI · Versión 2.1")


if section == "Predicción":
    # ============================================================
    # GENERAR PREDICCIÓN
    # ============================================================

    if generate:

        with st.spinner(
            "Ejecutando el modelo con los datos disponibles..."
        ):
            try:
                new_result = predict_next_hour(
                    method=METHOD_MAP[method],
                    horizon=horizon,
                    return_data=True,
                    history=st.session_state.recent_history,
                )

                st.session_state.prediction_result = new_result

                try:
                    save_prediction(
                        result=new_result,
                        horizon=horizon,
                    )
                except Exception:
                    st.warning(
                        "La predicción se ha generado, pero no "
                        "se ha podido guardar en la base de datos."
                    )
                
                st.session_state.last_update = pd.Timestamp.now(
                    tz="Europe/Madrid"
                )

                st.session_state.executed_horizon = horizon
                st.session_state.prediction_error = False

            except Exception:
                st.session_state.prediction_error = True

    result = st.session_state.prediction_result
    last_update = st.session_state.last_update
    prediction_error = st.session_state.prediction_error


    # ============================================================
    # ENCABEZADO PRINCIPAL
    # ============================================================

    if prediction_error:
        status_text = "Error de actualización"
        status_class = "error"

    elif result is not None:
        status_text = "Datos recibidos"
        status_class = "online"

    else:
        status_text = "Pendiente de consulta"
        status_class = "waiting"


    update_text = (
        f"Actualizado: {last_update:%d/%m/%Y · %H:%M:%S}"
        if last_update is not None
        else "Sin actualizaciones"
    )


    st.markdown(
        f"""
    <div class="dashboard-topbar">
    <div class="eyebrow">ENERGY INTELLIGENCE PLATFORM</div>
    <div class="connection-status {status_class}">
    <span class="status-dot"></span>
    ENTSO-E · {status_text}
    </div>
    </div>
    <div class="hero">
    <h1>Predicción energética inteligente.</h1>
    <p>Monitorización de la demanda eléctrica en España mediante inteligencia artificial.</p>
    <div class="update-time">{update_text}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


    if prediction_error:
        st.error(
            "No se pudo completar la predicción. "
            "Comprueba la conexión con ENTSO-E o selecciona "
            "un horizonte menor. Si existen resultados anteriores, "
            "se conservarán en pantalla."
        )


    if result is not None:

        settings_changed = (
            result["method"] != METHOD_MAP[method]
            or st.session_state.executed_horizon != horizon
        )

        if settings_changed:
            st.info(
                "Has cambiado la configuración. "
                "Pulsa «Generar predicción» para actualizar "
                "los resultados mostrados."
            )


    # ============================================================
    # PREPARAR MÉTRICAS
    # ============================================================

    if result is not None:

        history = result["history"].sort_values(
            "datetime_utc"
        ).copy()

        forecast = result["forecast"].sort_values(
            "datetime_utc"
        ).copy()

        actual_demand = history.iloc[-1]["demand_mw"]

        predicted_demand = forecast.iloc[-1][
            "predicted_demand_mw"
        ]

        actual_value = format_number(actual_demand)
        predicted_value = format_number(predicted_demand)

        last_observation = pd.Timestamp(
            result["last_observation"]
        ).tz_convert("Europe/Madrid")

        target_date = pd.Timestamp(
            forecast.iloc[-1]["datetime_utc"]
        ).tz_convert("Europe/Madrid")

        displayed_method = METHOD_LABELS.get(
            result["method"],
            result["method"],
        )

        target_value = target_date.strftime("%H:%M")

        actual_description = (
            f"Último registro · {last_observation:%d/%m %H:%M}"
        )

        target_description = (
            f"{target_date:%d/%m} · "
            f"{result['steps']} pasos desde el último dato"
        )

        if actual_demand > 0:

            variation = (
                (predicted_demand - actual_demand)
                / actual_demand
            ) * 100

            if variation > 0:
                variation_class = "positive"
                variation_arrow = "↗"

            elif variation < 0:
                variation_class = "negative"
                variation_arrow = "↘"

            else:
                variation_class = "neutral"
                variation_arrow = "→"

            prediction_description = (
                f"{variation_arrow} "
                f"{format_percentage(variation)} "
                "vs. último registro"
            )

        else:
            variation_class = "neutral"
            prediction_description = "Variación no disponible"

    else:

        actual_value = "—"
        predicted_value = "—"
        displayed_method = "—"
        target_value = "—"

        actual_description = "Esperando datos de ENTSO-E"
        prediction_description = "Sin predicción generada"
        target_description = "Pendiente de cálculo"
        variation_class = "neutral"


    # ============================================================
    # TARJETAS DE MÉTRICAS
    # ============================================================

    col1, col2, col3, col4 = st.columns(
        4,
        gap="medium",
    )

    with col1:
        metric_card(
            label="Demanda observada",
            value=actual_value,
            unit="MW",
            icon="energy",
            description=actual_description,
            variant="blue",
        )

    with col2:
        metric_card(
            label="Demanda prevista",
            value=predicted_value,
            unit="MW",
            icon="trend",
            description=prediction_description,
            variant=variation_class,
        )

    with col3:
        metric_card(
            label="Motor de predicción",
            value=displayed_method,
            icon="model",
            description="Random Forest · Machine Learning",
            variant="accent",
        )

    with col4:
        metric_card(
            label="Hora objetivo",
            value=target_value,
            icon="clock",
            description=target_description,
            variant="neutral",
        )


    # ============================================================
    # GRÁFICA DE DEMANDA
    # ============================================================

    st.markdown(
        f"""
    <div class="section-heading">Evolución de la demanda</div>
    <div class="section-description">
    Histórico de las últimas {history_hours} horas y proyección del modelo.
    </div>
    """,
        unsafe_allow_html=True,
    )


    if result is not None:

        history = history.tail(history_hours).copy()

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

        # Demanda observada
        fig.add_trace(
            go.Scatter(
                x=history["datetime_spain"],
                y=history["demand_mw"],
                mode="lines",
                name="Demanda observada",
                line=dict(
                    color="#58a6ff",
                    width=3,
                ),
                hovertemplate="%{y:,.0f} MW<extra></extra>",
            )
        )

        # Conexión entre el histórico y la predicción
        fig.add_trace(
            go.Scatter(
                x=[
                    history["datetime_spain"].iloc[-1],
                    forecast["datetime_spain"].iloc[0],
                ],
                y=[
                    history["demand_mw"].iloc[-1],
                    forecast["predicted_demand_mw"].iloc[0],
                ],
                mode="lines",
                line=dict(
                    color="#b4f75a",
                    width=3,
                    dash="dash",
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Predicciones
        fig.add_trace(
            go.Scatter(
                x=forecast["datetime_spain"],
                y=forecast["predicted_demand_mw"],
                mode="lines+markers",
                name="Predicción IA",
                line=dict(
                    color="#b4f75a",
                    width=3,
                    dash="dash",
                ),
                marker=dict(
                    size=7,
                    color="#b4f75a",
                ),
                hovertemplate="%{y:,.0f} MW<extra></extra>",
            )
        )

        # Zona correspondiente a las predicciones
        fig.add_vrect(
            x0=history["datetime_spain"].iloc[-1],
            x1=forecast["datetime_spain"].iloc[-1],
            fillcolor="rgba(180,247,90,0.04)",
            line_width=0,
            layer="below",
        )

        # Separación entre datos reales y predicciones
        fig.add_shape(
            type="line",
            x0=history["datetime_spain"].iloc[-1],
            x1=history["datetime_spain"].iloc[-1],
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(
                color="#657485",
                width=1,
                dash="dot",
            ),
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111923",
            plot_bgcolor="#111923",
            font=dict(
                color="#91a0af",
                family="DM Sans, sans-serif",
            ),
            height=430,
            margin=dict(
                l=25,
                r=25,
                t=55,
                b=25,
            ),
            xaxis=dict(
                title="Hora local · España",
                showgrid=False,
                tickformat="%H:%M" if history_hours == 24 else "%d/%m %H:%M",
            ),
            yaxis=dict(
                title="Demanda (MW)",
                gridcolor="#263341",
                zeroline=False,
                range=[
                    min(
                        history["demand_mw"].min(),
                        forecast["predicted_demand_mw"].min(),
                    ) * 0.95,
                    max(
                        history["demand_mw"].max(),
                        forecast["predicted_demand_mw"].max(),
                    ) * 1.05,
                ],
            ),
            legend=dict(
                orientation="h",
                x=0,
                y=1.12,
            ),
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="#16212d",
                bordercolor="#263341",
                font_color="#f0f5f8",
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": False,
            },
        )

        st.caption(
            "Fuente: ENTSO-E · "
            "Modelo: Random Forest · "
            "Zona horaria: Europe/Madrid"
        )


        # ========================================================
        # TABLA DE PREDICCIONES
        # ========================================================

        st.markdown(
            """
<div class="section-heading">Predicciones por hora</div>
<div class="section-description">
Detalle de las estimaciones generadas por el modelo.
</div>
""",
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

        if "method" in forecast.columns:

            table["Método"] = forecast["method"].map(
                METHOD_LABELS
            ).fillna(forecast["method"])

        else:
            table["Método"] = displayed_method

        table["Tipo"] = "Intermedia"

        table.iloc[
            -1,
            table.columns.get_loc("Tipo")
        ] = "Objetivo"

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

        st.caption(
            f"Última observación: "
            f"{last_observation:%d/%m/%Y %H:%M} · "
            f"Objetivo: {target_date:%d/%m/%Y %H:%M}"
        )

    else:

        st.markdown(
            """
<div class="empty-state">
<div class="empty-icon">⚡</div>
<div class="empty-title">Tu próxima predicción empieza aquí</div>
<div class="empty-description">
Selecciona un método y un horizonte en el panel lateral
para consultar los datos energéticos y generar una predicción.
</div>
</div>
""",
            unsafe_allow_html=True,
        )


elif section == "Comparativa":
    render_comparison(st.session_state.recent_history)

else:
    render_performance()
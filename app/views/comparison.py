import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from predict import (
    predict_next_hour,
    get_horizon_limits,
)

from energypredict.storage.prediction_repository import (
    save_prediction,
)


COMPARISON_METHODS = {
    "recursive": "Recursivo",
    "direct": "Directo",
    "hybrid": "Híbrido",
}

METHOD_COLORS = {
    "recursive": "#58a6ff",
    "direct": "#b4f75a",
    "hybrid": "#a78bfa",
}

METHOD_LINE_STYLES = {
    "recursive": "solid",
    "direct": "solid",
    "hybrid": "dash",
}

def get_comparison_max_horizon(history):
    """Calcula el horizonte común de los tres métodos."""

    return min(
        get_horizon_limits(history, method)["max_horizon"]
        for method in COMPARISON_METHODS
    )


def generate_comparison(history, horizon):
    """Ejecuta los tres métodos con los mismos datos."""

    max_horizon = get_comparison_max_horizon(history)

    if not 1 <= horizon <= max_horizon:
        raise ValueError(
            f"El horizonte común máximo es de "
            f"{max_horizon} horas."
        )

    results = {}

    for method in COMPARISON_METHODS:
        results[method] = predict_next_hour(
            method=method,
            horizon=horizon,
            return_data=True,
            history=history,
        )

    # Comprobar que todos los métodos predicen
    # exactamente las mismas horas.
    reference = results["recursive"]["forecast"][
        "datetime_utc"
    ].reset_index(drop=True)

    for method, result in results.items():
        dates = result["forecast"][
            "datetime_utc"
        ].reset_index(drop=True)

        if not dates.equals(reference):
            raise ValueError(
                "Los métodos no comparten las mismas "
                "horas de predicción. Inténtalo de nuevo."
            )

    return results


def build_comparison_table(results):
    """Construye una tabla conjunta de predicciones."""

    table = None

    for method, label in COMPARISON_METHODS.items():

        forecast = results[method]["forecast"][
            ["datetime_utc", "predicted_demand_mw"]
        ].copy()

        forecast = forecast.rename(
            columns={
                "predicted_demand_mw": label,
            }
        )

        if table is None:
            table = forecast
        else:
            table = table.merge(
                forecast,
                on="datetime_utc",
                how="inner",
                validate="one_to_one",
            )

    columns = list(COMPARISON_METHODS.values())

    table["Diferencia máxima (MW)"] = (
        table[columns].max(axis=1)
        - table[columns].min(axis=1)
    )

    return table

def render_comparison(history):
    """Renderiza la interfaz de comparación de modelos."""

    st.markdown(
        """
<div class="section-heading">Comparativa de modelos</div>
<div class="section-description">
Compara los métodos recursivo, directo e híbrido
utilizando los mismos datos de entrada.
</div>
""",
        unsafe_allow_html=True,
    )

    if history is None or history.empty:
        st.warning(
            "No hay datos disponibles de ENTSO-E."
        )
        return

    max_horizon = get_comparison_max_horizon(history)

    if max_horizon == 0:
        st.warning(
            "El retraso de ENTSO-E impide realizar "
            "una comparación entre los tres métodos."
        )
        return

    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = None

    if "comparison_horizon" not in st.session_state:
        st.session_state.comparison_horizon = None

    if "comparison_history_snapshot" not in st.session_state:
        st.session_state.comparison_history_snapshot = None

    if "comparison_generated_at" not in st.session_state:
        st.session_state.comparison_generated_at = None

    # CONFIGURACIÓN

    col1, col2 = st.columns([2, 1])

    with col1:
        if max_horizon > 1:
            horizon = st.slider(
                "Horizonte de comparación",
                min_value=1,
                max_value=max_horizon,
                value=min(3, max_horizon),
                format="%d h",
                key="comparison_horizon_slider",
            )
        else:
            horizon = 1
            st.info("Horizonte disponible: 1 hora.")

    with col2:
        st.caption(
            f"Horizonte común máximo: {max_horizon} h"
        )

        compare = st.button(
            "Comparar modelos",
            type="primary",
            use_container_width=True,
            key="compare_models_button",
        )

    # EJECUCIÓN

    if compare:
        with st.spinner(
            "Ejecutando los tres métodos de predicción..."
        ):
            try:
                reference_hour = pd.Timestamp.now(tz="UTC").floor("h")

                st.session_state.comparison_results = (
                    generate_comparison(history, horizon)
                )

                st.session_state.comparison_horizon = horizon
                generated_at = pd.Timestamp.now(tz="UTC")

                try:
                    for result in st.session_state.comparison_results.values():
                        save_prediction(
                            result=result,
                            horizon=horizon,
                            generated_at=generated_at,
                        )
                except Exception:
                    st.warning(
                        "No se han podido guardar todas las "
                        "predicciones de la comparativa."
                    )
                st.session_state.comparison_history_snapshot = history.copy(deep=True)
                st.session_state.comparison_generated_at = reference_hour

            except Exception:
                st.session_state.comparison_results = None

                st.error(
                    "No se pudo completar la comparación. "
                    "Actualiza los datos e inténtalo de nuevo."
                )

    results = st.session_state.comparison_results

    if results is None:
        st.info(
            "Selecciona un horizonte y pulsa "
            "«Comparar modelos» para comenzar."
        )
        return

    # COMPROBAR VIGENCIA DE LOS RESULTADOS

    snapshot = st.session_state.comparison_history_snapshot

    settings_changed = (
        st.session_state.comparison_horizon != horizon
    )

    data_changed = (
        snapshot is None or not history.equals(snapshot)
    )

    time_changed = (
        st.session_state.comparison_generated_at
        != pd.Timestamp.now(tz="UTC").floor("h")
    )

    if settings_changed or data_changed or time_changed:
        st.warning(
            "Los resultados mostrados corresponden a una "
            "comparación anterior. Pulsa «Comparar modelos» "
            "para actualizarlos."
        )

    # RESULTADOS

    table = build_comparison_table(results)

    final_row = table.iloc[-1]

    target_date = pd.Timestamp(
        final_row["datetime_utc"]
    ).tz_convert("Europe/Madrid")

    st.caption(
        f"Horizonte ejecutado: "
        f"{st.session_state.comparison_horizon} h · "
        f"Hora objetivo: {target_date:%d/%m/%Y %H:%M}"
    )

    # TARJETAS DE MÉTRICAS

    columns = st.columns(3, gap="medium")

    for column, (method, label) in zip(
        columns,
        COMPARISON_METHODS.items(),
    ):
        value = final_row[label]
        color = METHOD_COLORS[method]

        formatted_value = (
            f"{value:,.0f}".replace(",", ".")
        )

        with column:
            st.markdown(
                f"""
<div class="metric-card">
<div class="metric-label" style="color:{color}">
{label}
</div>
<div class="metric-value">
{formatted_value} <span>MW</span>
</div>
<div class="metric-label">
Demanda prevista a las {target_date:%H:%M}
</div>
</div>
""",
                unsafe_allow_html=True,
            )

    # GRÁFICA COMPARATIVA

    st.markdown(
        '<div class="section-heading">'
        'Evolución comparada de las predicciones'
        '</div>',
        unsafe_allow_html=True,
    )

    table["datetime_spain"] = (
        pd.to_datetime(
            table["datetime_utc"],
            utc=True,
        ).dt.tz_convert("Europe/Madrid")
    )

    fig = go.Figure()

    for method, label in COMPARISON_METHODS.items():

        fig.add_trace(
            go.Scatter(
                x=table["datetime_spain"],
                y=table[label],
                name=label,
                mode="lines+markers",
                line=dict(
                    color=METHOD_COLORS[method],
                    width=3,
                    dash=METHOD_LINE_STYLES[method],
                ),
                marker=dict(size=7),
                hovertemplate=(
                    "%{y:,.2f} MW<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111923",
        plot_bgcolor="#111923",
        font=dict(color="#91a0af"),
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            title="Hora local · España",
            showgrid=False,
            tickformat="%H:%M",
        ),
        yaxis=dict(
            title="Demanda (MW)",
            gridcolor="#263341",
            rangemode="normal",
        ),
        legend=dict(
            orientation="h",
            y=1.12,
        ),
        hovermode="x unified",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
        key="comparison_chart",
    )

    # DIFERENCIAS ENTRE MODELOS

    st.markdown(
        '<div class="section-heading">'
        'Diferencias entre modelos'
        '</div>',
        unsafe_allow_html=True,
    )

    model_columns = list(
        COMPARISON_METHODS.values()
    )

    max_difference = (
        final_row[model_columns].max()
        - final_row[model_columns].min()
    )

    average_prediction = final_row[
        model_columns
    ].mean()

    percentage_difference = (
        max_difference / average_prediction * 100
        if average_prediction != 0
        else 0
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Diferencia máxima",
            f"{max_difference:.2f} MW",
        )

    with col2:
        st.metric(
            "Diferencia relativa",
            f"{percentage_difference:.2f} %",
        )

    with col3:
        st.metric(
            "Predicción media",
            f"{average_prediction:.2f} MW",
        )

    # TABLA COMPARATIVA

    st.markdown(
        '<div class="section-heading">'
        'Comparativa por hora'
        '</div>',
        unsafe_allow_html=True,
    )

    display_table = table[
        [
            "datetime_spain",
            "Recursivo",
            "Directo",
            "Híbrido",
            "Diferencia máxima (MW)",
        ]
    ].copy()

    display_table["datetime_spain"] = (
        display_table["datetime_spain"]
        .dt.strftime("%d/%m/%Y %H:%M")
    )

    display_table = display_table.rename(
        columns={
            "datetime_spain": "Fecha y hora (España)",
        }
    )

    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            column: st.column_config.NumberColumn(
                column,
                format="%.2f MW",
            )
            for column in [
                "Recursivo",
                "Directo",
                "Híbrido",
                "Diferencia máxima (MW)",
            ]
        },
    )

    st.caption(
        "Fuente: ENTSO-E · Random Forest · "
        "Comparación sobre los mismos datos históricos."
    )
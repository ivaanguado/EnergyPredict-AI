import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from energypredict.storage.prediction_repository import (
    get_predictions,
)

from energypredict.evaluation.metrics import (
    calculate_metrics,
)

from energypredict.evaluation.evaluator import (
    run_evaluation,
)

from predict import get_recent_history


METHOD_LABELS = {
    "recursive": "Recursivo",
    "direct": "Directo",
    "hybrid": "Híbrido",
}

METHOD_COLORS = {
    "recursive": "#58a6ff",
    "direct": "#b4f75a",
    "hybrid": "#a78bfa",
}


def format_metric(value, unit=""):

    if value is None or pd.isna(value):
        return "—"

    return f"{value:.2f} {unit}".strip()


def render_performance():

    st.markdown(
        """
<div class="section-heading">Rendimiento de los modelos</div>
<div class="section-description">
Evaluación de las predicciones frente a la demanda
eléctrica real publicada por ENTSO-E.
</div>
""",
        unsafe_allow_html=True,
    )

    # ========================================================
    # ACTUALIZAR EVALUACIONES
    # ========================================================

    if st.button(
        "Actualizar evaluaciones",
        type="primary",
        key="update_evaluations",
    ):

        with st.spinner(
            "Consultando ENTSO-E y evaluando predicciones..."
        ):

            try:
                history = get_recent_history()

                evaluation = run_evaluation(history)

                st.session_state.recent_history = history

                st.success(
                    f"Nuevas predicciones evaluadas: "
                    f"{evaluation['predictions_evaluated']}"
                )

            except Exception:
                st.error(
                    "No se pudieron actualizar las evaluaciones."
                )

    predictions = get_predictions()

    if predictions.empty:
        st.info("Todavía no hay predicciones registradas.")
        return

    predictions["generated_at_utc"] = pd.to_datetime(
        predictions["generated_at_utc"],
        utc=True,
        format="mixed",
    )

    predictions["target_datetime_utc"] = pd.to_datetime(
        predictions["target_datetime_utc"],
        utc=True,
        format="mixed",
    )

    # ========================================================
    # FILTROS
    # ========================================================

    st.markdown(
        '<div class="section-heading">Filtros de evaluación</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_method = st.selectbox(
            "Método",
            ["Todos", "Recursivo", "Directo", "Híbrido"],
            key="performance_method",
        )

    with col2:
        selected_horizon = st.selectbox(
            "Horizonte",
            ["Todos"] + list(range(1, 7)),
            key="performance_horizon",
        )

    with col3:
        selected_period = st.selectbox(
            "Periodo",
            [
                "Todo el histórico",
                "Últimas 24 horas",
                "Últimos 7 días",
                "Últimos 30 días",
            ],
            key="performance_period",
        )

    filtered = predictions.copy()

    if selected_method != "Todos":

        filtered = filtered[
            filtered["method"].map(METHOD_LABELS)
            == selected_method
        ]

    if selected_horizon != "Todos":

        filtered = filtered[
            filtered["horizon_hours"] == selected_horizon
        ]

    periods = {
        "Últimas 24 horas": 1,
        "Últimos 7 días": 7,
        "Últimos 30 días": 30,
    }

    if selected_period in periods:

        cutoff = (
            pd.Timestamp.now(tz="UTC")
            - pd.Timedelta(days=periods[selected_period])
        )

        filtered = filtered[
            filtered["generated_at_utc"] >= cutoff
        ]

    # ========================================================
    # RESUMEN DE PREDICCIONES
    # ========================================================

    total = len(filtered)

    evaluated = filtered[
        filtered["actual_demand_mw"].notna()
    ].copy()

    evaluated_count = len(evaluated)

    pending_count = total - evaluated_count

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Predicciones registradas", total)

    with col2:
        st.metric("Predicciones evaluadas", evaluated_count)

    with col3:
        st.metric("Predicciones pendientes", pending_count)

    if total > 0:

        st.progress(evaluated_count / total)

        st.caption(
            f"{evaluated_count / total * 100:.1f} % "
            "de las predicciones evaluadas"
        )

    # ========================================================
    # MÉTRICAS GLOBALES
    # ========================================================

    st.markdown(
        '<div class="section-heading">Métricas de precisión</div>',
        unsafe_allow_html=True,
    )

    metrics = calculate_metrics(filtered)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "MAE",
            format_metric(metrics["mae"], "MW"),
        )

    with col2:
        st.metric(
            "RMSE",
            format_metric(metrics["rmse"], "MW"),
        )

    with col3:
        st.metric(
            "MAPE",
            format_metric(metrics["mape"], "%"),
        )

    # ========================================================
    # COMPARATIVA DE RENDIMIENTO
    # ========================================================

    st.markdown(
        '<div class="section-heading">'
        'Precisión por método'
        '</div>',
        unsafe_allow_html=True,
    )

    if evaluated.empty:

        st.info(
            "Todavía no hay predicciones evaluadas "
            "para los filtros seleccionados."
        )

    else:

        method_results = []

        for method, label in METHOD_LABELS.items():

            subset = filtered[
                filtered["method"] == method
            ]

            result = calculate_metrics(subset)

            if result["count"] > 0:

                method_results.append({
                    "method": method,
                    "label": label,
                    **result,
                })

        comparison = pd.DataFrame(method_results)

        if not comparison.empty:

            selected_metric = st.selectbox(
                "Métrica de comparación",
                ["MAE", "RMSE", "MAPE"],
                key="performance_metric",
            )

            metric_key = selected_metric.lower()

            unit = (
                "%" if selected_metric == "MAPE"
                else "MW"
            )

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=comparison["label"],
                    y=comparison[metric_key],
                    marker_color=[
                        METHOD_COLORS[method]
                        for method in comparison["method"]
                    ],
                    text=comparison[metric_key],
                    texttemplate="%{text:.2f}",
                    textposition="outside",
                )
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#111923",
                plot_bgcolor="#111923",
                font=dict(color="#91a0af"),
                height=350,
                margin=dict(l=25, r=25, t=40, b=25),
                xaxis=dict(
                    title="Método",
                    showgrid=False,
                ),
                yaxis=dict(
                    title=f"{selected_metric} ({unit})",
                    gridcolor="#263341",
                    rangemode="tozero",
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
                key="performance_chart",
            )

            st.caption(
                "Un error menor indica mayor precisión. "
                "Los métodos pueden tener diferentes "
                "cantidades de predicciones evaluadas."
            )

            comparison_table = comparison[
                ["label", "count", "mae", "rmse", "mape"]
            ].rename(columns={
                "label": "Método",
                "count": "Evaluadas",
                "mae": "MAE (MW)",
                "rmse": "RMSE (MW)",
                "mape": "MAPE (%)",
            })

            st.dataframe(
                comparison_table,
                use_container_width=True,
                hide_index=True,
            )

    # ========================================================
    # HISTÓRICO DE PREDICCIONES
    # ========================================================

    st.markdown(
        '<div class="section-heading">'
        'Histórico de predicciones'
        '</div>',
        unsafe_allow_html=True,
    )

    table = filtered.copy()

    table["Estado"] = table["actual_demand_mw"].apply(
        lambda value: (
            "Pendiente" if pd.isna(value)
            else "Evaluada"
        )
    )

    table["Error absoluto (MW)"] = (
        table["actual_demand_mw"]
        - table["predicted_demand_mw"]
    ).abs()

    table["Error porcentual (%)"] = (
        table["Error absoluto (MW)"]
        / table["actual_demand_mw"].where(
            table["actual_demand_mw"] != 0
        )
    ) * 100

    table["method"] = table["method"].map(
        METHOD_LABELS
    )

    for column in [
        "generated_at_utc",
        "target_datetime_utc",
    ]:

        table[column] = (
            table[column]
            .dt.tz_convert("Europe/Madrid")
            .dt.strftime("%d/%m/%Y %H:%M")
        )

    table = table.rename(columns={
        "generated_at_utc": "Generada",
        "target_datetime_utc": "Hora objetivo",
        "method": "Método",
        "horizon_hours": "Horizonte (h)",
        "predicted_demand_mw": "Predicción (MW)",
        "actual_demand_mw": "Demanda real (MW)",
    })

    table = table[[
        "Generada",
        "Hora objetivo",
        "Método",
        "Horizonte (h)",
        "Predicción (MW)",
        "Demanda real (MW)",
        "Error absoluto (MW)",
        "Error porcentual (%)",
        "Estado",
    ]]

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            column: st.column_config.NumberColumn(
                column,
                format="%.2f",
            )
            for column in [
                "Predicción (MW)",
                "Demanda real (MW)",
                "Error absoluto (MW)",
                "Error porcentual (%)",
            ]
        },
    )

    st.caption(
        "Fuente: SQLite · ENTSO-E · "
        "Zona horaria: Europe/Madrid"
    )
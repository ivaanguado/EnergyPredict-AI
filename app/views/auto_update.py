import pandas as pd
import streamlit as st
import traceback

from predict import get_recent_history

from energypredict.automation.scheduler import (
    run_automatic_cycle,
)


UPDATE_INTERVAL = 300


@st.fragment(run_every=UPDATE_INTERVAL)
def render_auto_update():

    defaults = {
        "auto_update_enabled": True,
        "auto_update_last_run": None,
        "auto_update_last_success": None,
        "auto_update_status": "waiting",
        "auto_update_evaluated": 0,
        "auto_update_generated": 0,
        "auto_update_skipped": 0,
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value

    st.toggle(
        "Actualización automática",
        key="auto_update_enabled",
    )

    if not st.session_state.auto_update_enabled:

        st.caption(
            "Actualización automática desactivada."
        )

        return

    now = pd.Timestamp.now(tz="UTC")

    last_run = st.session_state.auto_update_last_run

    should_update = (
        last_run is None
        or (now - last_run).total_seconds()
        >= UPDATE_INTERVAL
    )

    if should_update:

        st.session_state.auto_update_last_run = now

        try:

            print("\n[AUTO] Iniciando ciclo automático", flush=True)

            # 1. Consultar ENTSO-E
            history = get_recent_history()

            print("[AUTO] Histórico obtenido correctamente", flush=True)

            # 2. Ejecutar evaluación y generación
            print("[AUTO] Ejecutando scheduler...", flush=True)

            results = run_automatic_cycle(
                history=history
            )

            print("[AUTO] Scheduler completado", flush=True)

            evaluation = results["evaluation"]
            generation = results["generation"]

            print(
                f"[AUTO] Generadas: {generation['saved_predictions']}",
                flush=True,
            )

            print(
                f"[AUTO] Evaluadas: {evaluation['predictions_evaluated']}",
                flush=True,
            )

            print(
                f"[AUTO] Duplicados: {generation['skipped_predictions']}",
                flush=True,
            )

            # 3. Actualizar datos de la sesión
            st.session_state.recent_history = history

            st.session_state.auto_update_evaluated = (
                evaluation["predictions_evaluated"]
            )

            st.session_state.auto_update_generated = (
                generation["saved_predictions"]
            )

            st.session_state.auto_update_skipped = (
                generation["skipped_predictions"]
            )

            st.session_state.auto_update_last_success = (
                pd.Timestamp.now(tz="UTC")
            )

            st.session_state.auto_update_status = "online"

        except Exception as error:

            print(
                f"[AUTO] Error: {type(error).__name__}",
                flush=True,
            )

            traceback.print_exc()

            st.session_state.auto_update_status = "error"

        else:
            print(
                "[AUTO] Estado actualizado correctamente",
                flush=True,
            )

            print(
                "[AUTO] Recargando dashboard...",
                flush=True,
            )

            st.rerun()

    # Estado del servicio
    if st.session_state.auto_update_status == "online":

        st.caption(
            "🟢 Actualización automática activa"
        )

    elif st.session_state.auto_update_status == "error":

        st.caption(
            "🔴 Error en la última actualización"
        )

        st.caption(
            "Se conservan los últimos datos válidos. "
            "Se reintentará en el siguiente ciclo."
        )

    else:

        st.caption(
            "🟡 Esperando actualización"
        )

    # Último intento
    last_run = st.session_state.auto_update_last_run

    if last_run is not None:

        local_time = last_run.tz_convert(
            "Europe/Madrid"
        )

        st.caption(
            f"Último intento: {local_time:%H:%M:%S}"
        )

    # Última actualización correcta
    last_success = (
        st.session_state.auto_update_last_success
    )

    if last_success is not None:

        local_success = last_success.tz_convert(
            "Europe/Madrid"
        )

        st.caption(
            f"Último éxito: {local_success:%H:%M:%S}"
        )

    st.caption("Frecuencia: cada 5 minutos")

    st.caption(
        "Generación: por ciclo horario"
    )

    st.markdown("**Último ciclo correcto**")

    st.caption(
        "Predicciones generadas: "
        f"{st.session_state.auto_update_generated}"
    )

    st.caption(
        "Predicciones evaluadas: "
        f"{st.session_state.auto_update_evaluated}"
    )

    st.caption(
        "Duplicados evitados: "
        f"{st.session_state.auto_update_skipped}"
    )
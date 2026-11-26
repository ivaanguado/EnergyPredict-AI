import pandas as pd

from predict import (
    get_recent_history,
    get_horizon_limits,
    predict_next_hour,
)

from energypredict.storage.prediction_repository import (
    save_prediction,
    get_automatic_horizons,
)

from energypredict.evaluation.evaluator import (
    run_evaluation,
)


METHODS = [
    "recursive",
    "direct",
    "hybrid",
]


def generate_automatic_predictions(history):
    """Genera los horizontes automáticos disponibles."""

    if history is None or history.empty:
        raise ValueError(
            "No hay datos históricos disponibles."
        )

    generated_at = pd.Timestamp.now(tz="UTC")

    generation_hour = generated_at.floor("h")

    saved_predictions = 0
    skipped_predictions = 0

    method_summary = {}

    for method in METHODS:

        limits = get_horizon_limits(
            history,
            method,
        )

        if limits["current_hour"] != generation_hour:
            break

        max_horizon = limits["max_horizon"]

        existing_horizons = get_automatic_horizons(
            generation_hour,
            method,
        )

        available_horizons = set(
            range(1, max_horizon + 1)
        )

        missing_horizons = sorted(
            available_horizons - existing_horizons
        )

        skipped = len(
            available_horizons & existing_horizons
        )

        skipped_predictions += skipped

        method_summary[method] = {
            "available": max_horizon,
            "saved": 0,
            "skipped": skipped,
        }

        if not missing_horizons:
            continue

        # Ejecutar el método una sola vez
        result = predict_next_hour(
            method=method,
            horizon=max(missing_horizons),
            return_data=True,
            history=history,
        )

        # Evitar mezclar ciclos si cambia la hora
        expected_target = (
            generation_hour
            + pd.Timedelta(hours=max(missing_horizons))
        )

        if result["target_hour"] != expected_target:
            break

        forecast = result["forecast"].sort_values(
            "datetime_utc"
        )

        for horizon in missing_horizons:

            target_hour = (
                generation_hour
                + pd.Timedelta(hours=horizon)
            )

            partial_forecast = forecast[
                forecast["datetime_utc"] <= target_hour
            ].copy()

            if partial_forecast.empty:
                continue

            if (
                partial_forecast.iloc[-1]["datetime_utc"]
                != target_hour
            ):
                continue

            steps = int(
                (
                    target_hour
                    - result["last_observation"]
                )
                / pd.Timedelta(hours=1)
            )

            partial_result = {
                **result,
                "forecast": partial_forecast,
                "steps": steps,
                "target_hour": target_hour,
            }

            prediction_id = save_prediction(
                result=partial_result,
                horizon=horizon,
                generated_at=generated_at,
                source="automatic",
            )

            if prediction_id is not None:

                saved_predictions += 1
                method_summary[method]["saved"] += 1

            else:

                skipped_predictions += 1
                method_summary[method]["skipped"] += 1

    return {
        "generation_hour_utc": generation_hour,
        "saved_predictions": saved_predictions,
        "skipped_predictions": skipped_predictions,
        "methods": method_summary,
    }


def run_automatic_cycle(history=None):
    """Ejecuta un ciclo completo de actualización."""

    if history is None:
        history = get_recent_history()

    evaluation = run_evaluation(history)

    generation = generate_automatic_predictions(
        history
    )

    return {
        "evaluation": evaluation,
        "generation": generation,
    }


if __name__ == "__main__":

    print("\nENERGYPREDICT AI")
    print("CICLO AUTOMÁTICO")
    print("=" * 50)

    results = run_automatic_cycle()

    evaluation = results["evaluation"]
    generation = results["generation"]

    print("\nRESUMEN DE EVALUACIÓN")
    print("-" * 50)

    print(
        "Observaciones procesadas:",
        evaluation["observations_saved"],
    )

    print(
        "Predicciones evaluadas:",
        evaluation["predictions_evaluated"],
    )

    print("\nRESUMEN DE GENERACIÓN")
    print("-" * 50)

    print(
        "Ciclo horario UTC:",
        generation["generation_hour_utc"],
    )

    print(
        "Nuevas predicciones:",
        generation["saved_predictions"],
    )

    print(
        "Predicciones duplicadas evitadas:",
        generation["skipped_predictions"],
    )

    print("\nRESULTADOS POR MÉTODO")
    print("-" * 50)

    for method, summary in generation["methods"].items():

        print(
            f"{method.capitalize()}: "
            f"{summary['available']} disponibles | "
            f"{summary['saved']} guardadas | "
            f"{summary['skipped']} omitidas"
        )
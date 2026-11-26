import pandas as pd

from energypredict.storage.prediction_repository import (
    save_observations,
    evaluate_pending_predictions,
    count_predictions,
    get_pending_predictions,
)


def run_evaluation(history):
    """Actualiza las observaciones y evalúa las predicciones."""

    if history is None or history.empty:
        raise ValueError(
            "No hay datos históricos disponibles."
        )

    observations_saved = save_observations(history)

    predictions_evaluated = evaluate_pending_predictions()

    total_predictions = count_predictions()

    pending_predictions = len(
        get_pending_predictions()
    )

    return {
        "observations_saved": observations_saved,
        "predictions_evaluated": predictions_evaluated,
        "total_predictions": total_predictions,
        "pending_predictions": pending_predictions,
        "total_evaluated": (
            total_predictions - pending_predictions
        ),
        "evaluated_at_utc": pd.Timestamp.now(tz="UTC"),
    }


if __name__ == "__main__":

    from predict import get_recent_history

    print("Consultando datos reales de ENTSO-E...")

    history = get_recent_history()

    results = run_evaluation(history)

    print("\nEvaluación completada correctamente.")
    print("-" * 45)

    print(
        "Observaciones reales procesadas:",
        results["observations_saved"],
    )

    print(
        "Nuevas predicciones evaluadas:",
        results["predictions_evaluated"],
    )

    print(
        "Total de predicciones:",
        results["total_predictions"],
    )

    print(
        "Total de predicciones evaluadas:",
        results["total_evaluated"],
    )

    print(
        "Predicciones pendientes:",
        results["pending_predictions"],
    )
import pandas as pd

from energypredict.storage.database import (
    get_connection,
    initialize_database,
)


def to_utc_string(value):
    """Normaliza una fecha al formato UTC."""

    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp.isoformat()


def save_prediction(
    result,
    horizon,
    generated_at=None,
    source="manual",
):
    """Guarda una predicción manual o automática."""

    if source not in ("manual", "automatic"):
        raise ValueError(
            "Origen de predicción no válido."
        )

    initialize_database()

    forecast = result["forecast"].sort_values(
        "datetime_utc"
    )

    final_prediction = forecast.iloc[-1]

    if generated_at is None:
        generated_at = pd.Timestamp.now(tz="UTC")

    generated_at = pd.Timestamp(
        to_utc_string(generated_at)
    )

    generation_hour = generated_at.floor("h")

    connection = get_connection()

    try:
        cursor = connection.execute(
            """
            INSERT INTO predictions (
                generated_at_utc,
                last_observation_utc,
                target_datetime_utc,
                method,
                horizon_hours,
                model_steps,
                predicted_demand_mw,
                source,
                generation_hour_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)

            ON CONFLICT (
                generation_hour_utc,
                method,
                horizon_hours
            )
            WHERE source = 'automatic'
            DO NOTHING
            """,
            (
                to_utc_string(generated_at),
                to_utc_string(result["last_observation"]),
                to_utc_string(
                    final_prediction["datetime_utc"]
                ),
                result["method"],
                int(horizon),
                int(result["steps"]),
                float(
                    final_prediction["predicted_demand_mw"]
                ),
                source,
                to_utc_string(generation_hour),
            ),
        )

        connection.commit()

        if cursor.rowcount == 0:
            return None

        return cursor.lastrowid

    finally:
        connection.close()


def get_predictions():
    """Recupera todas las predicciones registradas."""

    initialize_database()

    connection = get_connection()

    try:
        return pd.read_sql_query(
            """
            SELECT *
            FROM predictions
            ORDER BY generated_at_utc DESC, id DESC
            """,
            connection,
        )

    finally:
        connection.close()


def get_pending_predictions():
    """Recupera las predicciones pendientes de evaluación."""

    initialize_database()

    connection = get_connection()

    try:
        return pd.read_sql_query(
            """
            SELECT *
            FROM predictions
            WHERE actual_demand_mw IS NULL
            ORDER BY target_datetime_utc ASC
            """,
            connection,
        )

    finally:
        connection.close()


def count_predictions():
    """Devuelve el número de predicciones almacenadas."""

    initialize_database()

    connection = get_connection()

    try:
        result = connection.execute(
            "SELECT COUNT(*) FROM predictions"
        ).fetchone()

        return result[0]

    finally:
        connection.close()

def save_observations(history):
    """Guarda únicamente observaciones reales sin interpolaciones."""

    initialize_database()

    if "contains_interpolation" not in history.columns:
        raise ValueError(
            "No se puede verificar la calidad de las observaciones."
        )

    valid_history = history.loc[
        history["contains_interpolation"].eq(False)
    ].copy()

    retrieved_at = to_utc_string(
        pd.Timestamp.now(tz="UTC")
    )

    observations = []

    for row in valid_history.itertuples(index=False):

        if pd.isna(row.demand_mw):
            continue

        observations.append(
            (
                to_utc_string(row.datetime_utc),
                float(row.demand_mw),
                retrieved_at,
            )
        )

    connection = get_connection()

    try:
        connection.executemany(
            """
            INSERT INTO observations (
                datetime_utc,
                demand_mw,
                retrieved_at_utc
            )
            VALUES (?, ?, ?)

            ON CONFLICT(datetime_utc)
            DO UPDATE SET
                demand_mw = excluded.demand_mw,
                retrieved_at_utc = excluded.retrieved_at_utc
            """,
            observations,
        )

        connection.commit()

        return len(observations)

    finally:
        connection.close()


def evaluate_pending_predictions():
    """Asocia las predicciones con las observaciones reales."""

    initialize_database()

    evaluated_at = to_utc_string(
        pd.Timestamp.now(tz="UTC")
    )

    connection = get_connection()

    try:
        cursor = connection.execute(
            """
            UPDATE predictions

            SET actual_demand_mw = (
                SELECT observations.demand_mw
                FROM observations
                WHERE observations.datetime_utc =
                    predictions.target_datetime_utc
            ),

            evaluated_at_utc = ?

            WHERE actual_demand_mw IS NULL

            AND EXISTS (
                SELECT 1
                FROM observations
                WHERE observations.datetime_utc =
                    predictions.target_datetime_utc
            )
            """,
            (evaluated_at,),
        )

        connection.commit()

        return cursor.rowcount

    finally:
        connection.close()

def get_automatic_horizons(generation_hour, method):
    """Consulta los horizontes automáticos ya registrados."""

    initialize_database()

    generation_hour = pd.Timestamp(
        to_utc_string(generation_hour)
    ).floor("h")

    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT horizon_hours
            FROM predictions
            WHERE source = 'automatic'
            AND generation_hour_utc = ?
            AND method = ?
            """,
            (
                to_utc_string(generation_hour),
                method,
            ),
        ).fetchall()

        return {
            row["horizon_hours"]
            for row in rows
        }

    finally:
        connection.close()

if __name__ == "__main__":

    print("Repositorio de predicciones operativo.")
    print(f"Predicciones almacenadas: {count_predictions()}")
    print(
        f"Predicciones pendientes: "
        f"{len(get_pending_predictions())}"
    )
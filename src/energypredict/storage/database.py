import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATABASE_DIR = PROJECT_ROOT / "data" / "database"
DATABASE_PATH = DATABASE_DIR / "energypredict.db"


def get_connection():
    """Abre una conexión con la base de datos SQLite."""

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30,
    )

    connection.row_factory = sqlite3.Row

    connection.execute("PRAGMA foreign_keys = ON")

    return connection


def initialize_database():
    """Crea las tablas e índices necesarios."""

    connection = get_connection()

    try:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generated_at_utc TEXT NOT NULL,
                last_observation_utc TEXT NOT NULL,
                target_datetime_utc TEXT NOT NULL,
                method TEXT NOT NULL,
                horizon_hours INTEGER NOT NULL,
                model_steps INTEGER NOT NULL,
                predicted_demand_mw REAL NOT NULL,
                actual_demand_mw REAL,
                evaluated_at_utc TEXT,

                CHECK (
                    method IN (
                        'recursive',
                        'direct',
                        'hybrid'
                    )
                ),

                CHECK (horizon_hours > 0),
                CHECK (model_steps > 0)
            );

            CREATE INDEX IF NOT EXISTS idx_predictions_target
            ON predictions(target_datetime_utc);

            CREATE INDEX IF NOT EXISTS idx_predictions_method
            ON predictions(method);

            CREATE INDEX IF NOT EXISTS idx_predictions_pending
            ON predictions(actual_demand_mw);

            CREATE TABLE IF NOT EXISTS observations (
                datetime_utc TEXT PRIMARY KEY,
                demand_mw REAL NOT NULL,
                retrieved_at_utc TEXT NOT NULL
            );
        """)
        # Migración: identificar el origen de las predicciones
        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(predictions)"
            )
        }

        if "source" not in columns:
            connection.execute("""
                ALTER TABLE predictions
                ADD COLUMN source TEXT NOT NULL DEFAULT 'manual'
            """)

        if "generation_hour_utc" not in columns:
            connection.execute("""
                ALTER TABLE predictions
                ADD COLUMN generation_hour_utc TEXT
            """)

        # Evitar predicciones automáticas duplicadas
        connection.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS
                idx_predictions_automatic_unique
            ON predictions (
                generation_hour_utc,
                method,
                horizon_hours
            )
            WHERE source = 'automatic'
        """)
        connection.commit()

    finally:
        connection.close()


if __name__ == "__main__":
    initialize_database()

    print("Base de datos inicializada correctamente.")
    print(f"Ubicación: {DATABASE_PATH}")
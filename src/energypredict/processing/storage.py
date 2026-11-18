
from pathlib import Path


def save_parquet(df, output_path):
    """Guarda un DataFrame en formato Parquet."""

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        path,
        engine="pyarrow",
        index=False,
    )

    print(f"Datos guardados en: {path}")

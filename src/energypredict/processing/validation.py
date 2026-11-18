
import pandas as pd


def validate_demand(df, frequency="5min"):
    """Genera un informe básico de calidad de datos."""

    required = {"datetime_utc", "demand_value"}

    if not required.issubset(df.columns):
        raise ValueError("El DataFrame no tiene el esquema esperado.")

    if df.empty:
        raise ValueError("El DataFrame está vacío.")

    dates = df["datetime_utc"].dropna().sort_values()

    report = {
        "total_records": len(df),
        "null_dates": int(df["datetime_utc"].isna().sum()),
        "null_values": int(df["demand_value"].isna().sum()),
        "duplicate_dates": int(
            df["datetime_utc"].duplicated().sum()
        ),
        "negative_values": int(
            (df["demand_value"] < 0).sum()
        ),
        "zero_values": int(
            (df["demand_value"] == 0).sum()
        ),
    }

    if not dates.empty:
        unique_dates = pd.DatetimeIndex(dates.unique())

        expected = pd.date_range(
            start=unique_dates.min(),
            end=unique_dates.max(),
            freq=frequency,
        )

        report["missing_hours_inside_range"] = len(
            expected.difference(unique_dates)
        )
    else:
        report["missing_hours_inside_range"] = None

    report["passed_basic_checks"] = (
        report["null_dates"] == 0
        and report["null_values"] == 0
        and report["duplicate_dates"] == 0
        and report["negative_values"] == 0
        and report["missing_hours_inside_range"] == 0
    )

    return report

"""Rol 2: temel istatistikler ve IQR aykırı değer kontrolü."""
import pandas as pd
import numpy as np

from src.config import SENSOR_COLUMNS


def analyze_data(df):
    summary = {}
    iqr_counts = {}
    for col in SENSOR_COLUMNS:
        s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            summary[col] = {"mean": None, "median": None, "min": None,
                            "max": None, "std": None, "count": 0}
            iqr_counts[col] = 0
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr_counts[col] = int(((s < q1 - 1.5 * (q3 - q1)) |
                               (s > q3 + 1.5 * (q3 - q1))).sum())
        std = s.std()
        summary[col] = {
            "mean": round(float(s.mean()), 2),
            "median": round(float(s.median()), 2),
            "min": round(float(s.min()), 2),
            "max": round(float(s.max()), 2),
            "std": round(float(std), 2) if pd.notna(std) else None,
            "count": int(s.count()),
        }
    return {
        "record_count": len(df),
        "sensor_summary": summary,
        "iqr_outlier_counts": iqr_counts,
        "missing_values": {str(k): int(v) for k,v in df.isna().sum().items()},
    }

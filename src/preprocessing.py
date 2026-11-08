"""Rol 2: veri kalitesini inceleme ve modeli beslemek için hazırlık."""
import pandas as pd
import numpy as np

from src.config import SENSOR_COLUMNS, TIME_COLUMN, TARGET_COLUMN


def preprocess_data(df):
    """Veriyi inceleyip dönüştürür, satır ve anormallikleri SİLMEZ.

    Özellikle etiketli değerlendirme verisi üzerinde otomatik düzeltme
    yapılmaz; bu verideki sıra dışı değerler model için önemlidir.
    """
    cleaned = df.copy()
    original_missing = cleaned.isna().sum().to_dict()
    for col in SENSOR_COLUMNS:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
        cleaned[col] = cleaned[col].replace([np.inf, -np.inf], np.nan)

    dates = pd.to_datetime(cleaned[TIME_COLUMN], errors="coerce", utc=True)
    cleaned[TIME_COLUMN] = dates
    suspicious_dates = ((dates.dt.year < 2000) & dates.notna()).sum()
    # Bu veride is_valid gerçek bool; farklı CSV'den gelirse de kontrol et.
    invalid = cleaned["is_valid"].astype(str).str.lower().eq("false")
    report = {
        "original_rows": len(df),
        "final_rows": len(cleaned),
        "missing_by_column": {k: int(v) for k, v in cleaned.isna().sum().items()},
        "invalid_flag_count": int(invalid.sum()),
        "duplicate_row_count": int(df.duplicated().sum()),
        "duplicate_id_count": int(df["id"].duplicated().sum()),
        "invalid_timestamp_count": int(dates.isna().sum()),
        "suspicious_timestamp_count": int(suspicious_dates),
        "initial_missing_count": int(sum(original_missing.values())),
        "actions": [
            "Sensör sütunları sayısal formata çevrildi.",
            "Zaman sütunu tarih formatına çevrildi.",
            "Anormal ölçümler ve etiketler korundu; satır silinmedi.",
        ],
    }
    return cleaned, report


def prepare_training_data(df):
    """Sadece eğitim verisinden geçersiz (is_valid=False) kayıtları çıkarır."""
    if TARGET_COLUMN in df.columns:
        raise ValueError("Etiketli değerlendirme verisi eğitim için kullanılmamalıdır.")
    valid = ~df["is_valid"].astype(str).str.lower().eq("false")
    train_df = df.loc[valid].copy()
    for col in SENSOR_COLUMNS:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan)
    return train_df

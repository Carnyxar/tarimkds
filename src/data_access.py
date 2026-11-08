"""CSV dosyalarını tek yerden okuyoruz."""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FILES = {
    "raw": "Raw Dataset of Radish Microgreens.csv",
    "train": "Radish Microgreens Training Dataset.csv",
    "evaluation": "Labeled Evaluation Dataset of Radish Microgreens.csv",
    "unlabeled": "Unlabeled Evaluation Dataset of Radish Microgreens.csv",
}


def load_data(dataset="raw"):
    if dataset not in FILES:
        raise ValueError(f"Bilinmeyen veri seti: {dataset}")
    return pd.read_csv(DATA_DIR / FILES[dataset])

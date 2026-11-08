"""Rol 3: Isolation Forest ve iki karar ağacı (Gini / Entropy).

Karar ağaçları, eğitim verisinde gerçek etiket bulunmadığı için
SONRADAN ÜRETİLMİŞ örnek anormalliklerle öğrenir. Gerçek anomali
etiketleri yalnızca ayrı değerlendirme veri setinde kullanılır.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.config import SENSOR_COLUMNS, TARGET_COLUMN


def _features(df, medians):
    x = df[SENSOR_COLUMNS].apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    return x.fillna(medians)


def train_models(training_df):
    """Üç modeli eğitim verisinde oluşturur. Etiketli veri kullanılmaz."""
    if TARGET_COLUMN in training_df.columns:
        raise ValueError("Değerlendirme etiketlerini eğitimde kullanmayın.")
    source = training_df[SENSOR_COLUMNS].apply(pd.to_numeric, errors="coerce")
    source = source.replace([np.inf, -np.inf], np.nan)
    medians = source.median()
    if medians.isna().any():
        raise ValueError("Eğitim verisinde eksiksiz üç sensör sütunu gerekli.")
    rng = np.random.default_rng(42)
    normal = source.fillna(medians)
    # Uygulamanın normal bir bilgisayarda hızlı çalışması için örneklem kullanılır.
    normal = normal.sample(n=min(8000, len(normal)), random_state=42)
    if len(normal) < 50:
        raise ValueError("Model eğitimi için çok az kayıt var.")

    isolation = IsolationForest(n_estimators=100, contamination=0.07,
                                random_state=42, n_jobs=-1)
    isolation.fit(normal)

    # Kopyalanan normal kayıtlardan sadece eğitim için yapay uç değerler üretiriz.
    anomaly_count = int(len(normal) * 0.35)
    synthetic = normal.sample(n=anomaly_count, random_state=23).copy().reset_index(drop=True)
    limits = normal.quantile([0.05, 0.95])
    for row in range(len(synthetic)):
        col = rng.integers(0, len(SENSOR_COLUMNS))
        name = SENSOR_COLUMNS[col]
        low, high = limits.loc[0.05, name], limits.loc[0.95, name]
        spread = max(float(high - low), 1.0)
        if rng.random() < 0.5:
            synthetic.at[row, name] = float(low - 0.6 * spread)
        else:
            synthetic.at[row, name] = float(high + 0.6 * spread)

    x_train = pd.concat([normal.reset_index(drop=True), synthetic], ignore_index=True)
    y_train = np.r_[np.zeros(len(normal), dtype=int),
                    np.ones(len(synthetic), dtype=int)]

    gini = DecisionTreeClassifier(criterion="gini", max_depth=5,
                                  min_samples_leaf=20, random_state=42)
    entropy = DecisionTreeClassifier(criterion="entropy", max_depth=5,
                                     min_samples_leaf=20, random_state=42)
    gini.fit(x_train, y_train)
    entropy.fit(x_train, y_train)

    return {"Isolation Forest": isolation,
            "Karar Ağacı (Gini)": gini,
            "Karar Ağacı (Entropy)": entropy,
            "_medians": medians}


def predict_anomalies(models, df):
    """Her kayıt ve her model için tek satır sonuç döndürür.

    Sütunlar: record_id, is_anomaly, anomaly_score, model_name.
    Skorlar farklı modeller arasında doğrudan kıyaslanamaz.
    """
    x = _features(df, models["_medians"])
    output = []
    for name in ("Isolation Forest", "Karar Ağacı (Gini)", "Karar Ağacı (Entropy)"):
        model = models[name]
        if name == "Isolation Forest":
            predicted = model.predict(x) == -1
            score = -model.decision_function(x)
        else:
            predicted = model.predict(x) == 1
            score = model.predict_proba(x)[:, 1]
        output.append(pd.DataFrame({
            "record_id": df["id"].to_numpy(),
            "is_anomaly": predicted.astype(bool),
            "anomaly_score": score.astype(float),
            "model_name": name,
        }))
    return pd.concat(output, ignore_index=True)


def evaluate_models(models, evaluation_df, predictions=None):
    """Etiketli değerlendirme verisiyle accuracy/precision/recall/F1 hesaplar."""
    if TARGET_COLUMN not in evaluation_df:
        raise ValueError("Model karşılaştırması için ground_truth sütunu gerekli.")
    labels = evaluation_df[TARGET_COLUMN]
    allowed = {"Data Anomaly", "Data Normal"}
    if not set(labels.dropna().unique()).issubset(allowed) or labels.isna().any():
        raise ValueError("Beklenmeyen ground_truth etiketi bulundu.")
    if evaluation_df["id"].duplicated().any():
        raise ValueError("Değerlendirme verisinde tekrar eden id var.")
    if predictions is None:
        predictions = predict_anomalies(models, evaluation_df)
    y_true = labels.eq("Data Anomaly").astype(int).to_numpy()
    scores = []
    for name in ("Isolation Forest", "Karar Ağacı (Gini)", "Karar Ağacı (Entropy)"):
        sub = predictions.loc[predictions["model_name"].eq(name)]
        # ID ile eşleştir; CSV sırasının sabit kalmasına bel bağlama.
        pred_map = sub.set_index("record_id")["is_anomaly"]
        if len(sub) != len(evaluation_df) or not evaluation_df["id"].isin(pred_map.index).all():
            raise ValueError(f"{name}: tahminler ve değerlendirme kayıtları eşleşmiyor.")
        y_pred = pred_map.reindex(evaluation_df["id"]).astype(int).to_numpy()
        scores.append({
            "model_name": name,
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 3),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 3),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 3),
            "predicted_anomalies": int(y_pred.sum()),
            "tp": int(((y_pred == 1) & (y_true == 1)).sum()),
            "fp": int(((y_pred == 1) & (y_true == 0)).sum()),
            "fn": int(((y_pred == 0) & (y_true == 1)).sum()),
            "tn": int(((y_pred == 0) & (y_true == 0)).sum()),
        })
    return pd.DataFrame(scores)

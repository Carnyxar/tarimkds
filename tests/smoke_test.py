"""Hızlı test: python -m tests.smoke_test"""
from src.data_access import load_data
from src.preprocessing import preprocess_data, prepare_training_data
from src.statistics import analyze_data
from src.models import train_models, predict_anomalies, evaluate_models


def main():
    raw = load_data("raw")
    train = load_data("train")
    evaluation = load_data("evaluation")
    assert len(raw) == 33852
    assert len(train) == 27081
    assert len(evaluation) == 6771
    assert evaluation["ground_truth"].eq("Data Anomaly").sum() == 1354
    clean, report = preprocess_data(raw)
    assert len(clean) == len(raw)
    assert report["invalid_flag_count"] == 33
    prepared_train = prepare_training_data(train)
    assert len(prepared_train) == 27053
    assert analyze_data(clean)["record_count"] == 33852
    models = train_models(prepared_train)
    predictions = predict_anomalies(models, evaluation)
    assert len(predictions) == 3 * len(evaluation)
    assert set(predictions["model_name"]) == {"Isolation Forest", "Karar Ağacı (Gini)", "Karar Ağacı (Entropy)"}
    metrics = evaluate_models(models, evaluation, predictions)
    assert len(metrics) == 3
    assert (metrics["f1"].between(0, 1)).all()
    print("Testler başarılı. Model sonuçları:")
    print(metrics[["model_name", "accuracy", "precision", "recall", "f1"]].to_string(index=False))


if __name__ == "__main__":
    main()


def format_anomaly_result(record_id, is_anomaly, score, model_name):
    return {
        "record_id": int(record_id),
        "is_anomaly": bool(is_anomaly),
        "anomaly_score": float(score),
        "model_name": str(model_name)
    }

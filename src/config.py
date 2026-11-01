
# Modellerin kullanacağı sensör ölçümleri
SENSOR_COLUMNS = [
    "soil_moisture",
    "temperature",
    "humidity",
]

# Ölçüm zamanı
TIME_COLUMN = "device_timestamp"

# Değerlendirme verisindeki gerçek anormallik etiketi
TARGET_COLUMN = "ground_truth"

# Ücretsiz Gemini API için düşünülen model
LLM_MODEL = "gemini-2.5-flash-lite"

## Canlı Uygulama

TarımKDS, tarımsal sensör verilerini analiz etmek ve anormal ölçümleri tespit etmek amacıyla geliştirilmiştir.

Uygulamayı herhangi bir kurulum yapmadan tarayıcı üzerinden kullanabilirsiniz:

**[TarımKDS Web Uygulamasını Aç](https://tarimkds-am4mmurbfruee6p6nvfkji.streamlit.app/)**

Uygulama; veri analizi, veri kalitesi kontrolü, üç farklı anormallik tespit modelinin karşılaştırılması ve Gemini destekli Türkçe raporlama özelliklerini içermektedir.
# TarımKDS — Tarımsal Veri Analizi ve Karar Destek Sistemi

Veri Madenciliği dersi için küçük bir **Streamlit** uygulaması. Turp mikrofiliz (radish microgreens) sensör verilerini inceler, üç anormallik tespit modelini karşılaştırır ve istenirse Gemini ile Türkçe rapor üretir.

## Neler var?

- **Veri Paneli:** CSV dosyaları, eksik/geçersiz veri bilgisi, ortalama/medyan, IQR uç değer sayıları ve zaman grafikleri.
- **Anormallik Analizi:** Isolation Forest, Gini ve Entropy karar ağacı tahminleri, grafik ve CSV çıktısı.
- **Model Karşılaştırması:** Accuracy, precision, recall, F1 ve karışıklık matrisi sayıları.
- **AI Asistanı:** Gemini API anahtarı varsa gerçek model yanıtıyla Türkçe rapor ve soru-cevap.

## Çalıştırma

Windows'ta **`calistir.bat`** dosyasına çift tıklayın. İlk çalışmada internet bağlantısı ile paketler kurulur.

Alternatif olarak proje klasöründe terminal açın:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Tarayıcıda genellikle `http://localhost:8501` adresi açılır. Çalışmayı durdurmak için terminalde `Ctrl+C` basın.

## Gemini isteğe bağlıdır

Ücretsiz bir Gemini API anahtarı için Google AI Studio kullanılabilir. Ücretsiz kota ve model erişimi değişebilir. Windows PowerShell'de **aynı terminalde**, Streamlit'i başlatmadan önce:

```powershell
$env:GEMINI_API_KEY="KENDI_ANAHTARIN"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Anahtarı **GitHub'a yüklemeyin** ve başkalarıyla paylaşmayın. Model adı `src/config.py` içindeki `LLM_MODEL` değişkeninde; mevcut değilse o değişkeni erişilebilir bir Gemini modeliyle değiştirmek gerekir. API anahtarı yoksa ilk üç sekme çalışır, AI sekmesi bunu açıkça söyler. **Hazır metinler gerçek LLM yanıtıymış gibi gösterilmez.**

## Veri seti ve yöntem

Kaynak: [Mendeley Data — Dataset of Radish Microgreens](https://data.mendeley.com/datasets/3cppktpvwj/1).

- `data/Raw Dataset of Radish Microgreens.csv`: 33.852 kayıt.
- `data/Radish Microgreens Training Dataset.csv`: 27.081 kayıt. `is_valid=False` işaretli 28 kayıt eğitimden çıkarılır.
- `data/Labeled Evaluation Dataset of Radish Microgreens.csv`: 6.771 kayıt. Etiketler `Data Normal` ve `Data Anomaly`; ayrı değerlendirmede kullanılır.
- `data/Unlabeled Evaluation Dataset of Radish Microgreens.csv`: Etiketsiz alternatif veri.

Kullanılan sensörler: `soil_moisture`, `temperature`, `humidity`. Tarih alanı `device_timestamp`. `ground_truth` **eğitimde kullanılmaz**. Veri setindeki bazı değerlendirme anomalileri deneysel olarak eklenmiştir; gerçek bitki hastalığı veya sensör arızası teşhisi değildir.

- Isolation Forest, etiket kullanılmadan normal eğitim ölçümlerinin bir örneklemiyle öğrenir.
- Karar ağaçlarının eğitiminde **yapay olarak üretilmiş anormal örnekler** kullanılır; bu gerçek anomali etiketleriyle öğrenmek değildir.
- Üç modelin çıktısı aynı 6.771 satırlık etiketli değerlendirme verisinde ölçülür. Hiçbir değerlendirme kaydı otomatik silinmez.
- IQR istatistiksel uç değer bulur; modelin anormallik kararından farklıdır. Anomali skoru kalibre edilmiş risk olasılığı değildir.
- Model sonuçları tarımsal müdahale tavsiyesi veya kesin teşhis değildir.

## Dosyalar

| Dosya | İşlev |
|---|---|
| `app.py` | Streamlit arayüzü |
| `src/data_access.py` | CSV yükleme |
| `src/preprocessing.py` | Veri kalitesi ve eğitim verisi hazırlığı |
| `src/statistics.py` | Tanımlayıcı istatistikler |
| `src/models.py` | Üç model, tahminler, performans |
| `src/llm_service.py` | Gemini raporu ve soru-cevap |
| `src/config.py` | Ortak sütun/model adları |
| `tests/smoke_test.py` | API gerektirmeyen basit test |

## Kontrol

```powershell
.\.venv\Scripts\python.exe -m tests.smoke_test
```

Bu test veri boyutlarını, veri temizliğini ve üç modelin tahmin/performans çıktısını kontrol eder. Gemini için ayrıca kullanıcı anahtarıyla uygulamadaki rapor düğmesini denemek gerekir.

Bu proje eğitim amaçlıdır; gerçek zamanlı sensör sistemi veya ticari tarımsal karar mekanizması değildir.

## son.zip ile ilişki

Bu paket, kullanıcı tarafından gönderilen **son.zip** sürümündeki sol menü düzenini, kart biçimlerini ve dört sekmeli görsel düzeni esas alır. Eski örnek CSV ve notebook `orijinal_ornek/` altında yalnızca referans olarak korunmuştur. Çalışan uygulama bu dosyalara bağlı değildir. Eski Ollama ve hazır tavsiye metinleri kullanılmaz.

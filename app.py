"""TarımKDS: sensör verilerini inceleme ve üç anormallik modelini karşılaştırma.

Arayüzün görünümü ve açılır yan menü son.zip sürümünden korunmuştur.
Öğrenci projesidir; gerçek sera otomasyonu veya güvenilir tarımsal teşhis değildir.
"""
from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import SENSOR_COLUMNS, TIME_COLUMN
from src.data_access import load_data
from src.preprocessing import preprocess_data, prepare_training_data
from src.statistics import analyze_data
from src.models import train_models, predict_anomalies, evaluate_models
from src.llm_service import generate_report, answer_question

st.set_page_config(
    page_title="Tarımsal Karar Destek Sistemi",
    layout="wide",
    initial_sidebar_state="expanded",  # son.zip değişikliği
)

# Özel kartlarımız Streamlit'in CANLI CSS tema değişkenlerini kullanır.
# Böylece Settings > Theme değiştirildiğinde Python kodunun yeniden
# çalışması gerekmez: renkler tarayıcıda otomatik güncellenir.
# Önceki sürümdeki sorun, ilk açılışta hesaplanmış sabit renklerdi.
chart_colors = {
    "chart_line": "#8b9bb0",
    "chart_marker": "#73869f",
    "chart_alert": "#d43f3f",
}

# Görünümü Streamlit'in yerleşik bileşenleri yönetir. Sabit renkli CSS
# başlıkları/etiketleri koyu temada okunmaz yaptığı için kaldırıldı.

# Veri dosyaları her arayüz etkileşiminde yeniden diskten okunmasın.
@st.cache_data
def get_data(name: str):
    return load_data(name)


# Eğitim veri seti değişmediği sürece modeller bir kez eğitilsin.
@st.cache_resource(show_spinner="Anormallik modelleri hazırlanıyor...")
def get_models():
    return train_models(prepare_training_data(get_data("train")))


@st.cache_data(show_spinner="Değerlendirme sonuçları hesaplanıyor...")
def get_results():
    evaluation_data = get_data("evaluation")
    trained_models = get_models()
    predictions = predict_anomalies(trained_models, evaluation_data)
    metrics = evaluate_models(trained_models, evaluation_data, predictions)
    return predictions, metrics


def format_count(value):
    return f"{int(value):,}".replace(",", ".")


def make_chart_data(df, max_points=1400):
    """1970 gibi şüpheli tarihleri sadece grafikte atla; kaynak veriyi koru."""
    result = df.copy()
    result[TIME_COLUMN] = pd.to_datetime(result[TIME_COLUMN], errors="coerce", utc=True)
    years = result[TIME_COLUMN].dt.year
    current = pd.Timestamp.now(tz="UTC").year
    result = result[(years >= 2000) & (years <= current + 1)].sort_values(TIME_COLUMN)
    if len(result) > max_points:
        result = result.iloc[::max(1, len(result) // max_points)]
    return result


SENSOR_NAMES = {
    "temperature": "Sıcaklık (°C)",
    "humidity": "Hava nemi (%)",
    "soil_moisture": "Toprak nemi",
}
MODEL_NAMES = ["Isolation Forest", "Karar Ağacı (Gini)", "Karar Ağacı (Entropy)"]

try:
    raw = get_data("raw")
    train = get_data("train")
    evaluation = get_data("evaluation")
    predictions, metrics = get_results()
except (OSError, KeyError, ValueError, RuntimeError) as exc:
    st.error(f"Veri / model açılırken hata oluştu: {exc}")
    st.stop()

# -----------------------------------------------------------------------------
# YAN PANEL — son.zip'teki açılır menü ve sade filtre düzeni korundu.
# -----------------------------------------------------------------------------
st.sidebar.markdown("### Veri ve Model Parametreleri")
data_choice = st.sidebar.radio(
    "Görüntülenecek veri",
    ["Etiketli değerlendirme", "Ham veri", "Eğitim verisi"],
)
source_data = {
    "Etiketli değerlendirme": evaluation,
    "Ham veri": raw,
    "Eğitim verisi": train,
}[data_choice]
processed, quality_report = preprocess_data(source_data)

st.sidebar.markdown("---")
st.sidebar.markdown("### Filtreler")
ids = sorted(processed["device_id"].dropna().astype(str).unique().tolist())
selected_device = st.sidebar.selectbox("Cihaz", ["Tümü"] + ids)
filtered = processed.copy()
if selected_device != "Tümü":
    filtered = filtered[filtered["device_id"].astype(str).eq(selected_device)]

# Tarih aralığı filtresi yalnızca tarihleri okunabilen kayıtları kapsar.
# İlk ekranda tüm tarihleri (örneğin veri setindeki 1970 tarihlerini) saklarız.
st.sidebar.caption("Grafiklerde şüpheli 1970 tarihleri gösterilmez; kaynak veri silinmez.")
use_dates = st.sidebar.checkbox("Tarih aralığına göre filtrele", value=False)
if use_dates:
    valid_times = make_chart_data(filtered)
    if not valid_times.empty:
        earliest = valid_times[TIME_COLUMN].min().date()
        latest = valid_times[TIME_COLUMN].max().date()
        dates = st.sidebar.date_input(
            "Başlangıç ve bitiş",
            value=(earliest, latest), min_value=earliest, max_value=latest
        )
        if len(dates) == 2:
            start = pd.Timestamp(dates[0], tz="UTC")
            stop = pd.Timestamp(dates[1], tz="UTC") + pd.Timedelta(days=1)
            filtered = filtered[(filtered[TIME_COLUMN] >= start) & (filtered[TIME_COLUMN] < stop)]
    else:
        st.sidebar.info("Bu cihaz için geçerli tarih bulunamadı.")

st.sidebar.markdown("---")
st.sidebar.markdown("### Model")
model_selected = st.sidebar.selectbox("Anormallik modeli", MODEL_NAMES)
st.sidebar.caption(
    "Üç model ayrı eğitim verisinde hazırlanır. Buradaki seçim modeli yeniden eğitmez."
)
st.sidebar.caption("Kaynak: Mendeley Data — Radish Microgreens")

# Değerlendirme tahminleri önceden hesaplanır. Filtrelemeyi sadece sonuçları
# görselleştirmek için uygularız; F1 gibi metrikler tüm eval verisinden gelir.
selected_predictions = predictions[predictions["model_name"].eq(model_selected)].copy()
inspected_all = evaluation.merge(
    selected_predictions, left_on="id", right_on="record_id", validate="one_to_one"
)
inspected, _ = preprocess_data(inspected_all)
if selected_device != "Tümü":
    inspected = inspected[inspected["device_id"].astype(str).eq(selected_device)]
if use_dates and "dates" in locals() and len(dates) == 2:
    inspected = inspected[(inspected[TIME_COLUMN] >= start) & (inspected[TIME_COLUMN] < stop)]

selected_anomalies = inspected[inspected["is_anomaly"]].copy()
stats = analyze_data(filtered)

# Tek veri özeti, hem üstteki hızlı yardımda hem AI sekmesinde kullanılır.
metric_row = metrics[metrics["model_name"].eq(model_selected)].iloc[0]
top = selected_anomalies.sort_values("anomaly_score", ascending=False).head(8)
stat_eval = analyze_data(inspected)
examples = top[["id", *SENSOR_COLUMNS]].rename(columns={"id": "record_id"}).to_dict(orient="records")
analysis = {
    "dataset_name": "Radish Microgreens — etiketli değerlendirme",
    "selected_device": selected_device,
    "record_count": int(len(inspected)),
    "anomaly_count": int(len(selected_anomalies)),
    "anomaly_rate_pct": round(100 * len(selected_anomalies) / len(inspected), 2) if len(inspected) else 0,
    "sensor_summary": stat_eval["sensor_summary"],
    "model_name": model_selected,
    "model_metrics_full_evaluation": {
        "accuracy": float(metric_row["accuracy"]),
        "precision": float(metric_row["precision"]),
        "recall": float(metric_row["recall"]),
        "f1": float(metric_row["f1"]),
    },
    "anomaly_examples": examples,
}

# -----------------------------------------------------------------------------
# ANA SAYFA — son.zip düzenindeki başlık, dört KPI kartı ve dört sekme
# -----------------------------------------------------------------------------
col_head, col_help = st.columns([4, 1.2])
with col_head:
    st.title("Tarımsal Sensör Karar Destek ve Anomali Analiz Sistemi")
    st.caption("Radish Microgreens sensör verilerinde istatistik, makine öğrenmesi ve Gemini raporları")
with col_help:
    st.markdown("<div style='padding-top:0.65rem;'></div>", unsafe_allow_html=True)
    with st.popover("Akıllı Yardım (Gemini)", width="stretch"):
        st.caption("Yalnızca eldeki sonuçlarla ilgili soru sorabilirsiniz.")
        quick_question = st.text_input("Sorunuzu yazın", key="quick_question")
        if st.button("Soruyu yanıtla", key="quick_ask", width="stretch"):
            if not quick_question.strip():
                st.warning("Önce bir soru yazın.")
            else:
                try:
                    with st.spinner("Gemini yanıtı hazırlanıyor..."):
                        st.markdown(answer_question(quick_question, analysis))
                except (RuntimeError, ValueError) as exc:
                    st.error(str(exc))

# Veri paneli için ham/kayıt sayısı, değerlendirme için model anomali sayısı.
summary_kpis = [
    ("Seçili Veri", format_count(len(filtered)), data_choice),
    ("Cihaz", selected_device, "Seçili istasyon"),
    ("Değerlendirme Kaydı", format_count(len(inspected)), "Seçili cihaz / tarih"),
    ("Anormallik", format_count(len(selected_anomalies)), model_selected),
]
kpi_cols = st.columns(4)
for column, (label, value, note) in zip(kpi_cols, summary_kpis):
    with column:
        st.metric(label, value)
        st.caption(note)

st.caption("Değerlendirme etiketleri eğitim için kullanılmaz; özet metrikler tüm değerlendirme setinden hesaplanır.")
tab_ts, tab_eda, tab_prep, tab_action = st.tabs([
    "Sensör İzleme", "İstatistiksel Analiz", "Veri Kalitesi ve Modeller", "AI Asistanı"
])

with tab_ts:
    st.markdown("##### Sensör Ölçümleri ve Anormallikler")
    col1, col2 = st.columns([1.2, 2])
    with col1:
        graph_col = st.selectbox("İncelenecek sensör", SENSOR_COLUMNS,
                                 format_func=lambda x: SENSOR_NAMES[x], key="sensor_plot")
    with col2:
        st.caption("Kırmızı işaretler seçili modelin anormal gördüğü değerlendirme kayıtlarıdır.")
    graph = make_chart_data(inspected)
    if graph.empty:
        st.info("Seçili filtrelerde çizilebilecek değerlendirme kaydı bulunamadı.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=graph[TIME_COLUMN], y=graph[graph_col], mode="lines+markers",
            name="Sensör ölçümü", marker=dict(size=3, color=chart_colors["chart_marker"]),
            line=dict(color=chart_colors["chart_line"], width=1.5),
        ))
        # Nokta örneklemesi bazı anomalileri kaçırmasın: işaretleri ayrıca al.
        marked = make_chart_data(selected_anomalies, max_points=700)
        if not marked.empty:
            fig.add_trace(go.Scatter(
                x=marked[TIME_COLUMN], y=marked[graph_col], mode="markers",
                name="Anormallik", marker=dict(color=chart_colors["chart_alert"], size=8),
            ))
        fig.update_layout(
            height=410,
            margin=dict(l=35, r=15, t=20, b=30),
            xaxis_title="Zaman", yaxis_title=SENSOR_NAMES[graph_col],
        )
        st.plotly_chart(fig, width="stretch")
    st.markdown("##### İşaretlenen Kayıtlar")
    st.caption("Anormallik skoru bir olasılık değildir. Modeller arası skorlar doğrudan karşılaştırılmaz.")
    show_cols = ["id", "device_id", TIME_COLUMN, *SENSOR_COLUMNS, "anomaly_score"]
    st.dataframe(selected_anomalies[show_cols].head(50), width="stretch", hide_index=True)
    st.download_button(
        "Seçilen modelin tahminlerini CSV olarak indir",
        data=inspected[["id", "device_id", *SENSOR_COLUMNS, "ground_truth", "is_anomaly", "anomaly_score"]]
            .to_csv(index=False).encode("utf-8-sig"),
        file_name="tarimkds_anormallik_sonuclari.csv", mime="text/csv",
    )

with tab_eda:
    st.markdown("##### Sensörlerin Temel İstatistikleri")
    table = pd.DataFrame(stats["sensor_summary"]).T.reset_index().rename(columns={
        "index": "Sensör", "mean": "Ortalama", "median": "Medyan",
        "min": "En düşük", "max": "En yüksek", "std": "Std. sapma",
        "count": "Ölçüm sayısı"})
    table["Sensör"] = table["Sensör"].map(SENSOR_NAMES)
    st.dataframe(table, width="stretch", hide_index=True)
    st.caption("IQR sınırı dışında kalan ölçümler yalnızca istatistiksel uç değer adayıdır.")
    st.write("IQR uç değer sayıları:", {SENSOR_NAMES[k]:v for k,v in stats["iqr_outlier_counts"].items()})
    a,b = st.columns(2)
    with a:
        st.markdown("##### Korelasyon")
        if len(filtered) > 1:
            corr = filtered[SENSOR_COLUMNS].corr().round(2)
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="Blues", zmin=-1,zmax=1)
            fig.update_layout(height=320, margin=dict(l=20,r=20,t=25,b=20))
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Korelasyon için yeterli ölçüm bulunamadı.")
    with b:
        st.markdown("##### Ölçüm Dağılımı")
        if not filtered.empty:
            fig = px.box(filtered, y="temperature", points="outliers", title="Sıcaklık dağılımı")
            fig.update_layout(height=320, margin=dict(l=20,r=20,t=25,b=20))
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Bu filtrede ölçüm bulunamadı.")

with tab_prep:
    st.markdown("##### Veri Kalitesi")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Toplam satır", format_count(quality_report["original_rows"]))
    c2.metric("Geçersiz işaretli", quality_report["invalid_flag_count"])
    c3.metric("Şüpheli tarih", quality_report["suspicious_timestamp_count"])
    c4.metric("Tekrarlanan ID", quality_report["duplicate_id_count"])
    st.caption(
        "Sayısal dönüşüm uygulandı. Otomatik silme, anormal ölçüm düzeltme veya interpolasyon yapılmadı; "
        "değerlendirme kayıtları korunuyor."
    )
    st.markdown("##### Veri Ön İşleme Özeti")
    st.dataframe(pd.DataFrame([
        {"İşlem": "Ham satır sayısı", "Sonuç": quality_report["original_rows"]},
        {"İşlem": "İşlem sonrası satır sayısı", "Sonuç": quality_report["final_rows"]},
        {"İşlem": "Eksik hücre", "Sonuç": sum(quality_report["missing_by_column"].values())},
        {"İşlem": "Tekrarlanan tüm satır", "Sonuç": quality_report["duplicate_row_count"]},
        {"İşlem": "Okunamayan tarih", "Sonuç": quality_report["invalid_timestamp_count"]},
    ]), width="stretch", hide_index=True)
    st.markdown("##### Üç Modelin Başarı Karşılaştırması")
    metrics_show = metrics[["model_name", "accuracy", "precision", "recall", "f1", "predicted_anomalies"]].copy()
    metrics_show.columns = ["Model", "Accuracy", "Precision", "Recall", "F1", "Anormal işareti"]
    st.dataframe(metrics_show, width="stretch", hide_index=True)
    melted = metrics.melt(id_vars="model_name", value_vars=["precision", "recall", "f1"],
                          var_name="Ölçüt", value_name="Değer")
    fig = px.bar(melted, x="model_name", y="Değer", color="Ölçüt", barmode="group",
                 range_y=[0,1], title="Model değerlendirme sonuçları")
    fig.update_layout(xaxis_title="Model", height=350)
    st.plotly_chart(fig, width="stretch")
    st.info(
        "Isolation Forest etiketsiz öğrenir. Gini ve Entropy ağaçları eğitim için üretilen yapay "
        "anormalliklerle öğrenir. Gerçek ground_truth etiketleri yalnızca değerlendirmede kullanılmıştır."
    )
    with st.expander("TP / FP / FN / TN sayılarını göster"):
        st.dataframe(metrics[["model_name", "tp", "fp", "fn", "tn"]],
                     hide_index=True, width="stretch")
    st.download_button("Temizlenmiş seçili veriyi CSV olarak indir",
        data=filtered.to_csv(index=False).encode("utf-8-sig"),
        file_name="tarimkds_islenmis_veri.csv", mime="text/csv")

with tab_action:
    st.markdown("##### Gemini ile Türkçe Rapor ve Soru-Cevap")
    st.caption("Gemini sensör arızası teşhisi koymaz; istatistikleri ve model çıktılarını Türkçe açıklar.")
    st.caption("API'ye bütün CSV yerine bu filtreye ait özet ve en fazla 8 örnek gönderilir.")
    if not os.getenv("GEMINI_API_KEY"):
        st.warning("Gemini API anahtarı tanımlanmamış. Diğer ekranlar anahtarsız çalışır.")
    if st.button("Türkçe rapor oluştur", key="run_report"):
        try:
            with st.spinner("Gemini raporu hazırlanıyor..."):
                st.session_state["report"] = (model_selected, selected_device, generate_report(analysis))
        except (RuntimeError, ValueError) as exc:
            st.error(str(exc))
    existing = st.session_state.get("report")
    if existing and existing[0:2] == (model_selected, selected_device):
        st.markdown(existing[2])
    question = st.text_input("Analiz hakkında sorunuzu yazın", key="llm_question",
                             placeholder="Bu veride kaç kayıt anormal işaretlenmiş?")
    if st.button("Soruyu yanıtla", key="run_answer"):
        try:
            with st.spinner("Yanıt bekleniyor..."):
                st.markdown(answer_question(question, analysis))
        except (RuntimeError, ValueError) as exc:
            st.error(str(exc))
    st.caption("Yapay zekâ yanıtları doğrulanmalı; otomatik sulama veya havalandırma kararı değildir.")

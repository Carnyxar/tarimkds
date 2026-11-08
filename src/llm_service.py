"""Rol 5: Gemini ile raporlama; tespit ve istatistik hesabı burada YAPILMAZ."""
import os
import json

from src.config import LLM_MODEL


def _ask_gemini(instruction, analysis):
    if not isinstance(analysis, dict):
        raise ValueError("Analiz özeti sözlük (dict) olmalı.")
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Gemini API anahtarı ayarlı değil (GEMINI_API_KEY).")
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        text = json.dumps(analysis, ensure_ascii=False, default=str)
        response = client.models.generate_content(
            model=LLM_MODEL, contents=instruction + "\n\nANALİZ ÖZETİ:\n" + text
        )
        answer = (response.text or "").strip()
        if not answer:
            raise RuntimeError("Gemini boş yanıt gönderdi.")
        return answer
    except RuntimeError:
        raise
    except Exception as exc:
        # Asıl API key veya erişim bilgilerini göstermeden anlaşılır hata.
        raise RuntimeError("Gemini yanıtı alınamadı. API anahtarı, kota veya bağlantıyı kontrol edin.") from exc


def generate_report(analysis: dict) -> str:
    return _ask_gemini(
        "Sen tarım sensörleri için veri analizi sonuçlarını Türkçe anlatan bir asistansın. "
        "Yalnızca verilen özete dayan. Sıcaklık, nem ve anomalileri kısa başlıklarla açıkla. "
        "Bir değerin neden anormal olduğuna ilişkin kanıtsız kesin açıklama yapma. "
        "Bitkiye özel eşik ve sulama miktarı uydurma. "
        "Skorları kalibre edilmiş risk olasılığı diye sunma. "
        "Bu bir öğrenci projesi, kısa ve anlaşılır yaz.",
        analysis,
    )


def answer_question(question: str, analysis: dict) -> str:
    if not question.strip():
        raise ValueError("Soru boş olamaz.")
    return _ask_gemini(
        "Sen bir tarım veri analizi yardımcısısın. "
        "Yalnızca verilen veri özetinden yanıt çıkar. Veride cevap yoksa açıkça söyle, "
        "bilgi uydurma ve kesin tarımsal müdahale önerme. "
        "Kullanıcı sorusu: " + question,
        analysis,
    )

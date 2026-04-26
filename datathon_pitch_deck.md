# DataLeague Datathon Pitch Deck

## 1. Problem
- Amaç: etiket olmadan organik ve manipülatif sosyal medya içeriklerini ayırmak.
- Çıktılar: organiklik/risk skoru, manipülasyon haritası ve canlı inference fonksiyonu.

---

## 2. Veri
- Dosya: `dataset/datathonFINAL.parquet`
- Kolonlar: metin, anahtar kelimeler, sentiment, emotion, tema, dil, platform URL, author hash, tarih.
- Kısıt: gerçek dünya sosyal medya verisi çok dilli ve gürültülü; bot/human etiketi yok.

---

## 3. Yöntem
- Gözetimsiz, açıklanabilir risk skorlama.
- Üç bileşen: içerik riski, author davranış riski, koordinasyon riski.
- Final skor; content, author ve coordination kanıtlarını birleştirir, metadata eksikse ağırlıkları yeniden normalize eder.
- Risk bantları: High (`>=0.65`), Review (`>=0.45`), Low.

---

## 4. Feature'lar
- İçerik: boş metin, düşük kanıtlı kısa metin, token tekrarı, link/hashtag/mention yoğunluğu, büyük harf, noktalama, uç sentiment.
- Author: yüksek hacim, burst posting, tekrar eden anlatılar, düşük konu çeşitliliği.
- Koordinasyon: kısa zaman pencerelerinde birden fazla author/platform üzerinde görülen generic olmayan keyword/text signature'ları.
- Güven: author yayılımı, kısa zaman penceresi, cluster boyutu ve exact narrative-match gücü.

---

## 5. Manipülasyon Haritası
- Üretim komutu: `python3 datathon_pipeline.py run --sample-size 200000`.
- Ana artifact: `artifacts/risk_map_language_platform.png`.
- Destekleyici tablolar: `risk_map_language_platform.csv`, `top_risk_segments.csv`, `time_spikes.csv`, `platform_normalized_risk.csv`.
- Destekleyici görseller: `risk_score_histogram.png`, `platform_normalized_risk.png`, `top_risk_authors.png`, `risk_funnel.png`.

---

## 6. Bulgular
- Bu slaytta `artifacts/top_risk_segments.csv` içindeki en üst satırlar kullanılmalı.
- Şüpheli dil/platform/tema kombinasyonları gösterilmeli ve baskın reason code'lar açıklanmalı.

---

## 7. Koordinasyon Cluster'ları
- Bu slaytta `artifacts/top_coordination_clusters.csv` içindeki örnekler kullanılmalı.
- Somut ve boş olmayan şüpheli gönderiler için `artifacts/presentation_examples.csv` kullanılmalı.
- Vurgu: aynı narrative signature, çoklu author, kısa zaman penceresi ve cluster confidence score.

---

## 8. Canlı Demo
- Fonksiyon: `predict_live(text, language=None, url=None, author_hash=None, date=None, english_keywords=None)`.
- Çıktı: label, risk score, organic score, top reasons, used features, psychological triggers.
- Senaryo bazlı hazırlık kontrolü için `artifacts/live_inference_benchmark.csv` kullanılmalı.
- Sınır: metadata güveni artırır; text-only tahminler içerik feature'larına dayanır.

---

## 9. Pazarlama Kanıtları
- Kısa jüri özeti için `artifacts/marketing_scorecard.md` kullanılmalı.
- `risk_funnel.png`: tüm veriden güçlü high-risk kanıta giden seçiciliği gösterir.
- `reason_code_breakdown.png` ve `psychological_trigger_breakdown.png`: açıklama kapsamını gösterir.
- `live_inference_benchmark.png`: gizli metin inference hazırlığını gösterir.
- Önemli çerçeve: etiket olmadığı için proje proxy kanıt metrikleri ve senaryo kontrolleri raporlar; etiketli performans metriği iddia etmez.

---

## 10. Üretilen Grafikler Ne Anlatıyor?
- `risk_score_histogram.png`: Skor dağılımı ve Review/High eşikleri; modelin seçiciliğini gösterir.
- `risk_map_language_platform.png`: Dil x platform risk yoğunluğu; manipülasyon haritasının ana görselidir.
- `language_manipulative_share.png`: Dillere göre Manipulative satır sayısını ve o dil içindeki Manipulative oranını birlikte gösterir.
- `top_suspicious_segments.png`: En yüksek Review + High hacmine sahip segmentleri gösterir.
- `platform_normalized_risk.png`: Platformları kendi hacmine göre normalize ederek adil kıyas sağlar.
- `top_risk_authors.png`: Author bazlı hacim, burst, tekrar ve çeşitlilik risklerini özetler.
- `hourly_suspicious_share.png`: Zamansal risk yoğunlaşmasını ve burst dönemlerini gösterir.
- `risk_funnel.png`: Seçicilik hunisi; tüm veriden güçlü reason destekli High örneklere geçişi gösterir.
- `reason_code_breakdown.png`: High-risk kararların hangi reason code'larla açıklandığını gösterir.
- `psychological_trigger_breakdown.png`: FOMO, aciliyet, kayıptan kaçınma, sosyal kanıt ve otorite taklidi kapsamını gösterir.
- `live_inference_benchmark.png`: Canlı inference senaryolarında risk skorlarını ve eşik davranışını gösterir.
- `coordination_confidence_bubble.png`: Koordinasyon cluster'larında zaman penceresi, güven, boyut ve risk ilişkisini gösterir.
- `evidence_quality_summary.png`: High-risk kararların savunulabilirlik kalitesini özetler.

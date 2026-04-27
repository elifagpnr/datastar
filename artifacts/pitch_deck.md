# DataLeague Datathon Pitch Deck

## 1. Problem
- Amaç: etiket olmadan organik ve manipülatif sosyal medya içeriklerini ayırmak.
- Çıktı: gönderi bazlı risk/organiklik skoru, manipülasyon haritası ve canlı inference fonksiyonu.

---

## 2. Veri
- Kaynak dosya: `dataset/datathonFINAL.parquet`
- Kolonlar: metin, anahtar kelimeler, sentiment, emotion, tema, dil, platform URL, author hash, tarih.
- Zorluk: bot/human etiketi yok; veri çok dilli, gürültülü ve gerçek sosyal medya metinlerinden oluşuyor.

---

## 3. Yaklaşım
- Gözetimsiz ve açıklanabilir risk skorlama.
- Üç ana sinyal: içerik riski, author davranış riski, koordinasyon riski.
- Final skor: content, author ve coordination kanıtlarının ağırlıklı birleşimi; eksik metadata olduğunda ağırlıklar yeniden normalize edilir.
- Canlı text-only tahminlerde yardımcı katman: pseudo-label ile eğitilmiş hash n-gram NLP text modeli.

---

## 4. Feature Grupları
- İçerik: boş metin, düşük kanıtlı kısa metin, tekrar, link/hashtag/mention yoğunluğu, büyük harf, noktalama patlaması, uç sentiment.
- Author: yüksek hacim, burst posting, tekrar eden anlatılar, düşük konu çeşitliliği.
- Koordinasyon: 72 saatten kısa pencerede, farklı author/platformlarda görülen generic olmayan keyword/text signature eşleşmeleri.

---

## 5. Manipülasyon Haritası
- Skorlanan örnek satır: 200,000
- Risk bantları: High >= 0.65, Review >= 0.45, aksi halde Low.
- Örneklemde Review + High oranı: 4.0%
- Örneklemde High-risk oranı: 1.8%
- Ana görsel: `artifacts/risk_map_language_platform.png`
- Temporal kampanya pencereleri: 0 Campaign Burst; görsel `artifacts/temporal_burst_windows.png`

---

## 6. En Şüpheli Segmentler
- en / x.com / Cryptocurrency: 799 Review/High gönderi (484 High)
- en / x.com / Technology: 544 Review/High gönderi (333 High)
- en / x.com / People: 505 Review/High gönderi (198 High)
- en / reddit.com / Technology: 459 Review/High gönderi (346 High)
- en / x.com / Entertainment: 420 Review/High gönderi (147 High)
- Platform-normalize risk tablosu: `artifacts/platform_normalized_risk.csv`

---

## 7. Koordinasyon Cluster Örnekleri
- kw:#paws|paws|share: boyut=8.0, author=8.0, pencere=0.3 saat, güven=0.94, risk=0.85
- kw:btc|https|join: boyut=7.0, author=7.0, pencere=1.8 saat, güven=0.94, risk=0.85
- kw:cinema|cinema history|greatest|greatest villains|history|repost|villains: boyut=4.0, author=4.0, pencere=23.6 saat, güven=0.91, risk=0.76
- Somut şüpheli gönderi örnekleri: `artifacts/presentation_examples.csv`
- Jüriye anlatılacak hikayeler: `artifacts/case_studies.md`
- En riskli author özetleri: `artifacts/top_risk_authors.csv`

---

## 8. Normalize Platform ve Author Sinyalleri
- news.ycombinator.com: risk indeksi=1.53, Review/High oranı=6.2%
- boards.4channel.org: risk indeksi=1.44, Review/High oranı=5.8%
- youtube.com: risk indeksi=1.18, Review/High oranı=4.8%
- b34e6852df...: author risk=0.97, gönderi=304, Review/High oranı=100.0%
- 31ce25fc27...: author risk=0.88, gönderi=8, Review/High oranı=0.0%
- 8d501770c7...: author risk=0.87, gönderi=3, Review/High oranı=100.0%

---

## 9. Kanıt Scorecard
- Skorlanan örnek satır: 200,000
- Review + High adet/oran: 8,033 (4.0%)
- High adet/oran: 3,663 (1.8%)
- High-risk boş olmayan oran: 97.9%
- Güçlü reason destekli High oranı: 80.6%
- Sunuma uygun güçlü High örnek: 2,953
- Ek görseller: `risk_funnel.png`, `reason_code_breakdown.png`, `psychological_trigger_breakdown.png`, `evidence_quality_summary.png`, `feature_importance_proxy.png`, `risk_component_contribution.png`.
- Kampanya dönemi kanıtı: `temporal_burst_windows.csv` ve `temporal_burst_windows.png`.
- Etiket sağlanmadığı için bunlar proxy kanıt metrikleri ve senaryo kontrolleridir; etiketli performans metriği iddiası değildir.

---

## 10. Canlı Demo ve Sınırlar
- Demo fonksiyonu: `predict_live(text, language=None, url=None, author_hash=None, date=None, english_keywords=None)`.
- Dönen alanlar: label, risk score, organic score, nlp_text_risk, top reasons, used features ve psychological triggers.
- Canlı inference hazırlık senaryoları: 10/10 başarılı.
- Benchmark artifact'i: `artifacts/live_inference_benchmark.csv`.
- NLP text model artifact'i: `artifacts/nlp_text_model.npz`; eğitim özeti: `artifacts/nlp_text_model_metadata.json`.
- NLP eğitim verisi: yüksek güvenli pseudo-positive ve pseudo-negative örneklerden sınıf başına 2,861 metin.
- Sınırlar: gözetimsiz eşikleme kullanılır; exact/near-exact narrative signature yaklaşımı vardır; metadata geldikçe güven seviyesi artar.

---

## 11. Grafikler Ne Anlatıyor?
- `risk_score_histogram.png`: Risk skorlarının dağılımını ve Review/High eşiklerini gösterir; modelin ne kadar seçici olduğunu anlatır.
- `risk_map_language_platform.png`: Dil x platform kırılımında Review + High yoğunluğunu gösterir; manipülasyon haritasının ana görselidir.
- `language_manipulative_share.png`: Dillere göre Manipulative satır sayısını ve o dil içindeki Manipulative oranını birlikte gösterir.
- `top_suspicious_segments.png`: En çok Review + High üreten dil/platform segmentlerini sıralar; riskin nerede yoğunlaştığını gösterir.
- `platform_normalized_risk.png`: Platformları kendi hacmine göre normalize eder; büyük platformların ham sayı avantajını dengeler.
- `top_risk_authors.png`: Author bazlı hacim, burst, tekrar ve konu çeşitliliği sinyallerini özetler.
- `hourly_suspicious_share.png`: Saatlik Review + High oranını gösterir; burst/kampanya dönemlerini yakalamaya yarar.
- `temporal_burst_windows.png`: Saatlik risk oranındaki z-score sıçramalarını ve Campaign Burst pencerelerini işaretler.
- `risk_funnel.png`: Tüm veriden güçlü reason destekli High örneklere daralan seçim hunisini gösterir.
- `reason_code_breakdown.png`: High-risk kararların hangi açıklanabilir reason code'lara dayandığını gösterir.
- `psychological_trigger_breakdown.png`: FOMO, aciliyet, kayıptan kaçınma, sosyal kanıt ve otorite taklidi sinyallerini gösterir.
- `live_inference_benchmark.png`: Sentetik canlı test senaryolarında modelin risk skorlarını eşiklerle birlikte gösterir.
- `coordination_confidence_bubble.png`: Cluster güveni, zaman penceresi, cluster boyutu ve coordination risk ilişkisini gösterir.
- `evidence_quality_summary.png`: High-risk kararların boş olmayan metin, güçlü reason, içerik/author/coordination desteği gibi kalite boyutlarını özetler.
- `feature_importance_proxy.png`: Etiketsiz ortamda klasik feature importance yerine skor formülüne göre katkı yapan sinyalleri gösterir.
- `risk_component_contribution.png`: Risk bandlarına göre content, author, coordination ve rule-floor katkılarını gösterir.
- `case_studies.md`: Finansal scam, koordinasyon ve psikolojik tetikleyici örneklerini gerçek skorlanmış satırlardan hikayeleştirir.

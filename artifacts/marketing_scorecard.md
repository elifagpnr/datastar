# Pazarlama Kanıt Scorecard

Bu bölüm gözetimsiz bir kanıt katmanıdır. Proxy metrikler, senaryo kontrolleri ve açıklama kapsamı kullanır; etiketli model başarısı iddiası taşımaz.

| Metrik | Değer | Açıklama |
|---|---:|---|
| Skorlanan örnek satır | 200,000 | Mevcut artifact üretiminde skorlanan toplam satır sayısı. |
| Review + High adet/oran | 8,033 (4.0%) | Analist incelemesine ayrılan seçici, düşük olmayan risk havuzu. |
| High adet/oran | 3,663 (1.8%) | Canlı inference sırasında Manipulative etiketi üreten yüksek riskli havuz. |
| High-risk boş olmayan oran | 97.9% | Boş metin anomalisi olmayan yüksek riskli satırların oranı. |
| Güçlü reason destekli High oranı | 80.6% | Güçlü manipülasyon sinyalleriyle açıklanabilen yüksek riskli kararların oranı. |
| Sunuma uygun güçlü High örnek | 2,953 | Kalite filtrelerinden sonra kalan, savunulabilir ve boş olmayan yüksek riskli örnek sayısı. |
| En yüksek platform-normalize risk indeksi | news.ycombinator.com: 1.53x | Yeterli örnek hacmine sahip platformlarda, veri seti ortalamasına göre göreli risk. |
| En riskli segment | en / x.com / Cryptocurrency | 799 Review+High gönderi. |
| Yüksek güvenli koordinasyon cluster | 9 | Cluster confidence >= 0.80, coordination risk >= 0.65 ve zaman penceresi <= 72 saat. |
| Canlı inference hazırlık kontrolü | 10/10 | Sentetik senaryo kontrolleri; etiketli performans metriği değildir. |

## Üretilen Grafikler Ne Anlatıyor?

| Grafik | Ne ifade ediyor? | Sunumda nasıl kullanılır? |
|---|---|---|
| `risk_score_histogram.png` | Tüm örneklerde risk skorunun dağılımını ve Review/High eşiklerini gösterir. | Modelin seçici davrandığını, her şeyi manipülatif saymadığını anlatmak için kullanılır. |
| `risk_map_language_platform.png` | Dil x platform kırılımında Review + High yoğunluğunu 0-700 sabit renk skalasıyla gösterir. | Manipülasyon haritası ve sıcak bölgeleri göstermek için ana görseldir. |
| `language_manipulative_share.png` | Dillere göre Manipulative satır sayısını ve her dil içindeki Manipulative oranını birlikte gösterir. | Büyük dillerdeki hacmi ve küçük/orta dillerdeki göreli risk yoğunluğunu aynı anda anlatır. |
| `top_suspicious_segments.png` | En çok Review + High üreten dil/platform segmentlerini sıralar. | Hangi platform/dil alanlarında riskin yoğunlaştığını hızlı anlatır. |
| `platform_normalized_risk.png` | Platform içi risk oranını veri seti ortalamasına göre normalize eder. | X gibi büyük platformların ham hacim avantajını dengeleyip adil kıyas sunar. |
| `top_risk_authors.png` | Davranışsal sinyallere göre en riskli author_hash değerlerini gösterir. | Burst, tekrar ve hacim davranışını tekil aktör seviyesinde kanıtlar. |
| `hourly_suspicious_share.png` | Saatlik Review + High oranını zaman üzerinde gösterir. | Kampanya/burst dönemlerini ve zamansal yoğunlaşmayı anlatır. |
| `temporal_burst_windows.png` | Saatlik risk oranındaki z-score sıçramalarını ve Campaign Burst pencerelerini gösterir. | Gerçek dünyada kampanya dönemlerini erken uyarı olarak yakalayabildiğimizi anlatır. |
| `risk_funnel.png` | Tüm satırlardan Review + High, High ve güçlü reason destekli High örneklere daralan huniyi gösterir. | Modelin seçici ve kanıt odaklı karar verdiğini pazarlamak için kullanılır. |
| `reason_code_breakdown.png` | High-risk kararlarını açıklayan reason code dağılımını gösterir. | Kararların kara kutu olmadığını, açıklanabilir sinyallere dayandığını gösterir. |
| `psychological_trigger_breakdown.png` | FOMO, aciliyet, kayıptan kaçınma, sosyal kanıt ve otorite taklidi sinyallerinin kapsamını gösterir. | Spam dilini psikolojik manipülasyon tetikleyicileriyle ilişkilendirmek için kullanılır. |
| `live_inference_benchmark.png` | Sentetik canlı inference senaryolarının risk skorlarını eşiklerle birlikte gösterir. | Jürinin gizli metin testine hazır olduğumuzu göstermek için kullanılır. |
| `coordination_confidence_bubble.png` | Coordination cluster'larının zaman penceresi, güven skoru, boyut ve coordination risk ilişkisini gösterir. | Koordineli davranış tespitinin sadece metin değil, zaman ve author yayılımıyla desteklendiğini anlatır. |
| `evidence_quality_summary.png` | High-risk kararların ne kadarının boş olmayan, güçlü reason destekli, içerik/author/coordination destekli olduğunu gösterir. | High-risk karar kalitesini ve savunulabilirliği özetler. |
| `feature_importance_proxy.png` | Etiket olmadığı için klasik feature importance yerine, skor formülüne göre High-risk satırlarda en çok katkı yapan engineered sinyalleri gösterir. | Hangi feature engineering kararlarının skoru taşıdığını dürüst ve açıklanabilir şekilde anlatır. |
| `risk_component_contribution.png` | Low/Review/High bandlarında content, author, coordination ve rule-floor katkılarının ortalama dağılımını gösterir. | Metadata ve text sinyallerinin final skorda nasıl birleştiğini görselleştirir. |
| `case_studies.md` | Gerçek skorlanmış satırlardan seçilen 3 savunulabilir hikayeyi özetler. | Jürinin 'somut örnek göster' sorusuna doğrudan cevap olarak kullanılır. |

Önerilen konumlandırma: Etiketsiz sosyal medya verisi için açıklanabilir, seçici ve canlı inference'a hazır risk skorlama sistemi.
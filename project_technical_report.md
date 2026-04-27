# Datathon Teknik Proje Raporu

Bu rapor, projenin başından mevcut hale gelene kadar veri üzerinde yapılan düzenlemeleri, modelleme yaklaşımını, feature engineering katmanlarını, kalibrasyon kararlarını, üretilen kanıt artifact'lerini ve canlı jüri testine hazırlık durumunu açıklar.

Ana çerçeve şudur: Bu görev etiketli bir sınıflandırma problemi değil, gözetimsiz manipülasyon/anomali tespiti problemidir. Bu nedenle proje `accuracy`, `precision`, `recall` gibi ground-truth performans iddiaları üzerine değil; açıklanabilir risk skoru, reason code, risk bandı, koordinasyon kanıtı ve canlı inference çıktısı üzerine kurulmuştur.

## 1. Yönetici Özeti

Projede `dataset/datathonFINAL.parquet` verisi üzerinde açıklanabilir bir manipülasyon risk skorlama pipeline'ı kuruldu. Sistem her sosyal medya metni için:

- `risk_score`
- `organic_score`
- `risk_band`
- `label`
- `reason_codes`
- `content_risk`
- `author_risk`
- `coordination_risk`
- `nlp_text_risk`
- `evidence_level`

üretir.

Sistem iki çalışma modunu destekler:

| Mod | Komut / Fonksiyon | Amaç |
|---|---|---|
| Tek metin canlı inference | `predict_live()` veya `python3 datathon_pipeline.py predict "..."` | Jürinin tekil gizli metin vermesi durumunda anlık skor üretmek. |
| CSV batch inference | `python3 datathon_pipeline.py predict-csv jury_texts.csv --output artifacts/jury_predictions.csv` | Jürinin 100+ satırlı CSV vermesi durumunda tüm metinleri tek seferde skorlamak. |
| Veri seti artifact üretimi | `python3 datathon_pipeline.py run --sample-size 200000` | Final skor dosyaları, grafikler, scorecard ve pitch deck üretmek. |

Final 200k örneklem artifact run özeti:

| Metrik | Değer |
|---|---:|
| Skorlanan örnek satır | 200,000 |
| Review + High satır | 10,264 |
| Review + High oran | 5.132% |
| High satır | 4,516 |
| High oran | 2.258% |
| Review satır | 5,748 |
| High-risk boş olmayan oran | 98.3% |
| Güçlü reason destekli High oranı | 82.9% |
| Sunuma uygun güçlü High örnek | 3,746 |
| Yüksek güvenli koordinasyon cluster | 9 |
| Canlı inference hazırlık benchmark | 10/10 |

Bu metrikler etiketli başarı metrikleri değildir. Bunlar sistemin seçicilik, açıklanabilirlik, canlı çalışabilirlik ve kanıt üretme kalitesini gösteren proxy metriklerdir.

## 2. Veri Üzerinde Yapılan İşlemler

Ham veri doğrudan değiştirilmedi. Yapılan tüm işlemler pipeline içinde türetilmiş kolonlar, temizlenmiş temsiller, skorlar, grafikler ve CSV artifact'leri üretmek içindir. Yani orijinal parquet veri manipüle edilmedi; analiz için normalize edildi ve zenginleştirildi.

### 2.1 Veri Okuma

Ana veri dosyası:

```text
dataset/datathonFINAL.parquet
```

Okuma ve örnekleme için `duckdb` kullanıldı. Bunun sebebi büyük parquet dosyasını belleğe tamamen yüklemeden hızlı örneklem alabilmek ve gerektiğinde author-level aggregate hesaplarını doğrudan parquet üzerinden yapabilmektir.

Kullanılan iki ana okuma stratejisi:

| İşlem | Açıklama | Neden |
|---|---|---|
| `load_sample()` | Parquet içinden `sample_size` kadar örnek alır. | Hızlı iterasyon ve final artifact üretimi. |
| `build_full_author_stats()` | Tüm parquet üzerinden author bazlı aggregate çıkarır. | Örneklemde az görünen ama tüm veride yoğun davranan author'ları kaçırmamak. |

### 2.2 Kolon Standardizasyonu

Pipeline şu kolonları bekler:

```text
original_text
english_keywords
sentiment
main_emotion
primary_theme
language
url
author_hash
date
```

Eksik kolonlar varsa pipeline otomatik olarak boş değerle tamamlar. Bunun amacı canlı inference ve CSV batch inference sırasında farklı formatlı verilerle kırılmadan çalışmaktır.

### 2.3 Metin Temizliği ve Normalizasyon

Yapılan temel düzenlemeler:

| İşlem | Açıklama | Amacı |
|---|---|---|
| Null değer temizliği | `None` / `NaN` metinler boş string'e çevrildi. | Boş metin anomalisini açık şekilde yakalamak. |
| Whitespace temizliği | Baş/son boşluklar kaldırıldı. | Sadece boşluk içeren metinleri gerçek boş metin gibi ele almak. |
| Casefold | Metinler karşılaştırma için küçük harfe ve case-insensitive forma çekildi. | Pattern eşleşmelerini tutarlı yapmak. |
| Türkçe karakter katlama | `ğ, ü, ş, ı, ö, ç` gibi karakterler ASCII karşılıklarına çevrildi. | Türkçe ve İngilizce sinyal sözlüklerini aynı normalize edilmiş uzayda çalıştırmak. |
| URL temizliği | Signature çıkarırken URL'ler metinden temizlendi. | Signature'ın linkten değil, anlatıdan oluşmasını sağlamak. |
| Hashtag/mention temizliği | Signature aşamasında `#...` ve `@...` temizlendi. | Kampanya anlatısını link/mention gürültüsünden ayırmak. |

### 2.4 Platform ve Zaman Bilgisi Üretimi

`url` kolonundan `platform_domain` çıkarıldı.

Örnek:

```text
https://x.com/... -> x.com
https://www.reddit.com/... -> reddit.com
```

`date` kolonu `date_ts` zaman damgasına çevrildi. Bu iki düzenleme şu analizleri mümkün kıldı:

- dil x platform risk haritası
- platform-normalized risk
- hourly suspicious share
- temporal burst windows
- author burst davranışı
- coordination cluster zaman penceresi

## 3. Neden Bu Modelleme Yaklaşımı Seçildi?

Yarışma yönergesine göre hazır test seti veya etiket yoktur. Bu yüzden klasik supervised model eğitimi, yani gerçek label ile `X -> y` öğrenen bir model kurmak metodolojik olarak sağlam değildir.

Bu nedenle ana modelleme yaklaşımı:

```text
Gözetimsiz + açıklanabilir risk skorlama
```

olarak seçildi.

### 3.1 Kullanılan Ana Skorlama Sistemi

Ana skorlama sistemi, üç bileşenli bir risk bileşim sistemidir:

```text
final risk_score =
  content_risk
+ author_risk
+ coordination_risk
```

Bu bileşenler ağırlıklı ortalama ile birleşir. Metadata eksikse ağırlıklar yeniden normalize edilir. Böylece sadece metin verilen canlı inference senaryosunda model çalışmaya devam eder; metadata geldiğinde ise author ve coordination kanıtları skoru güçlendirir.

Kullanılan ana ağırlıklar:

```text
content_weight = 0.35
author_weight = 0.30
coordination_weight = 0.35
```

### 3.2 Neden Rule-Based / Explainable Skor?

Bu yarışma için bu yaklaşımın avantajları:

| Gereksinim | Yaklaşımın Karşıladığı Nokta |
|---|---|
| Etiket yok | Label gerektirmez. |
| Jüri canlı metin verecek | `predict_live()` ve `predict-csv` ile anlık çalışır. |
| Kararın açıklanması gerekir | Her karar `reason_codes` ile açıklanır. |
| Manipülasyon haritası istenir | Dil/platform/author/cluster artifact'leri üretir. |
| Veri çok gürültülü | Kısa metin, boş metin, generic keyword, media keyword gibi durumlar ayrı ele alınır. |
| Sunumda savunulabilirlik önemli | Scorecard, grafikler ve presentation examples filtreleri vardır. |

### 3.3 Kullanılan Yardımcı NLP Modeli

Ana sistemin yanında, özellikle jürinin sadece `original_text` vereceği canlı test için hafif bir NLP destek katmanı eklendi:

```text
pseudo-label-hash-ngram-v1
```

Bu model:

- LLM değildir.
- Embedding modeli değildir.
- Dışarıdan etiketli veriyle eğitilmiş supervised model değildir.
- Mevcut açıklanabilir skorlamanın en güvenli pozitif ve negatif örneklerinden pseudo-label öğrenir.

NLP model metadata:

| Metrik | Değer |
|---|---:|
| Hash boyutu | 131,072 |
| Pozitif aday | 3,656 |
| Negatif aday | 112,968 |
| Eğitim dokümanı / sınıf | 3,656 |
| Pozitif doc feature | 2,626,537 |
| Negatif doc feature | 1,227,382 |
| Scoring | `sigmoid(mean(feature_log_odds)) with rule-evidence gate` |

Bu katman canlı metinde `nlp_text_risk` üretir. Ancak tek başına final kararı ezmez; rule layer'da risk kanıtı varsa destekleyici olarak çalışır. Bu gate, organik metinlerin sadece NLP skoru yüzünden yanlışlıkla High bandına taşınmasını engeller.

## 4. Risk Band ve Label Tasarımı

Üçlü risk bandı kullanıldı:

| Band | Eşik | Binary Label |
|---|---:|---|
| `High` | `risk_score >= 0.65` | `Manipulative` |
| `Review` | `0.45 <= risk_score < 0.65` | `Organic` |
| `Low` | `< 0.45` | `Organic` |

Neden bu tasarım?

- Jürinin istediği binary output korunur: `Organic` / `Manipulative`.
- Belirsiz ama şüpheli metinler `Review` bandında görünür.
- Model her şeyi manipülatif demediği için seçiciliği savunulabilir.
- Dashboard ve grafiklerde `Review + High` havuzu analiz edilebilir.

## 5. Başlangıçtan Bugüne Yapılan Ana Düzenlemeler

### 5.1 İlk Pipeline

İlk aşamada temel pipeline kuruldu:

- parquet okuma
- örneklem alma
- text feature extraction
- content risk skoru
- author risk skoru
- coordination risk skoru
- final `risk_score`
- `predict_live()`
- notebook ve artifact üretimi

### 5.2 Kısa Metin Kalibrasyonu

İlk 50k çıktıda önemli bir sorun görüldü: High-risk örneklerin büyük bölümü `<=3` karakterlik metinlerden geliyordu. Bu jüri demosunda zayıf görünürdü çünkü `yes`, `lol`, `No`, tek emoji gibi metinler tek başına manipülasyon kanıtı değildir.

Yapılan düzenleme:

| Durum | Yeni Davranış |
|---|---|
| Boş / whitespace metin | Yüksek risk kalabilir, reason: `EMPTY_TEXT`. |
| Boş olmayan çok kısa metin | Tek başına High olmaz, reason: `SHORT_TEXT_LOW_EVIDENCE`. |
| Kısa metin + link/CTA/repetition/author/coordination | Diğer kanıtlarla yükselir. |

Bu düzenleme false-positive riskini ciddi biçimde düşürdü.

### 5.3 Generic ve Media Keyword Filtreleri

Şu tür zayıf signature'lar coordination cluster üretmesin diye filtrelendi:

```text
yes, lol, nice, good, love, thank, thanks,
birthday, happy birthday, sorry, awesome,
jpeg, jpg, png, gif, preview, redd, pjpg, image, photo
```

Neden?

`happy birthday` veya `jpeg preview` gibi imzalar çok satırda geçebilir ama bu koordineli manipülasyon anlamına gelmez. Bu filtreler coordination tarafında savunulabilirliği artırdı.

### 5.4 Coordination Güven Kriterleri

Coordination risk üretmek için daha sıkı kurallar getirildi:

```text
cluster_size >= 4
cluster_author_nunique >= 3
signature_term_count >= 2
cluster_window_hours <= 72
```

Ayrıca boş `author_hash` değerleri author çeşitliliği sinyali sayılmadı.

Bu sayede uzun zamana yayılmış, generic veya medya kaynaklı cluster'lar top coordination listesine girmedi.

### 5.5 Public Output ve Risk Band

Public çıktılara `risk_band` eklendi:

- `High`
- `Review`
- `Low`

Bu, hem canlı inference hem dashboard hem de sunum için daha doğru bir ayrım sağladı.

### 5.6 Presentation Examples Filtresi

`presentation_examples.csv` için daha sıkı seçim kuralı getirildi:

- metin uzunluğu yeterli olmalı
- `EMPTY_TEXT` olmamalı
- URL-only veya media-only olmamalı
- strong reason içermeli
- coordination örneklerinde zaman penceresi 72 saatten kısa olmalı

Amaç, jüriye boş metin veya zayıf örnek göstermek yerine savunulabilir örnekler sunmaktır.

### 5.7 Pazarlama ve Kanıt Artifact'leri

Etiket olmadığı için accuracy iddia etmek yerine proxy kanıt artifact'leri eklendi:

- `marketing_scorecard.md`
- `risk_funnel.png`
- `reason_code_breakdown.png`
- `psychological_trigger_breakdown.png`
- `live_inference_benchmark.png`
- `coordination_confidence_bubble.png`
- `evidence_quality_summary.png`
- `language_manipulative_share.png`
- `platform_normalized_risk.png`

Bu grafiklerin amacı modeli olduğundan iyi göstermek değil; iyi çalıştığı yerleri dürüst ve ölçülebilir biçimde görünür kılmaktır.

### 5.8 Psikolojik Tetikleyici Katmanı

Spam/manipülasyon metinlerinin sadece teknik sinyallerle değil, psikolojik manipülasyon diliyle de ayrıştığı görüldü. Bu nedenle şu tetikleyiciler eklendi:

- FOMO
- Aciliyet
- Kayıptan kaçınma
- Sosyal kanıt
- Otorite taklidi

Bu katman özellikle jüri anlatısını güçlendirir: sistem sadece "spam" demez, "bu metin hangi manipülasyon tekniğini kullanıyor?" sorusuna da cevap verir.

### 5.9 Text-Only NLP Destek Katmanı

Jürinin gizli testte sadece `original_text` vereceği bildirildi. Bu nedenle author/coordination metadata olmadan da daha tutarlı çalışacak pseudo-label NLP destek modeli eklendi.

Bu model:

- hashed word unigram
- word bigram
- character 3/4/5-gram
- log-odds ağırlıkları

kullanır.

Ana amaç: sadece pattern sözlüklerine değil, geçmiş yüksek güvenli örneklerden öğrenilen metinsel örüntülere de dayanmak.

### 5.10 CSV Batch Inference

Son aşamada jüri 100+ satırlı CSV verebileceği için `predict-csv` komutu eklendi.

Desteklenen senaryolar:

| CSV Formatı | Çözüm |
|---|---|
| `original_text` kolonu var | Otomatik kullanılır. |
| Tek kolonlu CSV | Otomatik text kolonu kabul edilir. |
| Header yok | `--no-header`. |
| Metinler farklı kolon/hücrelere dağılmış | `--all-cells`. |
| Kolon adı farklı | `--text-column "kolon_adi"`. |
| Semicolon delimiter | `--sep ";"`. |

Bu, canlı jüri koşullarına doğrudan hazır olduğumuzu gösterir.

## 6. Feature Engineering Katmanları

Evet, projede kapsamlı feature engineering yapıldı. Feature'lar sadece model skoruna değil, açıklama, grafik, segmentasyon ve canlı inference çıktısına da hizmet eder.

### 6.1 Temel Metin Feature'ları

| Feature | Ne Ölçer? | Neden Gerekli? |
|---|---|---|
| `char_len` | Ham metin uzunluğu. | Çok kısa/çok uzun metin anomalileri. |
| `stripped_char_len` | Boşluklar temizlendikten sonra uzunluk. | Whitespace-only metinleri yakalamak. |
| `empty_text` | Metin boş mu? | Veri anomalisi veya silinmiş içerik sinyali. |
| `short_text` | Metin çok kısa mı? | `yes`, `lol`, `No` false-positive'lerini azaltmak. |
| `word_count` | Token sayısı. | Oran feature'ları için payda. |
| `unique_word_count` | Benzersiz token sayısı. | Tekrarı ölçmek. |
| `repetition_ratio` | Token tekrar oranı. | `FREE FREE FREE` gibi spam kalıpları. |
| `max_token_share` | En sık token'ın baskınlığı. | Kısa spamlerde tek kelime yoğunluğu. |
| `uppercase_ratio` | Büyük harf oranı. | `BUY NOW`, `URGENT`, `ACİL` sinyalleri. |
| `punctuation_density` | `!` ve `?` yoğunluğu. | Aciliyet, panik, clickbait dili. |
| `digit_ratio` | Rakam oranı. | Para, süre, ödül vaatleri. |
| `repeated_char_run` | Aynı karakterin art arda tekrarı. | Bot/spam yazım anomalisi. |

### 6.2 Link, Hashtag, Mention Feature'ları

| Feature | Ne Ölçer? | Neden Gerekli? |
|---|---|---|
| `url_count` | URL sayısı. | Phishing ve dış yönlendirme. |
| `hashtag_count` | Hashtag sayısı. | Trend/kampanya manipülasyonu. |
| `mention_count` | Mention sayısı. | Etiket spam ve ağ yayılımı. |
| `link_signal_density` | Link/hashtag/mention yoğunluğu. | Metnin aksiyon ve yayılım odaklı olup olmadığını ölçer. |

Formül:

```text
link_signal_density =
  (2 * url_count + hashtag_count + mention_count) / max(word_count, 1)
```

URL'ye daha yüksek ağırlık verilmesinin sebebi dış bağlantının phishing/spam riskini hashtag veya mention'dan daha güçlü taşımasıdır.

### 6.3 Call-to-Action Feature'ları

`call_to_action_count`, metinde kullanıcıyı hemen aksiyona çağıran terimleri sayar:

```text
buy, free, click, join, watch, follow, dm,
hemen, takip, tikla, link, bio, claim,
verify, wallet, telegram, vip, rt, vote
```

CTA tek başına manipülasyon değildir. Ancak link, urgency, finansal vaat veya phishing tehdidi ile birleştiğinde güçlü risk sinyalidir.

### 6.4 Scam / Spam Pattern Grupları

Metinsel manipülasyon türleri pattern gruplarına ayrıldı:

| Grup | Yakalanan Senaryo |
|---|---|
| `financial_bait` | Ek gelir, kripto, forex, garanti kar, 100x. |
| `phishing_threat` | Hesap kapatma, telif tehdidi, itiraz formu. |
| `engagement_bait` | Çekiliş, takip et, beğen, tag, DM, link in bio. |
| `adult_or_leak_bait` | İfşa, leaked video, özel fotoğraf. |
| `health_miracle` | Mucizevi kilo verme, detox, sağlık iddiası. |
| `prize_fee` | Ödül kazandınız, kargo ücreti ödeyin. |
| `debt_relief` | Kredi kartı borcu kurtarma vaadi. |
| `wallet_verification` | Cüzdan doğrulama, MetaMask, USDT bonus. |
| `trading_bot_bait` | Trading bot, 100% legit, linktr.ee. |
| `market_manipulation` | Gem alert, moon, Binance listing. |
| `political_mobilization` | RT yap, tag'e destek ver, gizli belgeler. |

Bu gruplardan iki üst feature çıkarılır:

```text
scam_signal_count
scam_signal_groups
```

`scam_signal_groups`, kaç farklı manipülasyon kategorisinin aynı metinde aktif olduğunu gösterir. Bu sayede tek bir kelimeye fazla güvenmek yerine çoklu kanıt aranır.

### 6.5 Psikolojik Tetikleyici Feature'ları

| Tetikleyici | Davranışsal Karşılığı | Örnek Sinyal |
|---|---|---|
| FOMO | Kaçırma korkusu. | Presale bitiyor, 100x olacak. |
| Urgency | Hızlı karar baskısı. | Son 15 dakika, hemen, 24 saat. |
| Loss aversion | Kayıptan kaçınma. | Hesabınız kapanacak, fonlarınızı kaybedebilirsiniz. |
| Social proof | Sürü etkisi. | 50.000 kullanıcı katıldı, herkes kazanıyor. |
| Authority impersonation | Otorite taklidi. | Meta support, resmi destek, banka. |

Bu katman kararın hem teknik hem davranışsal olarak açıklanmasını sağlar.

### 6.6 Sentiment Feature'ı

`sentiment_abs`, sentiment skorunun mutlak değeridir. Aşırı pozitif veya aşırı negatif metinleri yakalamak için kullanılır.

Neden?

Manipülatif metinler çoğu zaman ya yoğun vaat dili ya da yoğun tehdit/panik dili kullanır. Sentiment tek başına karar vermez ama diğer sinyallerle birlikte destekleyici feature'dır.

### 6.7 Narrative Signature Feature'ları

| Feature | Açıklama |
|---|---|
| `text_signature` | Metinden stopword dışı en baskın tokenları çıkarır. |
| `keyword_signature` | `english_keywords` alanından temizlenmiş anahtar kelime imzası üretir. |
| `narrative_signature` | Önce keyword signature, yoksa text signature kullanır. |

Amaç, aynı kampanya veya anlatı şablonunu tekrar eden içerikleri yakalamaktır.

Weak signature filtresi ile şu tür düşük anlamlı imzalar elenir:

```text
yes, lol, nice, good, birthday,
jpeg, jpg, png, preview, image, photo
```

Bu filtre coordination risk kalitesini artırır.

### 6.8 Author-Level Feature'lar

Author davranışını ölçen feature'lar:

| Feature | Amacı |
|---|---|
| `author_post_count_sample` | Örneklemdeki hacim. |
| `author_post_count_full` | Tüm verideki hacim. |
| `author_burst_pairs` | 5 dakika içinde ardışık gönderiler. |
| `author_text_signature_nunique` | Tekrar eden metin imzaları. |
| `author_keyword_signature_nunique` | Tekrar eden keyword imzaları. |
| `author_theme_nunique` | Konu çeşitliliği. |
| `author_language_nunique` | Dil çeşitliliği. |

Author risk formülü:

```text
author_risk =
  0.35 * author_volume_risk
+ 0.25 * author_burst_risk
+ 0.25 * author_repeat_risk
+ 0.15 * author_low_theme_diversity_risk
```

Bu katman metnin içeriğinden bağımsız davranışsal manipülasyon sinyali sağlar.

### 6.9 Coordination Feature'ları

Coordination analizi, aynı narrative signature'ın farklı author/platformlarda kısa sürede yayılıp yayılmadığını ölçer.

Kullanılan feature'lar:

| Feature | Amacı |
|---|---|
| `cluster_size` | Aynı anlatı kaç kez görülmüş? |
| `cluster_author_nunique` | Kaç farklı author var? |
| `cluster_platform_nunique` | Kaç farklı platform var? |
| `cluster_language_nunique` | Kaç farklı dil var? |
| `cluster_window_hours` | Yayılım kaç saatlik pencereye sıkışmış? |
| `signature_term_count` | İmza yeterince anlamlı mı? |
| `cluster_confidence_score` | Cluster sunumda savunulabilir mi? |

Coordination risk:

```text
coordination_risk =
  0.40 * coord_size_risk
+ 0.35 * coord_author_spread_risk
+ 0.15 * coord_platform_spread_risk
+ 0.10 * coord_time_compactness_risk
```

Cluster confidence:

```text
cluster_confidence_score =
  0.35 * coord_author_spread_risk
+ 0.25 * coord_time_compactness_risk
+ 0.20 * coord_size_risk
+ 0.10 * signature_richness
+ 0.10 * cluster_content_similarity
```

Çok dilli yayılım için bilinçli konumlandırma:

- `cluster_language_nunique` aynı narrative signature içinde kaç farklı dil görüldüğünü görünür kılar.
- Bu sprintte coordination risk formülüne ek bir `language_spread_risk` ağırlığı eklenmedi.
- Sistem çevrilmiş aynı kampanyaları semantik olarak eşleştirdiğini iddia etmez; yalnızca aynı signature altında çok dilli yayılım varsa bunu kanıt alanı olarak raporlar.

## 7. Content Risk Formülü

Ana içerik skoru:

```text
content_risk =
  0.20 * short_or_tiny
+ 0.10 * very_long
+ 0.20 * repetition
+ 0.17 * link_density
+ 0.10 * caps
+ 0.08 * punctuation
+ 0.10 * sentiment_extreme
+ 0.05 * cta
+ 0.20 * scam_signal
```

Bu katsayıların toplamı 1.20'dir ve bu kasıtlıdır. Content risk bölümü klasik anlamda toplamı 1 olan normalize edilmiş ağırlıklı ortalama değildir; additive evidence accumulation mantığıyla çalışır. Tekil zayıf sinyaller tek başına High üretmezken, link yoğunluğu, tekrar, CTA, sentiment ve scam pattern gibi birden fazla bağımsız sinyal aynı metinde birleşirse riskin hızlı yükselmesi istenir. Sonuç her zaman `_clip01` ile 0-1 aralığına kırpılır.

Ek floor kuralları:

| Durum | Minimum Risk | Neden |
|---|---:|---|
| Link + punctuation + CTA yüksek | 0.82 | Açık spam paketi. |
| Phishing / wallet / trading bot / market manipulation | 0.72 | Kritik manipülasyon senaryolarını kaçırmamak. |
| Politik mobilizasyon güçlü | 0.66 | Koordineli amplifikasyon dilini High bandına yaklaştırmak. |
| Çoklu ama daha zayıf lure | 0.50 | Review havuzuna almak. |
| Boş metin | 0.92 | Veri anomalisi veya silinmiş içerik. |
| Repeated character run | 0.55 | Spam yazım anomalisi. |

Bu tasarım modelin sadece doğrusal ağırlıklarla değil, kritik risk senaryolarında minimum güvenlik eşiğiyle çalışmasını sağlar.

## 8. NLP Destek Modeli Detayı

Pseudo-label NLP modeli, ana açıklanabilir skorlamadan türetilmiş güvenli örneklerle eğitilir.

Pozitif pseudo-label:

```text
risk_score >= 0.65
non-empty text
meaningful signature
strong reason code
```

Negatif pseudo-label:

```text
risk_score <= 0.20
LOW_CONTENT_RISK
non-empty text
strong reason yok
```

Feature'lar:

| NLP Feature | Amacı |
|---|---|
| Word unigram | Riskli tekil kelimeleri öğrenmek. |
| Word bigram | `wallet verification`, `shipping fee`, `trading bot` gibi ifadeleri yakalamak. |
| Character 3/4/5-gram | Yazım varyasyonları ve Türkçe/İngilizce karışımına dayanıklılık. |
| Hashing | Ek ML bağımlılığı olmadan hızlı ve taşınabilir model. `zlib.crc32` deterministik, hızlı ve standart kütüphane içinde olduğu için tercih edildi. |

Skorlama:

```text
feature_log_odds = log(pos_rate) - log(neg_rate)
nlp_text_risk = sigmoid(mean(feature_log_odds))
```

Neden bu model?

- Etiket yokken supervised model gibi davranmaz.
- Çok hafiftir; sklearn gerektirmez.
- Canlı inference sırasında hızlıdır.
- Jürinin sadece `original_text` verdiği durumda ek dayanıklılık sağlar.
- Ana reason-code sistemini bozmadan destekleyici sinyal üretir.

## 9. Kütüphaneler ve Kullanım Amaçları

| Kütüphane | Kullanım |
|---|---|
| `pandas` | DataFrame işlemleri, CSV/artifact üretimi. |
| `numpy` | Vektörel skor hesapları, clipping, hash n-gram model ağırlıkları. |
| `duckdb` | Büyük parquet dosyasından hızlı örneklem ve aggregate. |
| `matplotlib` | Sunum grafikleri ve kanıt artifact'leri. |
| `json`, `argparse`, `pathlib`, `re`, `unicodedata`, `zlib` | CLI, dosya yönetimi, regex patternleri, normalizasyon, hash feature üretimi. |

Kullanılmayanlar:

| Kullanılmadı | Neden |
|---|---|
| LLM | Yarışma ortamında dış modele bağımlılık ve açıklanabilirlik riski. |
| Embedding modeli | Zaman/bağımlılık maliyeti ve offline çalışabilirlik riski. |
| Gerçek supervised classifier | Etiket olmadığı için metodolojik olarak savunulamaz. |
| Accuracy/precision/recall iddiası | Ground truth label yok. |

## 10. Üretilen Artifact'ler ve Kanıt Değeri

| Artifact | Ne Kanıtlıyor? |
|---|---|
| `scored_sample.csv` | Tüm skorlanan örneklerin risk, label, reason ve feature çıktıları. |
| `presentation_examples.csv` | Sunuma uygun, boş olmayan, güçlü reason destekli örnekler. |
| `risk_score_histogram.png` | Modelin seçiciliği ve skor dağılımı. |
| `risk_map_language_platform.png` | Dil x platform manipülasyon haritası. |
| `language_manipulative_share.png` | Dillere göre High sayısı ve dil içi oran. |
| `platform_normalized_risk.png` | Büyük platform hacim etkisini normalize eden adil kıyas. |
| `top_risk_authors.csv/png` | Riskli author davranışları. |
| `top_coordination_clusters.csv` | Yüksek güvenli koordinasyon cluster'ları. |
| `coordination_confidence_bubble.png` | Cluster güveni, zaman penceresi ve risk ilişkisi. |
| `reason_code_breakdown.png` | High kararlarının açıklanabilir reason dağılımı. |
| `psychological_trigger_breakdown.png` | Manipülasyon dilinin psikolojik tetikleyici kapsamı. |
| `risk_funnel.png` | Tüm veriden güçlü kanıtlı High örneklere daralan seçim hunisi. |
| `temporal_burst_windows.csv/png` | Saatlik risk oranındaki z-score sıçramalarını ve Campaign Burst pencerelerini gösterir. |
| `evidence_quality_summary.png` | High-risk kararların savunulabilirlik kalitesi. |
| `live_inference_benchmark.csv/png` | Canlı inference hazırlık senaryoları. |
| `case_studies.md` | Gerçek skorlanmış satırlardan seçilmiş finansal scam, koordinasyon ve psikolojik tetikleyici hikayeleri. |
| `marketing_scorecard.md` | Jüriye kısa, dürüst kanıt özeti. |
| `nlp_text_model.npz` | Text-only NLP destek modeli. |
| `nlp_text_model_metadata.json` | NLP model eğitim özeti. |

## 11. Canlı Jüri Testine Hazırlık

### Tek Metin

```bash
python3 datathon_pipeline.py predict "BUY NOW!!! FREE FREE FREE #deal #promo https://spam.example.com"
```

Çıktı:

- `label`
- `risk_score`
- `risk_band`
- `organic_score`
- `top_reasons`
- `nlp_text_risk`
- `used_features`

### 100+ Satırlı CSV

```bash
python3 datathon_pipeline.py predict-csv jury_texts.csv --output artifacts/jury_predictions.csv
```

Desteklenen CSV durumları:

```bash
python3 datathon_pipeline.py predict-csv jury_texts.csv --no-header --output artifacts/jury_predictions.csv
python3 datathon_pipeline.py predict-csv jury_texts.csv --all-cells --output artifacts/jury_predictions.csv
python3 datathon_pipeline.py predict-csv jury_texts.csv --text-column "text" --output artifacts/jury_predictions.csv
python3 datathon_pipeline.py predict-csv jury_texts.csv --sep ";" --output artifacts/jury_predictions.csv
```

CSV çıktısında kaynak satır ve kolon bilgisi de tutulur:

```text
source_row
source_column
original_text
label
risk_band
risk_score
organic_score
top_reasons
nlp_text_risk
evidence_level
```

## 12. Validasyon ve Testler

Çalıştırılan temel kontroller:

```bash
python3 -m py_compile datathon_pipeline.py
python3 run_smoke_tests.py
python3 datathon_pipeline.py smoke-test
python3 -m json.tool ads.ipynb
python3 datathon_pipeline.py run --sample-size 200000
```

Smoke test kapsamı:

- Organik metin düşük risk olmalı.
- Açık spam metin High/Manipulative olmalı.
- Aynı kampanya metni 4+ farklı author ile kısa sürede coordination risk üretmeli.
- `happy birthday` gibi generic cluster coordination risk üretmemeli.
- Boş metin `EMPTY_TEXT` reason vermeli.
- `yes`, `lol`, `No` gibi kısa metinler Organic/Low kalmalı.
- Emoji-only metinler Organic/Low kalmalı ve `SHORT_TEXT_LOW_EVIDENCE` vermeli.
- URL-only metinler crash üretmemeli, `EMPTY_TEXT` olmamalı ve link-only reason taşımalı.
- Çok uzun tek karakter tekrarı `REPEATED_CHARACTER_RUN` ile Review bandına düşmeli.
- Headersız ve semicolon CSV dosyaları batch inference ile okunmalı.
- Türkçe kripto/gelir vaadi manipülatif yakalanmalı.
- Phishing tehdidi manipülatif yakalanmalı.
- Batch CSV prediction temel örneklerde doğru satır sayısını ve beklenen label'ları üretmeli.

Canlı inference benchmark:

| Senaryo | Beklenen |
|---|---|
| Crypto/FOMO scam | Review/High |
| Wallet verification phishing | Review/High |
| Trading bot income scam | Review/High |
| Prize/shipping fee scam | Review/High |
| Authority impersonation | Review/High |
| Social proof manipulation | Review/High |
| Political amplification | Review/High |
| Organic civic announcement | Low |
| Organic casual reply | Low |
| Organic news summary | Low |

Son benchmark sonucu:

```text
10/10 readiness check
```

Bu sonuç etiketli accuracy değildir. Canlı inference davranışının beklenen yönde olduğunu gösteren senaryo testidir.

## 13. Projenin Güçlü Yanları

- Etiket olmayan görev yapısına metodolojik olarak uyumludur.
- Jüriye canlı tek metin ve CSV batch inference sunabilir.
- Kararları reason code ile açıklanabilir.
- Metadata varsa author ve coordination sinyallerini kullanır.
- Metadata yoksa content + NLP destek katmanıyla çalışır.
- Risk haritası, platform-normalized risk ve dil bazlı oran grafikleriyle analiz üretir.
- Coordination cluster'ları zaman penceresi, author yayılımı ve signature anlamlılığıyla savunulabilir hale getirilmiştir.
- Kısa metin ve generic/media cluster false-positive'leri özellikle kalibre edilmiştir.
- Sunuma uygun örnekler kalite filtrelerinden geçirilir.
- Dış LLM veya internet bağımlılığı yoktur; lokal çalışır.

## 14. Sınırlamalar

Bu projenin güçlü olması, sınırlarının olmadığı anlamına gelmez. Sunumda dürüstçe şu çerçeve korunmalıdır:

- Ground truth label olmadığı için accuracy/precision/recall iddia edilmez.
- Pattern tabanlı sinyaller yeni spam jargonları için zamanla genişletilmelidir.
- Sadece `original_text` verilen canlı testte author ve coordination sinyalleri sınırlı kalır.
- NLP destek modeli pseudo-label kaynaklıdır; gerçek supervised model değildir.
- İroni, bağlam, görsel içerik veya dış bağlantı içeriği analiz edilmez.
- Çok kısa metinler düşük kanıt olarak ele alınır; bu kasıtlı bir false-positive azaltma tercihidir.

## 15. Jüriye Anlatım Önerisi

Kısa teknik anlatım:

> Etiketli test seti olmadığı için supervised accuracy modeli kurmadık. Bunun yerine içerik, author davranışı ve koordinasyon sinyallerinden oluşan açıklanabilir bir risk skorlama pipeline'ı geliştirdik. Her karar reason code ile açıklanıyor. Jürinin vereceği tekil metinleri veya 100+ satırlı CSV dosyasını canlı olarak skorlayabiliyoruz. Metadata varsa karar author ve coordination sinyalleriyle güçleniyor; sadece original_text varsa content feature'ları ve pseudo-label hash n-gram NLP destek modeliyle skor üretiyoruz.

Pazarlama cümlesi:

> Bu sistem, etiketsiz sosyal medya verisi için seçici, açıklanabilir, canlı inference'a hazır ve kanıt üreten bir manipülasyon risk skorlama çözümüdür.

Kaçınılması gereken iddia:

> Modelimizin accuracy'si şu kadar.

Doğru iddia:

> Etiket olmadığı için accuracy iddia etmiyoruz; bunun yerine seçicilik, açıklanabilirlik, canlı inference readiness, risk band dağılımı, reason code kapsamı ve coordination confidence gibi proxy kanıt metrikleri raporluyoruz.

## 16. Son Durum

Mevcut proje durumu sunuma çıkabilir seviyededir:

- Pipeline çalışıyor.
- Notebook geçerli.
- Tek metin inference hazır.
- CSV batch inference hazır.
- 200k final artifact seti üretilmiş durumda.
- Grafikler ve scorecard hazır.
- Feature engineering raporu hazır.
- Bu teknik proje raporu hazır.

Son kontrol komutları:

```bash
python3 run_smoke_tests.py
python3 datathon_pipeline.py predict "BUY NOW!!! FREE FREE FREE #deal #promo https://spam.example.com"
python3 datathon_pipeline.py predict "The city council published meeting notes and invited residents to comment before Friday."
python3 datathon_pipeline.py predict-csv jury_texts.csv --output artifacts/jury_predictions.csv
```

Beklenen demo davranışı:

- Açık spam/manipülasyon metinleri `High` veya en az `Review`.
- Organik kontrol metinleri `Low`.
- Output her zaman reason code ve risk band içermeli.

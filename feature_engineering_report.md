# Feature Engineering Raporu

Bu rapor, projede kullanılan feature engineering katmanlarını ve her feature grubunun hangi amaca hizmet ettiğini açıklar. Proje etiketli bir sınıflandırma problemi olarak değil, gözetimsiz/açıklanabilir risk skorlama problemi olarak tasarlandığı için feature'lar doğrudan `accuracy` üretmekten ziyade manipülasyon, spam, koordinasyon ve anomali sinyallerini ölçmek için kullanılır.

## 1. Veri Hazırlama ve Normalizasyon

| Feature / İşlem | Ne Yapıyor? | Amacı |
|---|---|---|
| Zorunlu kolon tamamlama | `original_text`, `english_keywords`, `sentiment`, `author_hash`, `date`, `url`, `language`, `primary_theme` gibi kolonlar eksikse boş değerle tamamlanır. | Pipeline'ın hem parquet veri setinde hem de canlı inference sırasında kırılmadan çalışmasını sağlar. |
| Metin temizleme | `None` değerler boş string'e çevrilir, metinler normalize edilir. | Boş, kısa veya problemli metinlerin ayrı sinyal olarak yakalanmasını sağlar. |
| Türkçe karakter katlama | `ı, ğ, ü, ş, ö, ç` gibi karakterler ASCII karşılıklarına çevrilir. | Türkçe ve İngilizce sinyal sözlüklerinin daha tutarlı eşleşmesini sağlar. |
| URL domain çıkarımı | `url` alanından `platform_domain` üretilir. | Platform bazlı risk haritası, platform-normalized risk ve segment analizi için kullanılır. |
| Tarih parse etme | `date` alanı `date_ts` zaman damgasına çevrilir. | Burst davranışı, zaman penceresi ve koordinasyon cluster analizi için gereklidir. |

## 2. Temel Metin Yapısı Feature'ları

| Feature | Nasıl Hesaplanıyor? | Ne İçin Kullanılıyor? |
|---|---|---|
| `char_len` | Ham metin uzunluğu. | Çok kısa veya çok uzun metinleri ayırmak için. |
| `stripped_char_len` | Baş/son boşluklar temizlendikten sonraki uzunluk. | Boş metin ve sadece whitespace içeren kayıtları ayırmak için. |
| `empty_text` | Temizlenmiş metin uzunluğu 0 ise 1. | Boş içerik anomalisi. High risk kalabilir ama reason olarak açıkça `EMPTY_TEXT` verilir. |
| `short_text` | Boş olmayan ama 3 karakter veya daha kısa metin. | `yes`, `lol`, `No`, tek emoji gibi false-positive riskini düşürmek için ayrı tutulur. |
| `word_count` | Token sayısı. | Diğer oran feature'larının paydası olarak kullanılır. |
| `unique_word_count` | Benzersiz token sayısı. | Tekrar davranışını ölçmek için kullanılır. |
| `repetition_ratio` | `1 - unique_word_count / word_count`. | `FREE FREE FREE`, aynı kelime tekrarları ve spam şablonlarını yakalar. |
| `max_token_share` | En sık geçen token'ın toplam token içindeki payı. | Kısa spamlerde tek token baskınlığını yakalar. |
| `uppercase_ratio` | Büyük harf sayısı / harf sayısı. | `BUY NOW`, `URGENT`, `ACİL` gibi bağıran reklam/spam dilini yakalar. |
| `digit_ratio` | Rakam sayısı / karakter sayısı. | Para, yüzdelik, süre ve ödül vaatlerinde yardımcı sinyal. |
| `punctuation_density` | `!` ve `?` sayısı / karakter sayısı. | Aciliyet, panik ve dikkat çekme dilini yakalar. |
| `repeated_char_run` | Aynı karakterin 5+ kez art arda gelmesi. | Bot/spam tarzı abartılı yazım kalıplarını yakalar. |

## 3. Link, Hashtag ve Mention Feature'ları

| Feature | Nasıl Hesaplanıyor? | Ne İçin Kullanılıyor? |
|---|---|---|
| `url_count` | Metindeki URL benzeri pattern sayısı. | Phishing, promosyon ve dış site yönlendirme riskini ölçer. |
| `hashtag_count` | `#...` pattern sayısı. | Kampanya, trend manipülasyonu ve engagement bait sinyali. |
| `mention_count` | `@...` pattern sayısı. | Etiketleme, mention spam ve ağ yayılımı sinyali. |
| `link_signal_density` | `(2 * url_count + hashtag_count + mention_count) / word_count`. | Link/hashtag/mention yoğunluğunu tek ölçekte birleştirir. URL'ye daha yüksek ağırlık verilir. |

Bu grup özellikle canlı inference sırasında metnin sadece içerik olarak değil, kullanıcıyı bir aksiyona veya dış bağlantıya yönlendirip yönlendirmediğini ölçer.

## 4. Call-to-Action Feature'ları

| Feature | Ne Yapıyor? | Amacı |
|---|---|---|
| `call_to_action_count` | `buy`, `free`, `click`, `join`, `dm`, `takip`, `tikla`, `link`, `claim`, `verify`, `rt`, `vote` gibi CTA terimlerini sayar. | Manipülatif metinlerde sık görülen "hemen aksiyon al" baskısını yakalar. |
| CTA yoğunluğu | `call_to_action_count / word_count` üzerinden skora girer. | Uzun metinde tek CTA ile kısa metinde yoğun CTA'yı ayırmak için. |

CTA feature'ı tek başına her zaman manipülatif sayılmaz. Ancak link yoğunluğu, finansal vaat, phishing tehdidi, FOMO veya urgency ile birleştiğinde risk skorunu güçlü biçimde yükseltir.

## 5. Spam ve Manipülasyon Pattern Grupları

Bu katman, sadece yüzeysel uzunluk/tekrar feature'ları yerine doğrudan manipülatif içerik kalıplarını yakalar.

| Pattern Grubu | Örnek Sinyaller | Amacı |
|---|---|---|
| `financial_bait` | Ek gelir, para kazanma, garanti kar, crypto, forex, 100x. | Finansal dolandırıcılık ve yatırım manipülasyonu sinyali. |
| `phishing_threat` | Hesap kapatma, telif ihlali, appeal form, 24 saat. | Platform/banka taklidi ve hesap kapatma tehdidi. |
| `engagement_bait` | Giveaway, çekiliş, takip et, beğen, tag, link in bio, DM. | Takipçi, etkileşim ve yönlendirme manipülasyonu. |
| `adult_or_leak_bait` | Private photo, ifşa, leaked video, silinmeden izle. | Merak ve şok etkisiyle clickbait/spam. |
| `health_miracle` | Mucizevi çay, detox, spor yapmadan kilo verme. | Sağlık iddialı manipülatif reklam. |
| `prize_fee` | Tebrikler kazandınız, gift card, shipping fee. | Ödül/kargo ücreti dolandırıcılığı. |
| `debt_relief` | Kredi kartı borcu, uzman danışman. | Borçtan kurtarma vaadiyle lead/spam. |
| `wallet_verification` | Wallet, MetaMask, verify, USDT bonus. | Kripto cüzdan phishing senaryoları. |
| `trading_bot_bait` | Trading bot, 100% legit, ask me how, linktr.ee. | Bot gelir vaadi ve yatırım dolandırıcılığı. |
| `market_manipulation` | Gem alert, moon, Binance listing, liquidity locked. | Pump/dump veya kripto hype manipülasyonu. |
| `political_mobilization` | RT yap, tag'e destek, gizli belgeler, ana akım medya göstermez. | Koordineli politik amplifikasyon dili. |

Bu patternlerden üretilen iki özet feature vardır:

| Feature | Anlamı | Amacı |
|---|---|---|
| `scam_signal_count` | Tüm pattern eşleşmelerinin toplamı. | Metindeki toplam manipülasyon sinyali yoğunluğunu ölçer. |
| `scam_signal_groups` | Kaç farklı pattern grubunun aktif olduğunu sayar. | Tek bir kelimeye aşırı güvenmek yerine çoklu kanıt arar. |

## 6. Psikolojik Tetikleyici Feature'ları

Jüri anlatısı için en güçlü katmanlardan biri budur. Spam metinlerinin sadece teknik değil, davranışsal manipülasyon dili kullandığını gösterir.

| Trigger | Yakalanan Davranış | Örnek |
|---|---|---|
| `fomo_trigger` | Kaçırma korkusu. | Presale bitiyor, 100x olacak, before listing. |
| `urgency_bait` | Acil karar baskısı. | Son 15 dakika, 24 saat içinde, hemen, right now. |
| `loss_aversion_trigger` | Kayıptan kaçınma. | Fonlarınızı kaybedebilirsiniz, hesabınız kapanacak. |
| `social_proof_trigger` | Sosyal kanıt/sürü etkisi. | 50.000 kullanıcı katıldı, herkes kazanıyor, trusted by. |
| `authority_impersonation` | Otorite taklidi. | Meta support, resmi destek, banka, HR department. |

Bu feature'lar hem risk skorunda hem de `psychological_trigger_breakdown.png` grafiğinde kullanılır. Böylece sistem sadece "spam" demekle kalmaz, metnin hangi psikolojik baskı tekniğini kullandığını da açıklar.

## 7. Sentiment Feature'ı

| Feature | Nasıl Hesaplanıyor? | Amacı |
|---|---|---|
| `sentiment_abs` | `sentiment` değerinin mutlak değeri, 0-1 aralığında kırpılır. | Aşırı pozitif veya aşırı negatif duygusal dili yakalar. |
| `EXTREME_SENTIMENT` reason | `sentiment_abs >= 0.85` olduğunda eklenir. | Panik, öfke, coşku veya yoğun vaat içeren metinleri açıklamak için. |

Sentiment tek başına karar vermez; metinsel manipülasyon sinyalleriyle birleştiğinde yardımcı kanıt olarak çalışır.

## 8. Narrative Signature Feature'ları

| Feature | Ne Yapıyor? | Amacı |
|---|---|---|
| `text_signature` | Metinden stopword dışı en baskın tokenları çıkarır. | Benzer metinleri kaba şekilde temsil eder. |
| `keyword_signature` | `english_keywords` alanından temizlenmiş anahtar kelime imzası üretir. | Veri setindeki hazır keyword bilgisini coordination analizi için kullanır. |
| `narrative_signature` | Önce keyword imzası, yoksa text imzası kullanır. | Aynı anlatı/mesaj şablonlarını cluster'lamak için ana anahtardır. |
| Weak signature filtresi | `yes`, `lol`, `nice`, `birthday`, `jpeg`, `png`, `preview` gibi generic/media terimleri filtrelenir. | `happy birthday` veya görsel dosya adı gibi zayıf cluster'ların coordination risk üretmesini engeller. |

Bu katmanın amacı exact duplicate aramaktan daha esnektir: aynı kampanya veya anlatı farklı satırlarda tekrarlandığında bunu koordinasyon sinyali olarak yakalar.

## 9. Author-Level Davranış Feature'ları

Author feature'ları metnin ne söylediğinden çok, hesabın nasıl davrandığına bakar.

| Feature | Nasıl Hesaplanıyor? | Amacı |
|---|---|---|
| `author_post_count_sample` | Örneklemde author'ın post sayısı. | Yoğun gönderi davranışını ölçer. |
| `author_post_count_full` | Tüm parquet üzerinden author post sayısı. | Örneklem yanlılığını azaltır, gerçek hacmi yakalar. |
| `author_url_nunique_full` | Author'ın kaç farklı URL/platform kullandığı. | Çoklu platform veya alan yayılımını görmek için. |
| `author_theme_nunique_full` | Author'ın tema çeşitliliği. | Tek konuya aşırı odaklanmış kampanya davranışını yakalar. |
| `author_language_nunique_full` | Author'ın dil çeşitliliği. | Çok dilli bot/spam davranışı için yardımcı sinyal. |
| `author_burst_pairs` | Aynı author'ın 5 dakika içinde art arda attığı post çiftleri. | Burst posting davranışı. |
| `author_text_signature_nunique` | Author'ın benzersiz metin imzası sayısı. | Tekrar eden mesaj şablonlarını ölçer. |
| `author_keyword_signature_nunique` | Author'ın benzersiz keyword imzası sayısı. | Keyword düzeyinde tekrar davranışını ölçer. |

Author risk alt skorları:

| Alt Skor | Formül Mantığı | Amacı |
|---|---|---|
| `author_volume_risk` | Author post sayısı p75-p95 aralığına göre normalize edilir. | Aşırı yüksek hacimli hesapları yakalar. |
| `author_burst_risk` | `author_burst_pairs / max(sample_posts - 1, 1)`. | Kısa sürede sık paylaşım yapan hesapları yakalar. |
| `author_repeat_risk` | Metin veya keyword imza tekrar oranının maksimumu. | Copy-paste veya şablon paylaşımı ölçer. |
| `author_low_theme_diversity_risk` | `1 - theme_nunique / sample_posts`. | Tek temaya sıkışmış kampanya davranışını yakalar. |

Son author skoru:

```text
author_risk =
  0.35 * author_volume_risk
+ 0.25 * author_burst_risk
+ 0.25 * author_repeat_risk
+ 0.15 * author_low_theme_diversity_risk
```

Örneklemde 3'ten az postu olan author'larda bu skor düşürülür. Bunun amacı az verili author'lar için aşırı güvenli karar vermemektir.

## 10. Coordination Cluster Feature'ları

Coordination feature'ları, aynı anlatının farklı author'lar tarafından kısa zaman aralığında yayılıp yayılmadığını ölçer.

| Feature | Ne Ölçüyor? | Amacı |
|---|---|---|
| `cluster_size` | Aynı `narrative_signature` içindeki satır sayısı. | Anlatının hacmini ölçer. |
| `cluster_author_nunique` | Bilinen farklı author sayısı. | Tek hesaptan değil, çoklu aktörden yayılımı ölçer. |
| `cluster_platform_nunique` | Kaç farklı platform/domain görüldüğü. | Cross-platform yayılım sinyali. |
| `cluster_language_nunique` | Kaç farklı dilde görüldüğü. | Çok dilli koordinasyon ihtimalini gösterir. |
| `cluster_window_hours` | İlk ve son görülme arasındaki saat farkı. | Kısa sürede yayılımı yakalar. |
| `signature_term_count` | Signature içindeki anlamlı terim sayısı. | Tek kelimelik zayıf cluster'ları elemek için. |
| `cluster_content_similarity` | High-confidence mask geçerse 1. | Aynı anlatı benzerliğini confidence hesabına dahil eder. |

High-confidence coordination için kullanılan temel şartlar:

```text
cluster_size >= 4
cluster_author_nunique >= 3
signature_term_count >= 2
cluster_window_hours <= 72
```

Coordination risk alt skorları:

| Alt Skor | Mantık | Amacı |
|---|---|---|
| `coord_size_risk` | Cluster büyüklüğü log ölçeğinde normalize edilir. | Büyük cluster'ları daha riskli sayar. |
| `coord_author_spread_risk` | Farklı author sayısı normalize edilir. | Çoklu hesap koordinasyonunu yakalar. |
| `coord_platform_spread_risk` | Farklı platform sayısı normalize edilir. | Cross-platform yayılımı ölçer. |
| `coord_time_compactness_risk` | `24 / (cluster_window_hours + 1)` kırpılır. | Kısa zaman penceresindeki yayılımı daha riskli sayar. |

Son coordination skoru:

```text
coordination_risk =
  0.40 * coord_size_risk
+ 0.35 * coord_author_spread_risk
+ 0.15 * coord_platform_spread_risk
+ 0.10 * coord_time_compactness_risk
```

Cluster confidence skoru:

```text
cluster_confidence_score =
  0.35 * coord_author_spread_risk
+ 0.25 * coord_time_compactness_risk
+ 0.20 * coord_size_risk
+ 0.10 * signature_richness
+ 0.10 * cluster_content_similarity
```

Bu skor, coordination risk'ten ayrı olarak "bu cluster jüriye gösterilecek kadar savunulabilir mi?" sorusuna hizmet eder.

## 11. Content Risk Skoru

Content risk, metin içinden çıkarılan yüzeysel, semantik ve psikolojik sinyallerin birleşimidir.

Temel ağırlıklı formül:

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

Ek floor kuralları:

| Kural | Etki | Amacı |
|---|---|---|
| Link + punctuation + CTA birlikte çok yüksekse | Risk en az `0.82`. | Açık spam paketini yakalamak. |
| Phishing, wallet verification, trading bot, market manipulation gibi güçlü lure varsa | Risk en az `0.72`. | Kritik manipülasyon senaryolarını kaçırmamak. |
| Politik mobilizasyon güçlü ise | Risk en az `0.66`. | Koordineli amplifikasyon dilini High bandına yaklaştırmak. |
| Daha zayıf ama çoklu lure varsa | Risk en az `0.50`. | İnceleme havuzuna almak. |
| Boş metin | Risk en az `0.92`. | Veri anomalisi olarak yüksek riskli işaretlemek. |
| Repeated character run | Risk en az `0.55`. | Spam yazım anomalilerini Review bandına almak. |

## 12. Final Risk Skoru ve Risk Band

Final skor üç ana bileşenden oluşur:

```text
risk_score =
  weighted_average(content_risk, author_risk, coordination_risk)
```

Kullanılan ağırlıklar:

```text
content_weight = 0.35
author_weight = 0.30
coordination_weight = 0.35
```

Önemli detay: Author veya coordination bilgisi güvenilir değilse denominator yeniden normalize edilir. Yani eksik metadata, skoru haksız yere düşürmez veya yükseltmez.

Risk band eşikleri:

| Band | Eşik | Label |
|---|---:|---|
| `High` | `risk_score >= 0.65` | `Manipulative` |
| `Review` | `0.45 <= risk_score < 0.65` | `Organic`, ama analist incelemesine uygun |
| `Low` | `< 0.45` | `Organic` |

Bu ayrım sunum açısından kritiktir: Binary label korunur ama `Review` bandı modelin şüpheli fakat kesin manipülatif demediği alanı görünür kılar.

## 13. Reason Code ve Açıklanabilirlik Feature'ları

Her satır için `reason_codes` üretilir. Bunlar modelin kararını jüriye açıklamak için kullanılır.

Örnek reason code'lar:

| Reason Code | Anlamı |
|---|---|
| `EMPTY_TEXT` | Metin boş veya whitespace. |
| `SHORT_TEXT_LOW_EVIDENCE` | Kısa ama tek başına güçlü kanıt değil. |
| `HIGH_TOKEN_REPETITION` | Aşırı kelime tekrarı var. |
| `LINK_HASHTAG_MENTION_DENSE` | Link/hashtag/mention yoğunluğu yüksek. |
| `CALL_TO_ACTION_LANGUAGE` | Aksiyon çağrısı yoğun. |
| `FINANCIAL_OR_CRYPTO_BAIT` | Finansal/kripto vaat dili var. |
| `PHISHING_URGENCY_THREAT` | Hesap kapatma veya telif tehdidi var. |
| `FOMO_TRIGGER` | Kaçırma korkusu tetikleniyor. |
| `LOSS_AVERSION_TRIGGER` | Kayıp korkusu kullanılıyor. |
| `SOCIAL_PROOF_TRIGGER` | Sosyal kanıt/sürü etkisi kullanılıyor. |
| `AUTHORITY_IMPERSONATION` | Otorite veya kurum taklidi var. |
| `COORDINATED_NARRATIVE_CLUSTER` | Benzer anlatı koordinasyon cluster'ında görülüyor. |

Bu katman sayesinde sistem kara kutu gibi çalışmaz; her kararın gerekçesi raporlanabilir.

## 14. Text-Only NLP Destek Feature'ları

Jüri canlı testte sadece `original_text` verebileceği için ek bir text-only NLP destek katmanı eklendi. Bu katman LLM veya embedding modeli değildir; dışarıdan etiketli veri kullanmaz.

### Eğitim Mantığı

Pseudo-label yaklaşımı kullanılır:

| Sınıf | Seçim Kuralı | Amaç |
|---|---|---|
| Pozitif pseudo-label | `risk_score >= 0.65`, boş olmayan metin, anlamlı signature, güçlü reason. | Mevcut sistemin en güvenli manipülasyon örneklerinden metin örüntüsü öğrenmek. |
| Negatif pseudo-label | `risk_score <= 0.20`, `LOW_CONTENT_RISK`, boş olmayan metin, güçlü reason yok. | Güvenli organik metin örüntülerini öğrenmek. |

### NLP Feature'ları

| Feature Tipi | Açıklama | Amacı |
|---|---|---|
| Word unigram | Tekil kelime hash feature'ları. | Riskli kelimeleri öğrenmek. |
| Word bigram | Ardışık iki kelime hash feature'ları. | `wallet verification`, `shipping fee`, `trading bot` gibi ifadeleri yakalamak. |
| Character 3/4/5-gram | Karakter düzeyi hash feature'ları. | Yazım varyasyonları, Türkçe/İngilizce karışımı ve küçük değişikliklere dayanıklılık. |
| Hash bucket | Feature'lar sabit boyutlu hash uzayına yazılır. | Ek bağımlılık olmadan hızlı ve taşınabilir model üretmek. |

### Skorlama Mantığı

Her hash feature için pozitif/negatif doküman frekansından log-odds ağırlığı hesaplanır:

```text
feature_log_odds = log(pos_rate) - log(neg_rate)
nlp_text_risk = sigmoid(mean(feature_log_odds))
```

Bu skor doğrudan final kararı ezmez. Rule layer'da en az bir risk kanıtı varsa destekleyici sinyal olarak kullanılır. Bu gate'in amacı organik metinleri sadece pseudo-label model yüksek gördü diye yanlışlıkla High bandına taşımamaktır.

## 15. Sunum ve Pazarlama Artifact Feature'ları

Bu feature'lar model kararından çok, modeli kanıtlamak ve jüriye anlatmak için üretilir.

| Artifact Feature | Dosya/Grafik | Amacı |
|---|---|---|
| Risk dağılımı | `risk_score_histogram.png` | Modelin seçici davrandığını göstermek. |
| Dil-platform yoğunluğu | `risk_map_language_platform.png` | Riskin hangi dil/platform kırılımlarında yoğunlaştığını göstermek. |
| Dil bazlı manipulative oranı | `language_manipulative_share.png` | Her dildeki manipulative hacmi ve oranını birlikte göstermek. |
| Platform-normalized risk | `platform_normalized_risk.csv/png` | Büyük platformların ham hacim avantajını dengelemek. |
| Author summary | `top_risk_authors.csv/png` | Riskli author davranışlarını hacim, burst ve tekrar üzerinden göstermek. |
| Cluster confidence | `coordination_confidence_bubble.png` | Coordination cluster'larının güven düzeyini görselleştirmek. |
| Reason breakdown | `reason_code_breakdown.png` | High kararlarının hangi açıklanabilir sinyallere dayandığını göstermek. |
| Psychological trigger breakdown | `psychological_trigger_breakdown.png` | Spam dilindeki psikolojik manipülasyon türlerini göstermek. |
| Evidence quality summary | `evidence_quality_summary.png` | High-risk kararların ne kadarının güçlü kanıtla desteklendiğini göstermek. |
| Live benchmark | `live_inference_benchmark.csv/png` | Gizli jüri metinlerine hazır canlı inference davranışını göstermek. |

## 16. Bu Feature Engineering Yaklaşımının Güçlü Yanları

- Etiketsiz veri için uygundur; supervised accuracy iddiası gerektirmez.
- Canlı gelen tekil metinlerde çalışır.
- Metadata varsa author ve coordination sinyalleriyle kararı güçlendirir.
- Her karar için reason code üretir.
- Psikolojik tetikleyiciler sayesinde jüriye daha anlaşılır bir hikaye sunar.
- Generic/media keyword cluster'ları filtrelenerek false-positive riski azaltılır.
- `Review` bandı sayesinde şüpheli ama kesin olmayan alanlar ayrı gösterilir.

## 17. Sınırlamalar ve Dikkat Edilmesi Gerekenler

- Gerçek label olmadığı için `accuracy`, `precision`, `recall` gibi metrikler iddia edilmemelidir.
- Pattern tabanlı sinyaller dil kapsamına bağlıdır; yeni diller veya yeni spam jargonları için sözlük genişletmek gerekir.
- Pseudo-label NLP modeli mevcut skorlamadan öğrenir; bu nedenle ground-truth supervised model gibi sunulmamalıdır.
- Author ve coordination feature'ları canlı tek metin testinde metadata yoksa kullanılamaz; bu durumda sistem content + NLP destek katmanına düşer.
- Boş metinler veri anomalisi olarak yüksek risk alır; sunum örneklerinde boş metinler özellikle filtrelenmelidir.

## 18. Jüriye Önerilen Anlatım

Kısa anlatım:

> Bu projede manipülasyonu tek bir model skoruna indirgemedik. Metnin yapısal özelliklerini, spam/pishing patternlerini, psikolojik tetikleyicileri, author davranışını ve koordinasyon cluster'larını ayrı ayrı feature olarak çıkardık. Final skor, hangi kanıtların mevcut olduğuna göre normalize edilen açıklanabilir bir risk skorudur. Canlı jüri metinlerinde metadata yoksa sistem text-only content ve pseudo-label NLP destek katmanıyla karar üretir; metadata varsa author ve coordination kanıtlarıyla kararı güçlendirir.

Sunumda vurgulanması gereken ana cümle:

> Etiketsiz sosyal medya verisinde hedefimiz accuracy iddiası değil; açıklanabilir, seçici, canlı inference'a hazır ve kanıt üreten bir manipülasyon risk skorlama pipeline'ı kurmaktır.

# Case Studies: Savunulabilir Risk Örnekleri

Bu dosya, jüri sunumunda somut örnek anlatımı için üretilmiştir. Örnekler gerçek skorlanmış satırlardan seçilir; sentetik benchmark yalnızca canlı inference hazırlığını göstermek için ayrı artifact olarak tutulur.

## Case Study 1: Finansal/Kripto veya Trading Scam

| Alan | Değer |
|---|---|
| Risk score | 0.788 |
| Risk band | High |
| Label | Manipulative |
| Reason codes | LINK_HASHTAG_MENTION_DENSE;CALL_TO_ACTION_LANGUAGE;FINANCIAL_OR_CRYPTO_BAIT;HIGH_AUTHOR_VOLUME;BURST_POSTING_WITHIN_5_MIN;AUTHOR_REPEATS_TEXT_OR_KEYWORDS |
| Language | en |
| Platform | x.com |
| Theme | Technology |
| Author | 8d501770c735a0e... |
| Date | 2024-11-20T20:31:26.000Z |
| Narrative signature | kw:channel|channel telegram|daily|daily forex|forex signal|signal|signal everyday|telegram |

**Metin preview**

> Free 3 to 5 signals available everyday in my channel Telegram http://t.me/Goldpro010 Get a daily Forex Signal everyday 98% Accuracy in Market Gold and Curruncy E http://t.me/Goldpro010 #GOLD #xauusd $xauusd

**Neden yakalandı?** Finansal vaat, kripto/trading dili, CTA veya link yönlendirmesi gibi güçlü reason code'lar aynı metinde birikiyor.

**Jüriye anlatım cümlesi:** Bu örnek, sistemin yalnızca kelime eşleşmesine değil; finansal bait, CTA ve risk bandı birleşimine bakarak manipülatif ticari vaadi yakaladığını gösterir.

**Sınırlama notu:** Bu karar etiketli doğruluk iddiası değildir; açıklanabilir risk sinyallerinin yoğunlaştığı bir örnektir.

## Case Study 2: Kısa Zaman Penceresinde Koordineli Anlatı

| Alan | Değer |
|---|---|
| Risk score | 0.680 |
| Risk band | High |
| Label | Manipulative |
| Reason codes | VERY_SHORT_TEXT;LINK_HASHTAG_MENTION_DENSE;COORDINATED_NARRATIVE_CLUSTER;MULTI_AUTHOR_SAME_NARRATIVE;SHORT_TIME_WINDOW_CLUSTER |
| Language | pl |
| Platform | x.com |
| Theme | Social |
| Author | c2bfafb9724b3db... |
| Date | 2024-11-14T20:13:55.000Z |
| Narrative signature | kw:#paws|paws|share |
| Cluster size | 8 |
| Cluster author count | 8 |
| Cluster platform count | 1 |
| Cluster language count | 1 |
| Cluster window hours | 0.32 |
| Cluster confidence | 0.940 |
| Coordination risk | 0.850 |

**Metin preview**

> Share your #PAWS

**Neden yakalandı?** Aynı narrative signature kısa zaman penceresinde birden fazla author tarafından paylaşılıyor; cluster boyutu, author yayılımı ve zaman kompaktlığı confidence skorunu destekliyor.

**Jüriye anlatım cümlesi:** Bu örnek, sistemin tekil metinden öteye geçip aynı anlatının kısa sürede çoklu hesaplarla yayıldığı kampanya benzeri davranışı görünür kıldığını gösterir.

**Sınırlama notu:** Sistem çeviri semantiği iddia etmez; aynı signature içinde çok dilli yayılım varsa `cluster_language_nunique` ile bunu görünür kılar.

## Case Study 3: Psikolojik Manipülasyon Tetikleyicisi

| Alan | Değer |
|---|---|
| Risk score | 0.743 |
| Risk band | High |
| Label | Manipulative |
| Reason codes | CALL_TO_ACTION_LANGUAGE;ENGAGEMENT_OR_LINK_BAIT;FOMO_TRIGGER;WALLET_VERIFICATION_BAIT;HIGH_AUTHOR_VOLUME;BURST_POSTING_WITHIN_5_MIN;LOW_AUTHOR_TOPIC_DIVERSITY |
| Language | en |
| Platform | x.com |
| Theme | Cryptocurrency |
| Author | da39a3ee5e6b4b0... |
| Date | 2024-11-19T19:41:45.000Z |
| Narrative signature | kw:angeles|duluth|lfb weverse|momentum|nct|paypal|presale code|toronto rosemont |

**Metin preview**

> wts lfb weverse membership presale code — NCT 127 THE MOMENTUM • $8 each stop • Paypal duluth newark toronto rosemont san antonio los angeles ~ dm to claim!

**Neden yakalandı?** FOMO, aciliyet, kayıptan kaçınma, sosyal kanıt veya otorite taklidi gibi davranışsal manipülasyon sinyalleri reason code olarak yakalanıyor.

**Jüriye anlatım cümlesi:** Bu örnek, sistemin spam dilini yalnızca teknik sinyallerle değil, psikolojik baskı mekanizmalarıyla da açıklayabildiğini gösterir.

**Sınırlama notu:** Psikolojik trigger'lar destekleyici kanıttır; final karar risk bandı ve diğer content/author/coordination sinyalleriyle birlikte değerlendirilir.

## Canlı Inference Benchmark Notu

`live_inference_benchmark.csv` sentetik ama gerçekçi senaryolarda canlı tahmin hazırlığını gösterir. Bu case study dosyasındaki örnekler ise mümkün olduğunca gerçek skorlanmış satırlardan seçilir.

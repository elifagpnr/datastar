Datathon Yol Planı
Özet
Problem etiketli sınıflandırma değil; gözetimsiz, açıklanabilir manipülasyon/anomali skoru kuracağız.
Teslimat MVP’si: çalışan notebook/script + 5-10 slayt pitch deck + canlı predict_live() fonksiyonu.
Ortam varsayımı: yerel laptop, yaklaşık 12 saat. Bu yüzden ağır LLM/embedding yerine hızlı, RAM-dostu, anlatılabilir skor sistemi.
Temel Strateji
İlk hücrede minimum bağımlılıkları hazırla: duckdb, pyarrow veya sadece duckdb, pandas, numpy, matplotlib.
Tüm 5M satırı pandas’a alma. duckdb.read_parquet() ile:
örneklem: 200k-500k satır model/EDA için,
full dataset: sadece groupby agregasyonları için.
Ana çıktı kolonları:
risk_score: 0-1 manipülasyon riski,
organic_score = 1 - risk_score,
label: Manipulative if risk_score >= 0.65, else Organic,
reason_codes: karar açıklamaları.
Skorlama Yaklaşımı
İçerik riski: metin uzunluğu uç değerleri, tekrar oranı, link/hashtag/mention yoğunluğu, büyük harf/ünlem oranı, boş/çok kısa metin, ekstrem sentiment.
Yazar riski: author başına post sayısı, kısa aralıklarla paylaşım, tekrar eden metin/keyword oranı, düşük tema çeşitliliği.
Koordinasyon riski: aynı/benzer english_keywords veya normalize edilmiş metnin farklı author’lar tarafından kısa zaman aralığında paylaşılması.
Nihai skor:
metadata varsa: 0.35 content + 0.30 author + 0.35 coordination,
canlı gizli testte sadece metin verilirse: mevcut bileşenlerin ağırlıkları yeniden normalize edilir.
english_keywords anlatı kümeleri için kullanılacak; canlı metin için basit regex/token temelli text özellikleriyle fallback yapılacak.
Görselleştirme ve Sunum
Manipülasyon haritası olarak notebook’tan şu grafikler üretilecek:
şüpheli içeriklerin language x url heatmap’i,
en riskli platform/dil/tema kombinasyonları,
saatlik/günlük şüpheli hacim spike grafiği,
örnek manipülatif kümeler: aynı keyword imzası + farklı author + yakın zaman.
Pitch deck akışı:
Problem ve veri,
Etiket olmadığı için unsupervised yaklaşım,
Feature grupları,
Skor formülü,
Manipülasyon haritası,
Örnek kümeler,
Canlı inference demo,
Limitasyonlar ve geliştirme alanları.
Canlı Demo Arayüzü
Notebook/script içinde şu fonksiyon net çalışmalı:
predict_live(text, language=None, url=None, author_hash=None, date=None, english_keywords=None) -> dict
Dönüş:
label,
risk_score,
organic_score,
top_reasons,
used_features.
Gizli testte metadata verilmezse fonksiyon sadece metinden karar verir ve bunu açıklamada belirtir.
Test Planı
Notebook baştan sona restart sonrası çalışmalı.
1k satırlık smoke test, 100k satırlık hızlı pipeline testi yapılmalı.
Boş metin, çok kısa metin, uzun spam metni, normal haber/metin, tekrar eden kampanya metni senaryoları denenmeli.
Sentetik test: aynı metni farklı author/time ile çoğaltınca coordination risk yüksek çıkmalı.
Sunumdan önce canlı predict_live() 5 örnekle prova edilmeli.
Zaman Planı
0-1 saat: bağımlılıklar, parquet okuma, örneklem.
1-3 saat: EDA ve agregasyonlar.
3-6 saat: skor sistemi ve reason codes.
6-8 saat: grafikler/manipülasyon haritası.
8-10 saat: canlı inference fonksiyonu ve notebook temizliği.
10-11 saat: pitch deck.
11-12 saat: prova, hata düzeltme, teslim formatı kontrolü.
Varsayımlar
Etiket yok, bu yüzden başarı metriği “accuracy” değil; açıklanabilirlik, tutarlı skor, iyi görsel bulgu ve çalışan canlı demo.
Web dashboard yerine notebook içi grafikler + sunum yeterli kabul edilecek.
ads.ipynb şu an boş görünüyor; çalışmayı temiz bir notebook/script olarak sıfırdan kurgulamak en güvenli yol.
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import unicodedata
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd


DATA_PATH = Path("dataset/datathonFINAL.parquet")
ARTIFACT_DIR = Path("artifacts")
RISK_THRESHOLD = 0.65
REVIEW_THRESHOLD = 0.45
MODEL_VERSION = "text-calibration-v6-credential-finance-patterns"
NLP_TEXT_MODEL_VERSION = "pseudo-label-hash-ngram-v1"
NLP_HASH_DIM = 2 ** 17
NLP_MAX_DOCS_PER_CLASS = 40_000
NLP_MIN_DOCS_PER_CLASS = 50
NLP_RANDOM_SEED = 42

TEXT_COL = "original_text"
TEST_ID_COL = "test_id"
KEYWORDS_COL = "english_keywords"
SENTIMENT_COL = "sentiment"
AUTHOR_COL = "author_hash"
DATE_COL = "date"
URL_COL = "url"

TEXT_COLUMN_CANDIDATES = {
    "original_text",
    "originaltext",
    "text",
    "content",
    "message",
    "body",
    "post",
    "tweet",
    "caption",
    "comment",
}

ID_COLUMN_CANDIDATES = {
    TEST_ID_COL,
    "id",
    "row_id",
    "case_id",
    "sample_id",
}

METADATA_COLUMN_CANDIDATES = {
    "language": {"language", "lang", "dil"},
    URL_COL: {URL_COL, "link", "domain", "platform", "platform_domain"},
    AUTHOR_COL: {AUTHOR_COL, "author", "user", "user_id", "account", "account_id"},
    DATE_COL: {DATE_COL, "created_at", "timestamp", "time", "datetime"},
    KEYWORDS_COL: {KEYWORDS_COL, "keywords", "english_keyword", "keyword"},
}

REQUIRED_COLUMNS = [
    TEXT_COL,
    KEYWORDS_COL,
    SENTIMENT_COL,
    "main_emotion",
    "primary_theme",
    "language",
    URL_COL,
    AUTHOR_COL,
    DATE_COL,
]

URL_PATTERN = (
    r"(?:https?://|www\.|[A-Za-z0-9.-]+\."
    r"(?:com|net|org|io|ai|co|edu|gov|me|ly|ee|xyz|site|online|top|link|app|dev|finance|biz|info|shop|live|click|ru|uk|tr|de|fr|es|it|br|in|jp|cn|kr|au|ca))"
)
TOKEN_RE = re.compile(r"[\w$#@']+", re.UNICODE)
WORD_RE = re.compile(r"[\w']+", re.UNICODE)
TURKISH_CHAR_MAP = str.maketrans(
    {
        "ı": "i",
        "İ": "i",
        "ğ": "g",
        "Ğ": "g",
        "ü": "u",
        "Ü": "u",
        "ş": "s",
        "Ş": "s",
        "ö": "o",
        "Ö": "o",
        "ç": "c",
        "Ç": "c",
    }
)

STOPWORDS = {
    "the",
    "and",
    "for",
    "you",
    "are",
    "that",
    "this",
    "with",
    "from",
    "have",
    "has",
    "will",
    "your",
    "our",
    "was",
    "were",
    "but",
    "not",
    "can",
    "all",
    "bir",
    "ve",
    "bu",
    "da",
    "de",
    "ile",
    "icin",
    "cok",
}

CALL_TO_ACTION_TERMS = {
    "buy",
    "free",
    "click",
    "join",
    "watch",
    "follow",
    "subscribe",
    "share",
    "retweet",
    "dm",
    "promo",
    "giveaway",
    "airdrop",
    "win",
    "discount",
    "deal",
    "limited",
    "urgent",
    "breaking",
    "hemen",
    "takip",
    "katil",
    "katilmak",
    "izle",
    "tikla",
    "tiklayin",
    "tiklamak",
    "link",
    "bio",
    "profil",
    "profile",
    "form",
    "doldur",
    "doldurun",
    "begen",
    "etiketle",
    "tag",
    "pay",
    "claim",
    "order",
    "siparis",
    "yaz",
    "gt",
    "f4f",
    "bedava",
    "ask",
    "verify",
    "verification",
    "wallet",
    "metamask",
    "bonus",
    "usdt",
    "telegram",
    "vip",
    "rt",
    "yayin",
    "yayinla",
    "yayinlayin",
    "kopyala",
    "kopyalayip",
    "destek",
    "versin",
    "vote",
}

CALL_TO_ACTION_TERMS.update(
    {
        "comprar",
        "compra",
        "gratis",
        "oferta",
        "clic",
        "clique",
        "cliquez",
        "kaufen",
        "kostenlos",
        "klicken",
        "acheter",
        "gratuit",
        "compra",
        "clicca",
        "beli",
        "menang",
        "klaim",
        "verifikasi",
        "kharido",
        "muft",
        "jeeto",
        "kamao",
        "abhi",
        "turant",
        "verificar",
        "verifique",
        "verifier",
        "verifizieren",
        "verifica",
        "reclamar",
        "resgatar",
        "beanspruchen",
        "subito",
        "sofort",
        "segera",
        "entra",
        "ottieni",
        "tautan",
        "enlace",
        "lien",
        "sorteo",
        "sorteio",
        "chame",
        "direct",
        "agora",
        "entre",
    }
)

SCAM_PATTERN_GROUPS = {
    "financial_bait": [
        r"\b(ek gelir|extra income|ayda|month extra income|gunde|daily|gunluk)\b",
        r"\b(kazanmak|kazanmak ister|earn|make|profit|guaranteed profit|guaranteed profits)\b",
        r"\b(\d[\d.,]*\s*(tl|usd|dolar|dollar|\$))\b",
        r"\b(crypto|kripto|vip crypto|vip kripto|forex|signal|sinyal)\b",
        r"\b(trading bot|trade bot|bot|portfolio|portfoy|piyasa|piyasalar|stock alerts|paid group)\b",
        r"\b(private investment club|investment club|members are doubling portfolios|doubling portfolios|double your money)\b",
        r"\b(profit signals|secret stock alerts|before the public hears|without effort|no risk|risk free)\b",
        r"\b(100x|x5|gem alert|dogemoon|moon|binance|liquidity locked|doxxed)\b",
    ],
    "phishing_threat": [
        r"\b(account|hesabiniz|hesabin).{0,60}\b(suspend|suspended|kapat|kapatilacaktir|closed)\b",
        r"\b(copyright|telif|ihlali|infringement|appeal|itiraz)\b",
        r"\b(24 hours|24 saat|formu doldur|fill out the form)\b",
        r"\b(cloud storage|banking session|password expires|suspicious activity|security alert)\b",
        r"\b(log in|login|confirm ownership|confirm your login|re-enter|enter your credentials|card details|card photo)\b",
        r"\b(deleted unless|permanent account closure|keep your account active|restore access)\b",
    ],
    "engagement_bait": [
        r"\b(giveaway|cekilis|hediye ceki|gift card|follow|takip|begen|tag|etiketle)\b",
        r"\b(like this post|like the post|like and share|like, share)\b",
        r"\b(followers|takipci|f4f|gt|yorumlara|comments)\b",
        r"\b(link in bio|link bioda|profilimdeki link|profile link|profile now|profilime gir|dm|dm'ye)\b",
    ],
    "adult_or_leak_bait": [
        r"\b(private photo|private photos|ozel fotograf|ozel foto|ifsa|ifsa videosu|leaked video)\b",
        r"\b(sizan|internete sizan|shocking leaked|silinmeden|before it gets deleted)\b",
    ],
    "health_miracle": [
        r"\b(mucizevi|miracle|detox|cay|tea|kilo verdim|lost \d+ pounds|without exercising|spor yapmadan)\b",
    ],
    "prize_fee": [
        r"\b(congratulations|tebrikler|kazandiniz|you've won|you won|odul|prize)\b",
        r"\b(kargo ucreti|shipping fee|claim your prize|odulunuzu almak)\b",
    ],
    "debt_relief": [
        r"\b(kredi karti borc|credit card debt|borclarinizdan kurtulmak|getting rid of your credit card debt)\b",
        r"\b(uzman danisman|expert consultant|consultants)\b",
    ],
    "urgency_bait": [
        r"\b(hemen|right now|urgent|limited|silinmeden|before it gets deleted|24 saat|24 hours)\b",
        r"\b(buy now|verify now|claim now|join now|click now|dm now|enter now|connect now)\b",
        r"\b(acil|cokmeden|son \d+ kisi|verification required|wallet.*required|required.*verification|before it hits)\b",
        r"\b(son sans|last chance|ending soon|ends soon|final call|kacirma|kacirmayin)\b",
        r"\b(son \d+ dakika|next \d+ minutes|only \d+ left|kontenjan son)\b",
    ],
    "fomo_trigger": [
        r"\b(on satis|presale|pre sale|early access|sold out|bitiyor|tukeniyor)\b",
        r"\b(bu kripto firlayacak|firlaya?cak|next gem|gem alert|moonshot|to the moon|100x|x100|x5)\b",
        r"\b(before it hits|before listing|binance listing|airdrop ends|limited supply)\b",
    ],
    "loss_aversion_trigger": [
        r"\b(kaybedebilirsiniz|kaybedeceksiniz|fonlarinizi kaybed|verilerinizi kaybed|hesabiniz kapan)\b",
        r"\b(lose your funds|lose access|your funds are at risk|data will be deleted|account will be suspended)\b",
        r"\b(askiya alindi|askiya alinacak|kapatilacaktir|sonlandirilacak|blocked unless)\b",
    ],
    "social_proof_trigger": [
        r"\b(\d[\d,.]*\s*(k|bin|m|million|milyon)?\s*(people|users|kisi|uye|yatirimci))\b",
        r"\b(herkes kazaniyor|everyone is joining|trusted by|binlerce kisi|milyonlarca kisi)\b",
        r"\b(sahte begeni|retweet|rt yap|trend yap|tag'e destek|tag e destek|yorumlara yaz)\b",
        r"\b(proof|screenshots|testimonial|kanit|dekont|kazanc kaniti)\b",
    ],
    "authority_impersonation": [
        r"\b(platform support|support team|official support|verification team|security team)\b",
        r"\b(banka|bankasi|bank|ik departmani|hr department|human resources)\b",
        r"\b(twitter support|x support|instagram support|meta support|binance support|metamask support)\b",
        r"\b(resmi|official).{0,30}\b(destek|support|form|verification)\b",
    ],
    "wallet_verification": [
        r"\b(wallet|metamask|trust wallet|phantom)\b",
        r"\b(verify|verification|required|claim|bonus|usdt)\b",
        r"\b(account).{0,40}\b(verify|verification|required)\b",
    ],
    "trading_bot_bait": [
        r"\b(trading bot|trade bot|bot)\b",
        r"\b(made\s+\$?\d[\d,.]*|just made|100% legit|ask me how)\b",
        r"\b(linktr\.ee|linktree|scamlink)\b",
        r"\b(100x|gem alert|liquidity locked|doxxed|binance|moon)\b",
    ],
    "market_manipulation": [
        r"\b(100x|gem alert|dogemoon|moon|airdrop|binance|liquidity locked|doxxed)\b",
        r"\b(coin is about to explode|about to explode|insider source|listing tomorrow|influencers.*reveal)\b",
        r"\b(x5|portfoy|portfolio|piyasalar|telegram grubu|vip telegram|vip)\b",
    ],
    "political_mobilization": [
        r"\b(tag'?e destek|tag e destek|destek versin|rt yap|retweet yap|sesimizi duyur)\b",
        r"\b(linki kopyalayip yayin|linki kopyala|kopyalayip yayin|yayin)\b",
        r"\b(gizli belgeler|belgeler sizdi|skandal|ana akim medya|asla gostermez|yalan haber|susturamazsiniz)\b",
        r"\b(uyan|oyun buyuk|gercekleri gormek|herkes bu tag)\b",
        r"\b(everyone needs to see|vote them out|see this video|must repost|truth gets deleted|deleted by the media)\b",
    ],
    "credential_phishing": [
        r"\b(password|credentials|login details|card details|card photo|banking session|bank account|account active)\b",
        r"\b(upload your id|enter your password|enter your credentials|re-enter your card|confirm your login)\b",
        r"\b(verify security|security verification|suspicious activity|keep your account active|restore access)\b",
    ],
}

MULTILINGUAL_SCAM_PATTERN_GROUPS = {
    "financial_bait": [
        r"\b(ingreso extra|dinero rapido|ganar dinero|renda extra|ganhar dinheiro|ganhe dinheiro|dinheiro rapido|dinheiro facil)\b",
        r"\b(penghasilan tambahan|uang cepat|revenu supplementaire|argent rapide|schnelles geld|geld verdienen)\b",
        r"\b(guadagno extra|soldi facili|paisa kamao|paise kamao|kamai kare|paisa kamayen)\b",
        r"\b(criptomoneda|criptomoeda|kripto|cryptomonnaie|krypto|kryptowahrung|criptovaluta)\b",
        r"\b(bot de trading|robot de trading|robo secreto|trading otomatis|handelsbot|bot di trading)\b",
        r"\b(sinais de lucro|sem risco|grupo fechado|groupe vip|gagnez de l'argent|sans effort|segnali garantiti|gruppo crypto)\b",
        r"\b(occasione unica|ultima chiamata|entra nel gruppo|coin prima|grupo vip de inversion|duplica tu dinero)\b",
        r"(اربح|ارباح|دخل اضافي|اكسب|عملات رقمية|تداول|بوت التداول|استرداد مالي)",
        r"(कमाएं|कमाई|पैसा|क्रिप्टो|ट्रेडिंग बॉट)",
        r"(稼げる|副収入|暗号資産|仮想通貨|取引ボット|投資グループ|利益を保証)",
        r"(수익|부업|암호화폐|코인|거래 봇)",
        r"(заработай|доход|крипто|криптовалюта|торговый бот)",
        r"(赚钱|收入|副业|加密货币|交易机器人)",
    ],
    "phishing_threat": [
        r"\b(cuenta|conta|akun|compte|konto|conto|profilo).{0,60}\b(suspendida|suspendido|suspensa|ditangguhkan|diblokir|suspendu|gesperrt|sospeso|bloccato)\b",
        r"\b(confirme su cuenta bancaria|cuenta bancaria|perdera el acceso|perdera acceso)\b",
        r"\b(kata sandi|verifikasi keamanan|geben sie.*passwort|konto geschlossen)\b",
        r"\b(derechos de autor|direitos autorais|hak cipta|droit d'auteur|urheberrecht|violacion|infraccion|infracao|violazione)\b",
        r"\b(24 horas|24 heures|24 stunden|24 jam|24 ore|formulario de apelacion|formulaire d'appel|formulir banding)\b",
        r"(حسابك.*(تعليق|ايقاف)|سيتم.*(تعليق|ايقاف).*حسابك|حقوق النشر|خلال 24 ساعة|نموذج الاعتراض|بيانات البطاقة)",
        r"(आपका खाता.*निलंबित|कॉपीराइट|24 घंटे)",
        r"(アカウント.*停止|著作権侵害|24時間.*異議)",
        r"(계정.*정지|저작권 침해|24시간|비밀번호.*카드 정보|카드 정보.*비밀번호)",
        r"(аккаунт.*заблокирован|авторск.*прав|24 часа)",
        r"(账户.*停用|账号.*暂停|版权侵权|24小时)",
    ],
    "engagement_bait": [
        r"\b(sorteo|sorteio|seguir|siga|etiqueta|curtir|ikuti|bagikan|suivez|suivre|folgen|teilen|seguire)\b",
        r"\b(enlace en bio|link na bio|tautan di bio|lien dans la bio|link in der bio|lien en bio)\b",
        r"(رابط في البايو|تابع|اعجاب|شارك)",
        r"(फॉलो|लाइक|बायो में लिंक|शेयर)",
        r"(プロフィールのリンク|フォロー|いいね|抽選)",
        r"(프로필 링크|팔로우|좋아요|공유)",
        r"(ссылка в био|подпишись|лайк|розыгрыш)",
        r"(简介链接|主页链接|关注|点赞|抽奖)",
    ],
    "prize_fee": [
        r"\b(felicidades|parabens|selamat|felicitations|gluckwunsch|congratulazioni).{0,50}\b(ganaste|ganhou|menang|gagne|gewonnen|vinto)\b",
        r"\b(tarjeta regalo|cartao presente|kartu hadiah|carte cadeau|geschenkkarte|buono regalo)\b",
        r"\b(tasa de envio|taxa de envio|biaya pengiriman|frais de livraison|versandkosten|spese di spedizione)\b",
        r"(تهانينا|ربحت|رسوم الشحن)",
        r"(बधाई|जीते|शिपिंग शुल्क)",
        r"(おめでとう|当選|送料)",
        r"(축하합니다|당첨|배송비)",
        r"(поздравляем|выиграли|доставка)",
        r"(恭喜|中奖|运费)",
    ],
    "urgency_bait": [
        r"\b(urgente|ahora mismo|ultima oportunidad|ultimos? \d+ minutos|agora mesmo|segera|terbatas)\b",
        r"\b(urgent|maintenant|sofort|letzte chance|subito|offerta limitata|offerta limitata|ultima chiamata|angebot endet)\b",
        r"(عاجل|الان|الآن|خلال|عرض محدود)",
        r"(तुरंत|अभी|अंतिम मौका|सीमित समय)",
        r"(今すぐ|今だけ|限定|最後のチャンス|まもなく終了)",
        r"(지금|긴급|마감|한정)",
        r"(срочно|сейчас|последний шанс|ограниченное предложение)",
        r"(立即|马上|限时|最后机会)",
    ],
    "fomo_trigger": [
        r"\b(preventa|pre venda|se agota|se agotara|listing binance|va a subir|vai explodir|akan naik|akan meledak)\b",
        r"\b(va exploser|sera epuise|wird explodieren|vorverkauf|esplodera|prevendita)\b",
        r"(اكتتاب مسبق|قبل الادراج|ستنطلق|لا تفوت)",
        r"(प्रीसेल|लिस्टिंग से पहले|मत चूको|100x)",
        r"(爆上げ|上場前|プレセール|見逃すな)",
        r"(상장 전|프리세일|놓치지 마세요|폭등)",
        r"(пресейл|до листинга|взлетит|не упустите)",
        r"(预售|上市前|暴涨|不要错过)",
    ],
    "loss_aversion_trigger": [
        r"\b(perdera?s? acceso|perdera?s? sus fondos|perder seus fundos|perder acesso|kehilangan akses|kehilangan dana)\b",
        r"\b(perdre acces|perdre vos fonds|zugriff verlieren|gelder verlieren|perdere accesso|perdere fondi)\b",
        r"\b(perdera el acceso|perdera acceso|sonst wird ihr konto geschlossen|konto geschlossen)\b",
        r"(ستفقد|ستفقد الوصول|ستفقد اموالك)",
        r"(पहुंच खो देंगे|धन खो देंगे|खाता बंद)",
        r"(アクセスを失う|資金を失う|アカウントが停止)",
        r"(접근 권한.*잃|자금.*잃|계정.*정지)",
        r"(потеряете доступ|потеряете средства)",
        r"(失去访问权限|资金.*丢失|账号.*停用)",
    ],
    "social_proof_trigger": [
        r"\b(miles de usuarios|millones de personas|todos se estan uniendo|milhares de pessoas|ribuan pengguna)\b",
        r"\b(des milliers|tout le monde rejoint|tausende nutzer|alle treten bei|migliaia di utenti)\b",
        r"(آلاف المستخدمين|الجميع ينضم)",
        r"(हजारों लोग|सब जुड़ रहे हैं)",
        r"(何千人|みんな参加)",
        r"(수천 명|모두 참여)",
        r"(тысячи пользователей|все присоединяются)",
        r"(数千用户|大家都在加入)",
    ],
    "authority_impersonation": [
        r"\b(soporte oficial|equipo de soporte|suporte oficial|equipe de suporte|dukungan resmi|tim dukungan)\b",
        r"\b(support officiel|equipe de securite|offizieller support|sicherheitsteam|supporto ufficiale|team sicurezza)\b",
        r"(الدعم الرسمي|فريق الدعم|فريق الامن|البنك)",
        r"(आधिकारिक समर्थन|सहायता टीम|सुरक्षा टीम|बैंक)",
        r"(公式サポート|セキュリティチーム|銀行)",
        r"(공식 지원|보안팀|은행)",
        r"(официальная поддержка|служба безопасности|банк)",
        r"(官方支持|安全团队|银行)",
    ],
    "wallet_verification": [
        r"\b(wallet|billetera|cartera|carteira|dompet|portefeuille|brieftasche|portafoglio).{0,50}\b(verificacion|verificacao|verifikasi|verification|verifizierung|verifica)\b",
        r"\b(verificar|verifique|verifier|verifizieren|verifica).{0,50}\b(wallet|billetera|cartera|carteira|dompet|portefeuille|brieftasche|portafoglio)\b",
        r"(محفظة.*تحقق|تحقق.*محفظة)",
        r"(वॉलेट.*सत्यापन|वॉलेट.*वेरिफ)",
        r"(ウォレット認証|ウォレット.*確認|ウォレットを接続)",
        r"(지갑 인증|지갑.*확인)",
        r"(проверка кошелька|кошелек.*провер)",
        r"(钱包验证|钱包.*验证|连接钱包|授权交易)",
    ],
    "market_manipulation": [
        r"\b(100x|x100|x5|gem alert|moonshot|airdrop|binance listing)\b",
        r"\b(to the moon|va a la luna|vai para a lua|akan moon|vers la lune|zum mond|alla luna|coin prima|coin.*influencer)\b",
        r"(الى القمر|ايردروب|بينانس)",
        r"(चांद तक|एयरड्रॉप|बिनांस)",
        r"(月まで|エアドロップ|バイナンス|無料トークン)",
        r"(투더문|에어드랍|바이낸스)",
        r"(туземун|эйрдроп|бинанс)",
        r"(冲月|空投|币安|免费代币)",
    ],
    "credential_phishing": [
        r"\b(cuenta bancaria|kata sandi|verifikasi keamanan|passwort|konto geschlossen|donnees bancaires)\b",
        r"\b(informazioni bancarie|dati della carta|carte bancaire|datos de tarjeta|dados do cartao)\b",
        r"(بيانات البطاقة|كلمة المرور|ادخل بيانات|أدخل بيانات)",
        r"(पासवर्ड|कार्ड विवरण|लॉगिन विवरण)",
        r"(パスワード|カード情報|ログイン情報)",
        r"(비밀번호|카드 정보|로그인 정보)",
        r"(пароль|данные карты|данные входа)",
        r"(密码|银行卡信息|登录信息)",
    ],
}

for _group_name, _patterns in MULTILINGUAL_SCAM_PATTERN_GROUPS.items():
    SCAM_PATTERN_GROUPS.setdefault(_group_name, []).extend(_patterns)
del _group_name, _patterns

PSYCHOLOGICAL_TRIGGER_GROUPS = [
    "fomo_trigger",
    "urgency_bait",
    "loss_aversion_trigger",
    "social_proof_trigger",
    "authority_impersonation",
]

STRONG_PRESENTATION_REASONS = {
    "FINANCIAL_OR_CRYPTO_BAIT",
    "WALLET_VERIFICATION_BAIT",
    "TRADING_BOT_BAIT",
    "MARKET_MANIPULATION_BAIT",
    "PHISHING_URGENCY_THREAT",
    "FOMO_TRIGGER",
    "LOSS_AVERSION_TRIGGER",
    "SOCIAL_PROOF_TRIGGER",
    "AUTHORITY_IMPERSONATION",
    "CREDENTIAL_OR_PAYMENT_DATA_PHISHING",
    "COORDINATED_AMPLIFICATION_LANGUAGE",
}

LIVE_INFERENCE_BENCHMARK_CASES = [
    {
        "category": "Crypto/FOMO scam",
        "expected_behavior": "Review/High",
        "text": "Presale ends in the next 15 minutes. This crypto will go 100x after Binance listing. Buy now!",
    },
    {
        "category": "Wallet verification phishing",
        "expected_behavior": "Review/High",
        "text": "Official Meta support requires wallet verification in 24 hours to avoid losing access. Click to verify.",
    },
    {
        "category": "Trading bot income scam",
        "expected_behavior": "Review/High",
        "text": "Just made $10,000 in one week using this trading bot. Ask me how in DM. 100% legit. linktr.ee/scamlink",
    },
    {
        "category": "Prize/shipping fee scam",
        "expected_behavior": "Review/High",
        "text": "Congratulations, you won a $1,000 gift card. Please pay the shipping fee now to claim your prize.",
    },
    {
        "category": "Authority impersonation",
        "expected_behavior": "Review/High",
        "text": "Platform support team detected suspicious activity. Fill out the official appeal form before your account is suspended.",
    },
    {
        "category": "Social proof manipulation",
        "expected_behavior": "Review/High",
        "text": "Trusted by 50,000 users. Everyone is joining our VIP Telegram signal group. Last chance to enter.",
    },
    {
        "category": "Political amplification",
        "expected_behavior": "Review/High",
        "text": "Everyone needs to see this video right now. Vote them out and push this tag until it trends.",
    },
    {
        "category": "Organic civic announcement",
        "expected_behavior": "Low",
        "text": "The city council published meeting notes and invited residents to comment before Friday.",
    },
    {
        "category": "Organic casual reply",
        "expected_behavior": "Low",
        "text": "Thanks for sharing the photos from the event. It was nice to see everyone there.",
    },
    {
        "category": "Organic news summary",
        "expected_behavior": "Low",
        "text": "The report summarizes quarterly public transport ridership and planned maintenance work.",
    },
]

GENERIC_SIGNATURE_TERMS = {
    "yes",
    "lol",
    "nice",
    "good",
    "love",
    "thank",
    "thanks",
    "birthday",
    "happy",
    "happy birthday",
    "sorry",
    "i'm sorry",
    "i'm",
    "awesome",
}

MEDIA_SIGNATURE_TERMS = {
    "jpeg",
    "jpg",
    "png",
    "gif",
    "preview",
    "redd",
    "pjpg",
    "image",
    "photo",
}

WEAK_SIGNATURE_TERMS = GENERIC_SIGNATURE_TERMS | MEDIA_SIGNATURE_TERMS


def missing_runtime_dependencies(include_plotting: bool = True) -> List[str]:
    packages = ["duckdb"]
    if include_plotting:
        packages.append("matplotlib")
    return [pkg for pkg in packages if importlib.util.find_spec(pkg) is None]


def require_runtime_dependencies(include_plotting: bool = True) -> None:
    missing = missing_runtime_dependencies(include_plotting=include_plotting)
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing dependency/dependencies: {joined}. "
            "Run: python3 -m pip install -r requirements.txt"
        )


def _clip01(values: Any) -> Any:
    return np.clip(values, 0.0, 1.0)


def _clean_scalar(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def normalize_column_name(value: Any) -> str:
    normalized = fold_text_for_match(value)
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return re.sub(r"_+", "_", normalized).strip("_")


def fold_text_for_match(value: Any) -> str:
    text = _clean_scalar(value).casefold().translate(TURKISH_CHAR_MAP)
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


def unicode_text_for_match(value: Any) -> str:
    text = _clean_scalar(value).casefold().translate(TURKISH_CHAR_MAP)
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def _pattern_count(text: str, patterns: List[str], extra_text: Optional[str] = None) -> int:
    haystacks = [text]
    if extra_text and extra_text != text:
        haystacks.append(extra_text)
    return sum(
        1
        for pattern in patterns
        if any(re.search(pattern, haystack, flags=re.IGNORECASE) for haystack in haystacks if haystack)
    )


def normalize_text(text: Any) -> str:
    value = fold_text_for_match(text)
    value = re.sub(URL_PATTERN, " ", value, flags=re.IGNORECASE)
    value = re.sub(r"[@#]\w+", " ", value)
    value = re.sub(r"[^\w']+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def extract_domain(value: Any) -> str:
    raw = _clean_scalar(value).lower()
    if not raw:
        return "unknown"
    candidate = raw if "://" in raw else f"https://{raw}"
    parsed = urlparse(candidate)
    domain = parsed.netloc or parsed.path.split("/")[0]
    domain = domain.replace("www.", "").strip(".")
    return domain or "unknown"


def keyword_signature(value: Any, max_terms: int = 8) -> str:
    raw = fold_text_for_match(value)
    if not raw:
        return ""
    terms: List[str] = []
    for part in re.split(r"[,;|/\n\t]+", raw):
        cleaned = re.sub(r"[^\w$#@' ]+", " ", part, flags=re.UNICODE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned) >= 3 and cleaned not in STOPWORDS:
            terms.append(cleaned)
    if not terms:
        return ""
    counter = Counter(terms)
    top_terms = [term for term, _ in counter.most_common(max_terms)]
    return "|".join(sorted(set(top_terms)))


def signature_terms(signature: Any) -> List[str]:
    raw = _clean_scalar(signature).lower()
    if not raw:
        return []
    payload = raw.split(":", 1)[-1]
    return [term.strip() for term in payload.split("|") if term.strip()]


def is_weak_narrative_signature(signature: Any) -> bool:
    terms = signature_terms(signature)
    if not terms:
        return True
    return all(term in WEAK_SIGNATURE_TERMS for term in terms)


def text_signature(text: Any, max_terms: int = 10) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""
    tokens = [
        token
        for token in WORD_RE.findall(normalized)
        if len(token) >= 3 and token not in STOPWORDS
    ]
    if not tokens:
        return ""
    counter = Counter(tokens)
    top_tokens = [token for token, _ in counter.most_common(max_terms)]
    return "|".join(sorted(set(top_tokens)))


def narrative_signature(text: Any, keywords: Any = None) -> str:
    kw_sig = keyword_signature(keywords)
    if kw_sig:
        return f"kw:{kw_sig}"
    txt_sig = text_signature(text)
    return f"tx:{txt_sig}" if txt_sig else ""


def _token_stats(text: str) -> Tuple[int, int, float, float, int]:
    tokens = [token.lower() for token in TOKEN_RE.findall(fold_text_for_match(text))]
    word_count = len(tokens)
    if word_count == 0:
        return 0, 0, 0.0, 0.0, 0
    counts = Counter(tokens)
    unique_count = len(counts)
    repetition_ratio = 1.0 - (unique_count / max(word_count, 1))
    max_token_share = max(counts.values()) / max(word_count, 1)
    cta_count = sum(count for token, count in counts.items() if token.strip("#@$") in CALL_TO_ACTION_TERMS)
    return word_count, unique_count, repetition_ratio, max_token_share, cta_count


def compute_text_features(df: pd.DataFrame) -> pd.DataFrame:
    texts = df.get(TEXT_COL, pd.Series("", index=df.index)).fillna("").astype(str)
    stripped_texts = texts.str.strip()
    folded_texts = texts.apply(fold_text_for_match)
    unicode_texts = texts.apply(unicode_text_for_match)
    normalized_texts = texts.apply(normalize_text)
    stats = [_token_stats(text) for text in texts.tolist()]
    features = pd.DataFrame(
        stats,
        columns=[
            "word_count",
            "unique_word_count",
            "repetition_ratio",
            "max_token_share",
            "call_to_action_count",
        ],
        index=df.index,
    )

    char_len = texts.str.len()
    stripped_char_len = stripped_texts.str.len()
    letter_count = texts.str.count(r"[A-Za-z]")
    uppercase_count = texts.str.count(r"[A-Z]")
    digit_count = texts.str.count(r"[0-9]")
    punctuation_count = texts.str.count(r"[!?]")

    features["char_len"] = char_len.astype(float)
    features["stripped_char_len"] = stripped_char_len.astype(float)
    features["empty_text"] = (stripped_char_len == 0).astype(float)
    features["short_text"] = ((stripped_char_len > 0) & (stripped_char_len <= 3)).astype(float)
    features["url_count"] = texts.str.count(URL_PATTERN, flags=re.IGNORECASE).astype(float)
    features["hashtag_count"] = texts.str.count(r"#\w+").astype(float)
    features["mention_count"] = texts.str.count(r"@\w+").astype(float)
    features["url_only_or_link_only"] = (
        (stripped_char_len > 0)
        & ((features["url_count"] + features["hashtag_count"] + features["mention_count"]) > 0)
        & (normalized_texts.str.len() == 0)
    ).astype(float)
    features["punctuation_count"] = punctuation_count.astype(float)
    features["uppercase_ratio"] = np.divide(
        uppercase_count,
        np.maximum(letter_count, 1),
        out=np.zeros(len(df), dtype=float),
        where=np.maximum(letter_count, 1) != 0,
    )
    features["digit_ratio"] = np.divide(
        digit_count,
        np.maximum(char_len, 1),
        out=np.zeros(len(df), dtype=float),
        where=np.maximum(char_len, 1) != 0,
    )
    features["link_signal_density"] = (
        (2.0 * features["url_count"]) + features["hashtag_count"] + features["mention_count"]
    ) / np.maximum(features["word_count"], 1)
    features["punctuation_density"] = features["punctuation_count"] / np.maximum(char_len, 1)
    features["repeated_char_run"] = texts.apply(
        lambda value: 1.0 if re.search(r"([A-Za-z0-9])\1{4,}", value or "") else 0.0
    )
    for group_name, patterns in SCAM_PATTERN_GROUPS.items():
        features[f"{group_name}_count"] = [
            float(_pattern_count(folded, patterns, unicode_text))
            for folded, unicode_text in zip(folded_texts.tolist(), unicode_texts.tolist())
        ]
    scam_group_cols = [f"{group_name}_count" for group_name in SCAM_PATTERN_GROUPS]
    features["scam_signal_count"] = features[scam_group_cols].sum(axis=1)
    features["scam_signal_groups"] = features[scam_group_cols].gt(0).sum(axis=1).astype(float)

    sentiment = pd.to_numeric(df.get(SENTIMENT_COL, pd.Series(0.0, index=df.index)), errors="coerce")
    features["sentiment_abs"] = sentiment.fillna(0.0).abs().clip(0.0, 1.0)
    return features


def score_content_features(features: pd.DataFrame) -> pd.Series:
    char_len = features["stripped_char_len"]
    word_count = features["word_count"]

    empty = features["empty_text"] >= 1.0
    short_or_tiny = np.where(char_len < 24, _clip01((24.0 - char_len) / 24.0), 0.0)
    very_long = _clip01((char_len - 700.0) / 1300.0)
    repetition_raw = _clip01(np.maximum(features["repetition_ratio"], features["max_token_share"] - 0.15))
    repetition = np.where(word_count >= 3, repetition_raw, 0.0)
    link_density = _clip01(features["link_signal_density"] * 5.0)
    caps = _clip01((features["uppercase_ratio"] - 0.35) / 0.45)
    punctuation = _clip01(features["punctuation_density"] * 18.0)
    sentiment_extreme = np.power(features["sentiment_abs"], 1.4)
    cta = _clip01(features["call_to_action_count"] / np.maximum(word_count, 1) * 7.0)
    scam_signal = _clip01(features["scam_signal_count"] / 4.0)
    scam_groups = features["scam_signal_groups"]

    risk = (
        0.20 * short_or_tiny
        + 0.10 * very_long
        + 0.20 * repetition
        + 0.17 * link_density
        + 0.10 * caps
        + 0.08 * punctuation
        + 0.10 * sentiment_extreme
        + 0.05 * cta
        + 0.20 * scam_signal
    )
    stacked_spam = (link_density >= 0.70) & (punctuation >= 0.50) & (cta >= 0.70)
    risk = np.where(stacked_spam, np.maximum(risk, 0.82), risk)
    high_risk_lure = (
        (features["phishing_threat_count"] > 0)
        | (features["prize_fee_count"] >= 2)
        | ((features["financial_bait_count"] > 0) & ((features["engagement_bait_count"] > 0) | (cta >= 0.35)))
        | (
            (features["financial_bait_count"] >= 2)
            & (
                (features["urgency_bait_count"] > 0)
                | (features["fomo_trigger_count"] > 0)
                | (features["market_manipulation_count"] > 0)
                | (cta >= 0.20)
            )
        )
        | ((features["engagement_bait_count"] > 0) & (cta >= 0.50))
        | ((features["adult_or_leak_bait_count"] > 0) & ((features["engagement_bait_count"] > 0) | (features["urgency_bait_count"] > 0)))
        | ((features["health_miracle_count"] > 0) & (cta >= 0.20))
        | ((features["debt_relief_count"] > 0) & (cta >= 0.10))
        | (features["credential_phishing_count"] > 0)
        | (features["wallet_verification_count"] > 0)
        | (features["trading_bot_bait_count"] > 0)
        | ((features["loss_aversion_trigger_count"] > 0) & (features["urgency_bait_count"] > 0))
        | (
            (features["authority_impersonation_count"] > 0)
            & (
                (features["urgency_bait_count"] > 0)
                | (features["loss_aversion_trigger_count"] > 0)
                | (features["phishing_threat_count"] > 0)
                | (features["wallet_verification_count"] > 0)
                | (cta >= 0.20)
            )
        )
        | (
            (features["fomo_trigger_count"] > 0)
            & (
                (features["financial_bait_count"] > 0)
                | (features["market_manipulation_count"] > 0)
                | (features["engagement_bait_count"] > 0)
            )
            & ((features["urgency_bait_count"] > 0) | (cta >= 0.20))
        )
        | (
            (features["market_manipulation_count"] > 0)
            & (
                (features["financial_bait_count"] > 0)
                | (features["engagement_bait_count"] > 0)
                | (features["urgency_bait_count"] > 0)
            )
        )
    )
    political_high = (
        ((features["political_mobilization_count"] >= 2) & ((features["urgency_bait_count"] > 0) | (cta >= 0.20)))
        | ((features["political_mobilization_count"] > 0) & (cta >= 0.35))
    )
    review_lure = (
        (scam_groups >= 2)
        | ((features["engagement_bait_count"] > 0) & (cta >= 0.25))
        | (features["political_mobilization_count"] > 0)
        | ((features["market_manipulation_count"] > 0) & (features["urgency_bait_count"] > 0))
        | (features["credential_phishing_count"] > 0)
        | (features["fomo_trigger_count"] > 0)
        | (features["loss_aversion_trigger_count"] > 0)
        | (features["social_proof_trigger_count"] > 0)
        | (features["authority_impersonation_count"] > 0)
    )
    risk = np.where(high_risk_lure, np.maximum(risk, 0.72), risk)
    risk = np.where(political_high, np.maximum(risk, 0.66), risk)
    risk = np.where(review_lure, np.maximum(risk, 0.50), risk)
    risk = np.where(empty, np.maximum(risk, 0.92), risk)
    risk = np.where(features["repeated_char_run"] > 0, np.maximum(risk, 0.55), risk)
    return pd.Series(_clip01(risk), index=features.index, name="content_risk")


def content_reason_codes(row: pd.Series) -> List[str]:
    reasons: List[str] = []
    stripped_len = row.get("stripped_char_len", row.get("char_len", 0))
    if stripped_len == 0:
        reasons.append("EMPTY_TEXT")
    elif stripped_len <= 3:
        reasons.append("SHORT_TEXT_LOW_EVIDENCE")
    elif stripped_len < 24:
        reasons.append("VERY_SHORT_TEXT")
    if stripped_len > 700:
        reasons.append("VERY_LONG_TEXT")
    if row["word_count"] >= 3 and (row["repetition_ratio"] >= 0.45 or row["max_token_share"] >= 0.35):
        reasons.append("HIGH_TOKEN_REPETITION")
    if row["link_signal_density"] >= 0.22:
        reasons.append("LINK_HASHTAG_MENTION_DENSE")
    if row.get("url_only_or_link_only", 0) > 0:
        reasons.append("URL_ONLY_OR_LINK_ONLY")
    if row["uppercase_ratio"] >= 0.55 and row["word_count"] >= 5:
        reasons.append("HIGH_UPPERCASE_RATIO")
    if row["punctuation_density"] >= 0.04:
        reasons.append("PUNCTUATION_BURST")
    if row["sentiment_abs"] >= 0.85:
        reasons.append("EXTREME_SENTIMENT")
    if row["call_to_action_count"] >= 2:
        reasons.append("CALL_TO_ACTION_LANGUAGE")
    if row.get("financial_bait_count", 0) > 0:
        reasons.append("FINANCIAL_OR_CRYPTO_BAIT")
    if row.get("phishing_threat_count", 0) > 0:
        reasons.append("PHISHING_URGENCY_THREAT")
    if row.get("engagement_bait_count", 0) > 0:
        reasons.append("ENGAGEMENT_OR_LINK_BAIT")
    if row.get("adult_or_leak_bait_count", 0) > 0:
        reasons.append("ADULT_OR_LEAK_BAIT")
    if row.get("health_miracle_count", 0) > 0:
        reasons.append("HEALTH_MIRACLE_CLAIM")
    if row.get("prize_fee_count", 0) > 0:
        reasons.append("PRIZE_OR_FEE_BAIT")
    if row.get("debt_relief_count", 0) > 0:
        reasons.append("DEBT_RELIEF_BAIT")
    if row.get("urgency_bait_count", 0) > 0:
        reasons.append("URGENCY_LANGUAGE")
    if row.get("fomo_trigger_count", 0) > 0:
        reasons.append("FOMO_TRIGGER")
    if row.get("loss_aversion_trigger_count", 0) > 0:
        reasons.append("LOSS_AVERSION_TRIGGER")
    if row.get("social_proof_trigger_count", 0) > 0:
        reasons.append("SOCIAL_PROOF_TRIGGER")
    if row.get("authority_impersonation_count", 0) > 0:
        reasons.append("AUTHORITY_IMPERSONATION")
    if row.get("credential_phishing_count", 0) > 0:
        reasons.append("CREDENTIAL_OR_PAYMENT_DATA_PHISHING")
    if row.get("wallet_verification_count", 0) > 0:
        reasons.append("WALLET_VERIFICATION_BAIT")
    if row.get("trading_bot_bait_count", 0) > 0:
        reasons.append("TRADING_BOT_BAIT")
    if row.get("market_manipulation_count", 0) > 0:
        reasons.append("MARKET_MANIPULATION_BAIT")
    if row.get("political_mobilization_count", 0) > 0:
        reasons.append("COORDINATED_AMPLIFICATION_LANGUAGE")
    if row["repeated_char_run"] > 0:
        reasons.append("REPEATED_CHARACTER_RUN")
    if not reasons:
        reasons.append("LOW_CONTENT_RISK")
    return reasons


def prepare_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    for col in REQUIRED_COLUMNS:
        if col not in prepared.columns:
            prepared[col] = np.nan
    prepared[TEXT_COL] = prepared[TEXT_COL].fillna("").astype(str)
    prepared[KEYWORDS_COL] = prepared[KEYWORDS_COL].fillna("").astype(str)
    prepared[AUTHOR_COL] = prepared[AUTHOR_COL].fillna("").astype(str)
    prepared["language"] = prepared["language"].fillna("unknown").astype(str).str.lower()
    prepared["primary_theme"] = prepared["primary_theme"].fillna("unknown").astype(str)
    prepared["platform_domain"] = prepared[URL_COL].apply(extract_domain)
    prepared["date_ts"] = pd.to_datetime(prepared[DATE_COL], errors="coerce", utc=True)
    prepared["text_signature"] = prepared[TEXT_COL].apply(text_signature)
    prepared["keyword_signature"] = prepared[KEYWORDS_COL].apply(keyword_signature)
    prepared["narrative_signature"] = [
        narrative_signature(text, keywords)
        for text, keywords in zip(prepared[TEXT_COL], prepared[KEYWORDS_COL])
    ]
    return prepared


def build_author_stats(
    df: pd.DataFrame,
    full_author_stats: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    valid = df[df[AUTHOR_COL].fillna("").astype(str).str.len() > 0].copy()
    if valid.empty:
        return pd.DataFrame(columns=[AUTHOR_COL, "author_risk"])

    grouped = valid.groupby(AUTHOR_COL, dropna=False)
    stats = grouped.agg(
        author_post_count_sample=(AUTHOR_COL, "size"),
        author_url_nunique=("platform_domain", "nunique"),
        author_theme_nunique=("primary_theme", "nunique"),
        author_language_nunique=("language", "nunique"),
        author_text_signature_nunique=("text_signature", "nunique"),
        author_keyword_signature_nunique=("keyword_signature", "nunique"),
    ).reset_index()

    sorted_valid = valid.sort_values([AUTHOR_COL, "date_ts"]).copy()
    sorted_valid["gap_seconds"] = (
        sorted_valid.groupby(AUTHOR_COL)["date_ts"].diff().dt.total_seconds()
    )
    gap_stats = sorted_valid.groupby(AUTHOR_COL).agg(
        author_median_gap_seconds=("gap_seconds", "median"),
        author_burst_pairs=("gap_seconds", lambda values: int((values.dropna() <= 300).sum())),
    ).reset_index()
    stats = stats.merge(gap_stats, on=AUTHOR_COL, how="left")

    if full_author_stats is not None and not full_author_stats.empty:
        stats = stats.merge(full_author_stats, on=AUTHOR_COL, how="left")
    else:
        stats["author_post_count_full"] = stats["author_post_count_sample"]
        stats["author_url_nunique_full"] = stats["author_url_nunique"]
        stats["author_theme_nunique_full"] = stats["author_theme_nunique"]
        stats["author_language_nunique_full"] = stats["author_language_nunique"]

    for col in [
        "author_post_count_full",
        "author_url_nunique_full",
        "author_theme_nunique_full",
        "author_language_nunique_full",
    ]:
        stats[col] = stats[col].fillna(stats[col.replace("_full", "")] if col.replace("_full", "") in stats else 0)

    counts = stats["author_post_count_full"].astype(float)
    p75 = max(float(np.nanpercentile(counts, 75)), 1.0)
    p95 = max(float(np.nanpercentile(counts, 95)), p75 + 1.0)
    volume_risk = _clip01((counts - p75) / (p95 - p75))

    sample_counts = stats["author_post_count_sample"].clip(lower=1).astype(float)
    burst_risk = stats["author_burst_pairs"].fillna(0.0) / np.maximum(sample_counts - 1.0, 1.0)
    text_repeat_risk = 1.0 - (
        stats["author_text_signature_nunique"].astype(float) / np.maximum(sample_counts, 1.0)
    )
    keyword_repeat_risk = 1.0 - (
        stats["author_keyword_signature_nunique"].astype(float) / np.maximum(sample_counts, 1.0)
    )
    low_theme_diversity = 1.0 - (
        stats["author_theme_nunique"].astype(float) / np.maximum(sample_counts, 1.0)
    )

    stats["author_volume_risk"] = _clip01(volume_risk)
    stats["author_burst_risk"] = _clip01(burst_risk)
    stats["author_repeat_risk"] = _clip01(np.maximum(text_repeat_risk, keyword_repeat_risk))
    stats["author_low_theme_diversity_risk"] = _clip01(low_theme_diversity)
    stats["author_risk"] = _clip01(
        0.35 * stats["author_volume_risk"]
        + 0.25 * stats["author_burst_risk"]
        + 0.25 * stats["author_repeat_risk"]
        + 0.15 * stats["author_low_theme_diversity_risk"]
    )
    stats.loc[stats["author_post_count_sample"] < 3, "author_risk"] *= 0.6
    return stats


def author_reason_codes(row: pd.Series) -> List[str]:
    reasons: List[str] = []
    if row.get("author_volume_risk", 0.0) >= 0.6:
        reasons.append("HIGH_AUTHOR_VOLUME")
    if row.get("author_burst_risk", 0.0) >= 0.4:
        reasons.append("BURST_POSTING_WITHIN_5_MIN")
    if row.get("author_repeat_risk", 0.0) >= 0.5:
        reasons.append("AUTHOR_REPEATS_TEXT_OR_KEYWORDS")
    if row.get("author_low_theme_diversity_risk", 0.0) >= 0.7:
        reasons.append("LOW_AUTHOR_TOPIC_DIVERSITY")
    if not reasons and row.get("author_risk", 0.0) >= 0.65:
        reasons.append("HIGH_AUTHOR_RISK")
    return reasons


def build_coordination_stats(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["narrative_signature"].fillna("").astype(str).str.len() > 0].copy()
    valid = valid[~valid["narrative_signature"].apply(is_weak_narrative_signature)].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "narrative_signature",
                "coordination_risk",
                "cluster_size",
                "cluster_author_nunique",
                "cluster_platform_nunique",
                "cluster_language_nunique",
                "cluster_window_hours",
                "signature_term_count",
                "cluster_content_similarity",
                "cluster_confidence_score",
            ]
        )
    valid["signature_term_count"] = valid["narrative_signature"].apply(
        lambda value: len(signature_terms(value))
    )
    valid["known_author_hash"] = valid[AUTHOR_COL].where(
        valid[AUTHOR_COL].fillna("").astype(str).str.strip().ne(""),
        np.nan,
    )

    grouped = valid.groupby("narrative_signature", dropna=False)
    stats = grouped.agg(
        cluster_size=("narrative_signature", "size"),
        cluster_author_nunique=("known_author_hash", "nunique"),
        cluster_platform_nunique=("platform_domain", "nunique"),
        cluster_language_nunique=("language", "nunique"),
        cluster_theme_nunique=("primary_theme", "nunique"),
        cluster_first_seen=("date_ts", "min"),
        cluster_last_seen=("date_ts", "max"),
        cluster_example_text=(TEXT_COL, lambda values: _clean_scalar(values.iloc[0])[:220]),
        signature_term_count=("signature_term_count", "max"),
    ).reset_index()

    hours = (
        stats["cluster_last_seen"] - stats["cluster_first_seen"]
    ).dt.total_seconds().fillna(0.0) / 3600.0
    stats["cluster_window_hours"] = hours

    sizes = stats["cluster_size"].astype(float)
    p95 = max(float(np.nanpercentile(sizes, 95)), 4.0)
    size_risk = _clip01((np.log1p(sizes) - math.log1p(2.0)) / (math.log1p(p95) - math.log1p(2.0)))
    author_spread = _clip01((stats["cluster_author_nunique"].astype(float) - 1.0) / 4.0)
    platform_spread = _clip01((stats["cluster_platform_nunique"].astype(float) - 1.0) / 3.0)
    compact_window = _clip01(24.0 / (stats["cluster_window_hours"] + 1.0))

    coordinated_mask = (
        (stats["cluster_size"] >= 4)
        & (stats["cluster_author_nunique"] >= 3)
        & (stats["signature_term_count"] >= 2)
        & (stats["cluster_window_hours"] <= 72)
    )
    stats["coord_size_risk"] = np.where(coordinated_mask, size_risk, 0.0)
    stats["coord_author_spread_risk"] = np.where(coordinated_mask, author_spread, 0.0)
    stats["coord_platform_spread_risk"] = np.where(coordinated_mask, platform_spread, 0.0)
    stats["coord_time_compactness_risk"] = np.where(coordinated_mask, compact_window, 0.0)
    stats["coordination_risk"] = _clip01(
        0.40 * stats["coord_size_risk"]
        + 0.35 * stats["coord_author_spread_risk"]
        + 0.15 * stats["coord_platform_spread_risk"]
        + 0.10 * stats["coord_time_compactness_risk"]
    )
    signature_richness = _clip01((stats["signature_term_count"].astype(float) - 1.0) / 5.0)
    stats["cluster_content_similarity"] = np.where(coordinated_mask, 1.0, 0.0)
    stats["cluster_confidence_score"] = _clip01(
        0.35 * stats["coord_author_spread_risk"]
        + 0.25 * stats["coord_time_compactness_risk"]
        + 0.20 * stats["coord_size_risk"]
        + 0.10 * signature_richness
        + 0.10 * stats["cluster_content_similarity"]
    )
    return stats[stats["coordination_risk"] > 0].reset_index(drop=True)


def coordination_reason_codes(row: pd.Series) -> List[str]:
    reasons: List[str] = []
    if row.get("coordination_risk", 0.0) >= 0.35:
        reasons.append("COORDINATED_NARRATIVE_CLUSTER")
    if row.get("cluster_author_nunique", 0) >= 3:
        reasons.append("MULTI_AUTHOR_SAME_NARRATIVE")
    if row.get("cluster_platform_nunique", 0) >= 2:
        reasons.append("CROSS_PLATFORM_SPREAD")
    if row.get("cluster_window_hours", 999999) <= 72 and row.get("cluster_size", 0) >= 4:
        reasons.append("SHORT_TIME_WINDOW_CLUSTER")
    return reasons


def combine_component_scores(df: pd.DataFrame) -> pd.Series:
    content_weight = 0.35
    author_weight = 0.30
    coordination_weight = 0.35

    content = df["content_risk"].fillna(0.0)
    if "author_post_count_sample" in df.columns:
        author_sample_count = pd.to_numeric(df["author_post_count_sample"], errors="coerce").fillna(0)
    else:
        author_sample_count = pd.Series(0, index=df.index)
    if "author_post_count_full" in df.columns:
        author_full_count = pd.to_numeric(df["author_post_count_full"], errors="coerce").fillna(0)
    else:
        author_full_count = author_sample_count
    author_available = (
        (df[AUTHOR_COL].fillna("").astype(str).str.len() > 0)
        & ((author_sample_count >= 3) | (author_full_count >= 5))
    )
    coord_available = df["coordination_risk"].fillna(0.0) > 0

    numerator = content_weight * content
    denominator = pd.Series(content_weight, index=df.index, dtype=float)

    numerator += np.where(author_available, author_weight * df["author_risk"].fillna(0.0), 0.0)
    denominator += np.where(author_available, author_weight, 0.0)

    numerator += np.where(
        coord_available,
        coordination_weight * df["coordination_risk"].fillna(0.0),
        0.0,
    )
    denominator += np.where(coord_available, coordination_weight, 0.0)
    combined = pd.Series(_clip01(numerator / denominator), index=df.index, name="risk_score")
    coord = df["coordination_risk"].fillna(0.0)
    author = df["author_risk"].fillna(0.0)
    combined = np.where(coord >= 0.80, np.maximum(combined, 0.68), combined)
    combined = np.where((coord >= 0.65) & (content >= 0.35), np.maximum(combined, 0.66), combined)
    combined = np.where((author >= 0.80) & ((content >= 0.35) | (coord >= 0.35)), np.maximum(combined, 0.66), combined)
    return pd.Series(_clip01(combined), index=df.index, name="risk_score")


def risk_band_from_score(scores: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [scores >= RISK_THRESHOLD, scores >= REVIEW_THRESHOLD],
            ["High", "Review"],
            default="Low",
        ),
        index=scores.index,
        name="risk_band",
    )


def make_reason_codes(row: pd.Series) -> str:
    reasons: List[str] = []
    reasons.extend(row.get("content_reason_codes", []))
    reasons.extend(author_reason_codes(row))
    reasons.extend(coordination_reason_codes(row))
    if row.get("risk_score", 0.0) < 0.35 and "LOW_CONTENT_RISK" not in reasons:
        reasons.append("LOW_OVERALL_RISK")
    unique_reasons = list(dict.fromkeys(reasons))
    return ";".join(unique_reasons[:8])


def score_dataframe(
    df: pd.DataFrame,
    full_author_stats: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    prepared = prepare_input_dataframe(df)
    text_features = compute_text_features(prepared)
    scored = pd.concat([prepared.reset_index(drop=True), text_features.reset_index(drop=True)], axis=1)
    scored["content_risk"] = score_content_features(text_features).reset_index(drop=True)
    scored["content_reason_codes"] = text_features.apply(content_reason_codes, axis=1).reset_index(drop=True)

    author_stats = build_author_stats(scored, full_author_stats=full_author_stats)
    coordination_stats = build_coordination_stats(scored)

    if not author_stats.empty:
        scored = scored.merge(author_stats, on=AUTHOR_COL, how="left")
    else:
        scored["author_risk"] = 0.0
    if not coordination_stats.empty:
        scored = scored.merge(coordination_stats, on="narrative_signature", how="left")
    else:
        scored["coordination_risk"] = 0.0

    for col in [
        "author_risk",
        "author_volume_risk",
        "author_burst_risk",
        "author_repeat_risk",
        "author_low_theme_diversity_risk",
        "author_post_count_sample",
        "author_post_count_full",
        "coordination_risk",
    ]:
        if col not in scored:
            scored[col] = 0.0
        scored[col] = scored[col].fillna(0.0)
    for col, default in [
        ("cluster_size", 0),
        ("cluster_author_nunique", 0),
        ("cluster_platform_nunique", 0),
        ("cluster_language_nunique", 0),
        ("cluster_window_hours", np.nan),
        ("signature_term_count", 0),
        ("cluster_confidence_score", 0.0),
        ("cluster_content_similarity", 0.0),
    ]:
        if col not in scored:
            scored[col] = default

    scored["risk_score"] = combine_component_scores(scored)
    scored["organic_score"] = 1.0 - scored["risk_score"]
    scored["label"] = np.where(scored["risk_score"] >= RISK_THRESHOLD, "Manipulative", "Organic")
    scored["risk_band"] = risk_band_from_score(scored["risk_score"])
    scored["reason_codes"] = scored.apply(make_reason_codes, axis=1)

    context = {
        "author_stats": author_stats,
        "coordination_stats": coordination_stats,
    }
    return scored, context


def _duckdb_connection() -> Any:
    require_runtime_dependencies(include_plotting=False)
    import duckdb

    return duckdb.connect(database=":memory:")


def inspect_dataset(data_path: Path = DATA_PATH) -> Dict[str, Any]:
    con = _duckdb_connection()
    data_path = Path(data_path)
    try:
        row_count = con.execute("SELECT count(*) FROM read_parquet(?)", [str(data_path)]).fetchone()[0]
        schema = con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(data_path)]).fetchdf()
        date_range = con.execute(
            """
            SELECT min(try_cast(date AS TIMESTAMP)) AS min_date,
                   max(try_cast(date AS TIMESTAMP)) AS max_date
            FROM read_parquet(?)
            """,
            [str(data_path)],
        ).fetchdf()
    finally:
        con.close()
    return {
        "data_path": str(data_path),
        "row_count": int(row_count),
        "schema": schema,
        "date_range": date_range,
    }


def load_sample(
    data_path: Path = DATA_PATH,
    sample_size: int = 200_000,
    seed: int = 42,
) -> pd.DataFrame:
    con = _duckdb_connection()
    data_path = Path(data_path)
    try:
        try:
            con.execute("SELECT setseed(?)", [((seed % 10_000) / 10_000.0)])
        except Exception:
            pass
        total_rows = con.execute("SELECT count(*) FROM read_parquet(?)", [str(data_path)]).fetchone()[0]
        fraction = min(1.0, max((sample_size * 1.25) / max(total_rows, 1), sample_size / max(total_rows, 1)))
        try:
            sample = con.execute(
                """
                SELECT *
                FROM read_parquet(?)
                WHERE random() < ?
                LIMIT ?
                """,
                [str(data_path), fraction, sample_size],
            ).fetchdf()
        except Exception:
            sample = con.execute(
                """
                SELECT *
                FROM read_parquet(?)
                LIMIT ?
                """,
                [str(data_path), sample_size],
            ).fetchdf()
        if len(sample) > sample_size:
            sample = sample.sample(n=sample_size, random_state=seed)
        sample = sample.reset_index(drop=True)
    finally:
        con.close()
    return sample


def build_full_author_stats(data_path: Path = DATA_PATH) -> pd.DataFrame:
    con = _duckdb_connection()
    data_path = Path(data_path)
    try:
        stats = con.execute(
            """
            SELECT
                author_hash,
                count(*) AS author_post_count_full,
                count(DISTINCT url) AS author_url_nunique_full,
                count(DISTINCT primary_theme) AS author_theme_nunique_full,
                count(DISTINCT language) AS author_language_nunique_full
            FROM read_parquet(?)
            WHERE author_hash IS NOT NULL AND author_hash <> ''
            GROUP BY author_hash
            """,
            [str(data_path)],
        ).fetchdf()
    finally:
        con.close()
    return stats


def selected_export_columns(scored: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "risk_score",
        "presentation_priority",
        "risk_band",
        "organic_score",
        "label",
        "reason_codes",
        "content_risk",
        "author_risk",
        "coordination_risk",
        "language",
        "platform_domain",
        "primary_theme",
        AUTHOR_COL,
        DATE_COL,
        "narrative_signature",
        "cluster_size",
        "cluster_author_nunique",
        "cluster_platform_nunique",
        "cluster_window_hours",
        "signature_term_count",
        "cluster_confidence_score",
        "cluster_content_similarity",
        "stripped_char_len",
        TEXT_COL,
        KEYWORDS_COL,
    ]
    existing = [col for col in columns if col in scored.columns]
    export = scored[existing].copy()
    if TEXT_COL in export.columns:
        export["original_text_preview"] = export[TEXT_COL].fillna("").astype(str).str.slice(0, 350)
        export = export.drop(columns=[TEXT_COL])
    return export.sort_values("risk_score", ascending=False)


def split_reason_codes(value: Any) -> List[str]:
    if not isinstance(value, str):
        return []
    return [reason.strip() for reason in value.split(";") if reason.strip()]


def has_any_reason(value: Any, reason_set: Iterable[str]) -> bool:
    reasons = set(split_reason_codes(value))
    return bool(reasons.intersection(set(reason_set)))


def text_has_meaningful_signature(value: Any) -> bool:
    return bool(text_signature(value))


def stable_hash(value: str, modulo: int = NLP_HASH_DIM) -> int:
    return zlib.crc32(value.encode("utf-8")) % modulo


def nlp_text_feature_ids(text: Any, hash_dim: int = NLP_HASH_DIM) -> np.ndarray:
    normalized = normalize_text(text)
    if not normalized:
        return np.array([], dtype=np.int64)
    tokens = [
        token
        for token in WORD_RE.findall(normalized)
        if len(token) >= 2 and token not in STOPWORDS
    ]
    feature_ids: set[int] = set()
    for token in tokens[:120]:
        feature_ids.add(stable_hash(f"w:{token}", hash_dim))
    for left, right in zip(tokens[:119], tokens[1:120]):
        feature_ids.add(stable_hash(f"wb:{left}_{right}", hash_dim))

    char_source = " ".join(tokens)[:600]
    for ngram_size in (3, 4, 5):
        if len(char_source) < ngram_size:
            continue
        for idx in range(0, len(char_source) - ngram_size + 1):
            piece = char_source[idx : idx + ngram_size]
            if piece.strip():
                feature_ids.add(stable_hash(f"c{ngram_size}:{piece}", hash_dim))
    if not feature_ids:
        return np.array([], dtype=np.int64)
    return np.fromiter(feature_ids, dtype=np.int64)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def train_nlp_text_model(scored: pd.DataFrame, artifacts_dir: Path) -> Optional[Dict[str, Any]]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    text_len = scored["stripped_char_len"].fillna(0)
    reason_values = scored["reason_codes"].fillna("")
    strong_reason = reason_values.apply(lambda value: has_any_reason(value, STRONG_PRESENTATION_REASONS))
    not_empty = ~reason_values.str.contains("EMPTY_TEXT", regex=False)

    positive = scored[
        (scored["risk_score"].fillna(0.0) >= RISK_THRESHOLD)
        & (text_len >= 35)
        & strong_reason
        & not_empty
        & scored[TEXT_COL].apply(text_has_meaningful_signature)
    ].copy()
    negative = scored[
        (scored["risk_score"].fillna(1.0) <= 0.20)
        & (text_len >= 35)
        & reason_values.str.contains("LOW_CONTENT_RISK", regex=False)
        & (~strong_reason)
        & scored[TEXT_COL].apply(text_has_meaningful_signature)
    ].copy()

    usable_docs_per_class = min(len(positive), len(negative), NLP_MAX_DOCS_PER_CLASS)
    metadata = {
        "model_version": NLP_TEXT_MODEL_VERSION,
        "hash_dim": NLP_HASH_DIM,
        "positive_candidates": int(len(positive)),
        "negative_candidates": int(len(negative)),
        "train_docs_per_class": int(usable_docs_per_class),
        "trained": bool(usable_docs_per_class >= NLP_MIN_DOCS_PER_CLASS),
        "positive_rule": "risk_score >= 0.65, non-empty, meaningful text, strong reason",
        "negative_rule": "risk_score <= 0.20, LOW_CONTENT_RISK, non-empty, meaningful text",
    }

    if usable_docs_per_class < NLP_MIN_DOCS_PER_CLASS:
        (artifacts_dir / "nlp_text_model_metadata.json").write_text(
            json.dumps(_json_safe(metadata), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        pd.DataFrame([metadata]).to_csv(artifacts_dir / "nlp_pseudo_label_training_summary.csv", index=False)
        return None

    positive = positive.sample(n=usable_docs_per_class, random_state=NLP_RANDOM_SEED)
    negative = negative.sample(n=usable_docs_per_class, random_state=NLP_RANDOM_SEED)

    pos_df = np.zeros(NLP_HASH_DIM, dtype=np.float32)
    neg_df = np.zeros(NLP_HASH_DIM, dtype=np.float32)
    pos_doc_features = 0
    neg_doc_features = 0
    for text in positive[TEXT_COL].fillna("").astype(str):
        ids = nlp_text_feature_ids(text)
        if len(ids):
            pos_df[ids] += 1.0
            pos_doc_features += int(len(ids))
    for text in negative[TEXT_COL].fillna("").astype(str):
        ids = nlp_text_feature_ids(text)
        if len(ids):
            neg_df[ids] += 1.0
            neg_doc_features += int(len(ids))

    alpha = 0.5
    pos_rate = (pos_df + alpha) / (usable_docs_per_class + 2.0 * alpha)
    neg_rate = (neg_df + alpha) / (usable_docs_per_class + 2.0 * alpha)
    feature_log_odds = np.clip(np.log(pos_rate) - np.log(neg_rate), -5.0, 5.0).astype(np.float32)

    metadata.update(
        {
            "positive_doc_features": int(pos_doc_features),
            "negative_doc_features": int(neg_doc_features),
            "alpha": float(alpha),
            "scoring": "sigmoid(mean(feature_log_odds)) with rule-evidence gate",
        }
    )
    np.savez_compressed(
        artifacts_dir / "nlp_text_model.npz",
        feature_log_odds=feature_log_odds,
        hash_dim=np.array([NLP_HASH_DIM], dtype=np.int64),
        scale=np.array([1.0], dtype=np.float32),
    )
    (artifacts_dir / "nlp_text_model_metadata.json").write_text(
        json.dumps(_json_safe(metadata), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame([metadata]).to_csv(artifacts_dir / "nlp_pseudo_label_training_summary.csv", index=False)
    return {
        "feature_log_odds": feature_log_odds,
        "hash_dim": NLP_HASH_DIM,
        "scale": 1.0,
        "metadata": metadata,
    }


def load_nlp_text_model(artifacts_dir: Path = ARTIFACT_DIR) -> Optional[Dict[str, Any]]:
    model_path = artifacts_dir / "nlp_text_model.npz"
    if not model_path.exists():
        return None
    data = np.load(model_path)
    metadata_path = artifacts_dir / "nlp_text_model_metadata.json"
    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    hash_dim = int(data["hash_dim"][0]) if "hash_dim" in data else NLP_HASH_DIM
    scale = float(data["scale"][0]) if "scale" in data else 2.2
    return {
        "feature_log_odds": data["feature_log_odds"].astype(np.float32),
        "hash_dim": hash_dim,
        "scale": scale,
        "metadata": metadata,
    }


def predict_nlp_text_risk(text: Any, model: Optional[Dict[str, Any]]) -> Optional[float]:
    if not model or "feature_log_odds" not in model:
        return None
    weights = model["feature_log_odds"]
    hash_dim = int(model.get("hash_dim", len(weights)))
    ids = nlp_text_feature_ids(text, hash_dim=hash_dim)
    if len(ids) == 0:
        return None
    evidence = weights[ids]
    if len(evidence) == 0:
        return None
    raw = float(model.get("scale", 1.0)) * float(np.mean(evidence))
    return float(_clip01(_sigmoid(raw)))


def _zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    std = float(numeric.std(ddof=0))
    if std <= 1e-12:
        return pd.Series(0.0, index=values.index)
    return (numeric - float(numeric.mean())) / std


def build_temporal_burst_windows(time_spikes: pd.DataFrame) -> pd.DataFrame:
    if time_spikes.empty:
        return pd.DataFrame(
            columns=[
                "hour",
                "posts",
                "suspicious_posts",
                "high_posts",
                "avg_risk",
                "suspicious_share",
                "high_share",
                "suspicious_share_zscore",
                "high_share_zscore",
                "burst_score",
                "burst_label",
            ]
        )
    bursts = time_spikes.copy()
    if "high_posts" not in bursts.columns:
        bursts["high_posts"] = 0
    if "suspicious_share" not in bursts.columns:
        bursts["suspicious_share"] = bursts["suspicious_posts"] / bursts["posts"].clip(lower=1)
    bursts["high_share"] = bursts["high_posts"] / bursts["posts"].clip(lower=1)
    bursts["suspicious_share_zscore"] = _zscore(bursts["suspicious_share"])
    bursts["high_share_zscore"] = _zscore(bursts["high_share"])
    bursts["burst_score"] = np.maximum(
        bursts["suspicious_share_zscore"].fillna(0.0),
        bursts["high_share_zscore"].fillna(0.0),
    )
    bursts["burst_label"] = np.where(
        (bursts["burst_score"] >= 2.0) & (bursts["posts"] >= 100),
        "Campaign Burst",
        "Normal",
    )
    return bursts.sort_values("hour").reset_index(drop=True)


def _text_preview(value: Any, limit: int = 360) -> str:
    text = re.sub(r"\s+", " ", _clean_scalar(value))
    if len(text) <= limit:
        return text
    return f"{text[: max(limit - 3, 0)]}..."


def _format_float(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(numeric) or math.isinf(numeric):
        return "n/a"
    return f"{numeric:.{digits}f}"


def _format_case_section(
    title: str,
    row: Optional[pd.Series],
    why: str,
    jury_sentence: str,
    limitation: str,
    extra_fields: Optional[List[Tuple[str, Any]]] = None,
) -> List[str]:
    lines = [f"## {title}", ""]
    if row is None:
        lines.extend(["Uygun gerçek örnek bulunamadı.", ""])
        return lines
    fields = [
        ("Risk score", _format_float(row.get("risk_score"))),
        ("Risk band", row.get("risk_band", "n/a")),
        ("Label", row.get("label", "n/a")),
        ("Reason codes", row.get("reason_codes", "n/a")),
        ("Language", row.get("language", "n/a")),
        ("Platform", row.get("platform_domain", "n/a")),
        ("Theme", row.get("primary_theme", "n/a")),
        ("Author", _text_preview(row.get(AUTHOR_COL, ""), 18) or "n/a"),
        ("Date", row.get(DATE_COL, "n/a")),
        ("Narrative signature", _text_preview(row.get("narrative_signature", ""), 120) or "n/a"),
    ]
    if extra_fields:
        fields.extend(extra_fields)
    lines.append("| Alan | Değer |")
    lines.append("|---|---|")
    for key, value in fields:
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "**Metin preview**",
            "",
            f"> {_text_preview(row.get(TEXT_COL, row.get('original_text_preview', '')), 420)}",
            "",
            f"**Neden yakalandı?** {why}",
            "",
            f"**Jüriye anlatım cümlesi:** {jury_sentence}",
            "",
            f"**Sınırlama notu:** {limitation}",
            "",
        ]
    )
    return lines


def build_case_studies(scored: pd.DataFrame, artifacts_dir: Path) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    usable = scored[
        scored["stripped_char_len"].fillna(0).ge(35)
        & scored["stripped_char_len"].fillna(0).le(500)
        & (~scored["reason_codes"].fillna("").str.contains("EMPTY_TEXT", regex=False))
        & (~scored["reason_codes"].fillna("").str.contains("VERY_LONG_TEXT", regex=False))
        & scored[TEXT_COL].apply(text_has_meaningful_signature)
    ].copy()

    financial_reasons = {
        "FINANCIAL_OR_CRYPTO_BAIT",
        "TRADING_BOT_BAIT",
        "MARKET_MANIPULATION_BAIT",
    }
    financial_candidates = usable[
        usable["reason_codes"].apply(lambda value: has_any_reason(value, financial_reasons))
    ].sort_values(["risk_score", "content_risk"], ascending=False)
    financial_row = financial_candidates.iloc[0] if not financial_candidates.empty else None

    cluster_cols = [
        "narrative_signature",
        "coordination_risk",
        "cluster_confidence_score",
        "cluster_size",
        "cluster_author_nunique",
        "cluster_platform_nunique",
        "cluster_language_nunique",
        "cluster_window_hours",
        "signature_term_count",
        "cluster_example_text",
    ]
    existing_cluster_cols = [col for col in cluster_cols if col in scored.columns]
    clusters = scored[existing_cluster_cols].drop_duplicates() if existing_cluster_cols else pd.DataFrame()
    if not clusters.empty:
        clusters = clusters[
            clusters["cluster_confidence_score"].fillna(0.0).ge(0.80)
            & clusters["cluster_size"].fillna(0).ge(4)
            & clusters["cluster_author_nunique"].fillna(0).ge(3)
            & clusters["cluster_window_hours"].fillna(float("inf")).le(72)
            & (~clusters["narrative_signature"].fillna("").apply(is_weak_narrative_signature))
        ].sort_values(["cluster_confidence_score", "coordination_risk", "cluster_size"], ascending=False)
    cluster_row = None
    if not clusters.empty:
        best_cluster = clusters.iloc[0]
        cluster_matches = scored[scored["narrative_signature"] == best_cluster["narrative_signature"]].copy()
        cluster_matches["case_priority"] = np.maximum(
            cluster_matches["risk_score"].fillna(0.0),
            cluster_matches["coordination_risk"].fillna(0.0),
        )
        cluster_row = cluster_matches.sort_values("case_priority", ascending=False).iloc[0]

    primary_psychological_reasons = {
        "FOMO_TRIGGER",
        "LOSS_AVERSION_TRIGGER",
        "SOCIAL_PROOF_TRIGGER",
        "AUTHORITY_IMPERSONATION",
    }
    psychological_candidates = usable[
        usable["reason_codes"].apply(lambda value: has_any_reason(value, primary_psychological_reasons))
    ].copy()
    if not psychological_candidates.empty:
        psychological_candidates["psychological_case_priority"] = psychological_candidates["reason_codes"].apply(
            lambda value: (
                4
                if "FOMO_TRIGGER" in split_reason_codes(value)
                else 3
                if "LOSS_AVERSION_TRIGGER" in split_reason_codes(value)
                else 2
                if "AUTHORITY_IMPERSONATION" in split_reason_codes(value)
                else 1
            )
        )
    if financial_row is not None and not psychological_candidates.empty:
        psychological_candidates = psychological_candidates[
            psychological_candidates["narrative_signature"] != financial_row.get("narrative_signature")
        ]
    psychological_candidates = psychological_candidates.sort_values(
        ["psychological_case_priority", "risk_score", "content_risk"],
        ascending=False,
    )
    psychological_row = psychological_candidates.iloc[0] if not psychological_candidates.empty else None

    lines = [
        "# Case Studies: Savunulabilir Risk Örnekleri",
        "",
        "Bu dosya, jüri sunumunda somut örnek anlatımı için üretilmiştir. Örnekler gerçek skorlanmış satırlardan seçilir; sentetik benchmark yalnızca canlı inference hazırlığını göstermek için ayrı artifact olarak tutulur.",
        "",
    ]
    lines.extend(
        _format_case_section(
            "Case Study 1: Finansal/Kripto veya Trading Scam",
            financial_row,
            "Finansal vaat, kripto/trading dili, CTA veya link yönlendirmesi gibi güçlü reason code'lar aynı metinde birikiyor.",
            "Bu örnek, sistemin yalnızca kelime eşleşmesine değil; finansal bait, CTA ve risk bandı birleşimine bakarak manipülatif ticari vaadi yakaladığını gösterir.",
            "Bu karar etiketli doğruluk iddiası değildir; açıklanabilir risk sinyallerinin yoğunlaştığı bir örnektir.",
        )
    )
    cluster_extra = []
    if cluster_row is not None:
        cluster_extra = [
            ("Cluster size", _format_float(cluster_row.get("cluster_size"), 0)),
            ("Cluster author count", _format_float(cluster_row.get("cluster_author_nunique"), 0)),
            ("Cluster platform count", _format_float(cluster_row.get("cluster_platform_nunique"), 0)),
            ("Cluster language count", _format_float(cluster_row.get("cluster_language_nunique"), 0)),
            ("Cluster window hours", _format_float(cluster_row.get("cluster_window_hours"), 2)),
            ("Cluster confidence", _format_float(cluster_row.get("cluster_confidence_score"), 3)),
            ("Coordination risk", _format_float(cluster_row.get("coordination_risk"), 3)),
        ]
    lines.extend(
        _format_case_section(
            "Case Study 2: Kısa Zaman Penceresinde Koordineli Anlatı",
            cluster_row,
            "Aynı narrative signature kısa zaman penceresinde birden fazla author tarafından paylaşılıyor; cluster boyutu, author yayılımı ve zaman kompaktlığı confidence skorunu destekliyor.",
            "Bu örnek, sistemin tekil metinden öteye geçip aynı anlatının kısa sürede çoklu hesaplarla yayıldığı kampanya benzeri davranışı görünür kıldığını gösterir.",
            "Sistem çeviri semantiği iddia etmez; aynı signature içinde çok dilli yayılım varsa `cluster_language_nunique` ile bunu görünür kılar.",
            extra_fields=cluster_extra,
        )
    )
    lines.extend(
        _format_case_section(
            "Case Study 3: Psikolojik Manipülasyon Tetikleyicisi",
            psychological_row,
            "FOMO, aciliyet, kayıptan kaçınma, sosyal kanıt veya otorite taklidi gibi davranışsal manipülasyon sinyalleri reason code olarak yakalanıyor.",
            "Bu örnek, sistemin spam dilini yalnızca teknik sinyallerle değil, psikolojik baskı mekanizmalarıyla da açıklayabildiğini gösterir.",
            "Psikolojik trigger'lar destekleyici kanıttır; final karar risk bandı ve diğer content/author/coordination sinyalleriyle birlikte değerlendirilir.",
        )
    )

    benchmark_path = artifacts_dir / "live_inference_benchmark.csv"
    if benchmark_path.exists():
        lines.extend(
            [
                "## Canlı Inference Benchmark Notu",
                "",
                "`live_inference_benchmark.csv` sentetik ama gerçekçi senaryolarda canlı tahmin hazırlığını gösterir. Bu case study dosyasındaki örnekler ise mümkün olduğunca gerçek skorlanmış satırlardan seçilir.",
                "",
            ]
        )

    path = artifacts_dir / "case_studies.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_tables(scored: pd.DataFrame, artifacts_dir: Path) -> Dict[str, Any]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    high_risk = scored[scored["risk_band"] == "High"].copy()
    review_or_high = scored[scored["risk_band"].isin(["Review", "High"])].copy()
    suspicious = review_or_high

    selected_export_columns(scored).to_csv(artifacts_dir / "scored_sample.csv", index=False)
    presentation_examples = scored[
        (scored["stripped_char_len"].fillna(0) >= 35)
        & (~scored["narrative_signature"].apply(is_weak_narrative_signature))
        & (~scored["reason_codes"].fillna("").str.contains("EMPTY_TEXT", regex=False))
        & (scored["reason_codes"].apply(lambda value: has_any_reason(value, STRONG_PRESENTATION_REASONS)))
        & (scored[TEXT_COL].apply(text_has_meaningful_signature))
    ].copy()
    presentation_examples["presentation_priority"] = np.maximum(
        presentation_examples["risk_score"].fillna(0.0),
        presentation_examples["coordination_risk"].fillna(0.0),
    )
    coord_window_ok = (
        presentation_examples["coordination_risk"].fillna(0.0).lt(0.55)
        | presentation_examples["cluster_window_hours"].fillna(float("inf")).le(72)
    )
    presentation_examples = (
        presentation_examples[(presentation_examples["presentation_priority"] >= 0.55) & coord_window_ok]
        .sort_values("presentation_priority", ascending=False)
        .drop_duplicates(subset=["narrative_signature", "platform_domain", "language"])
    )
    selected_export_columns(presentation_examples).head(10).to_csv(
        artifacts_dir / "presentation_examples.csv",
        index=False,
    )

    language_platform = (
        suspicious.groupby(["language", "platform_domain"], dropna=False)
        .agg(
            suspicious_posts=("risk_score", "size"),
            high_posts=("risk_band", lambda values: int((values == "High").sum())),
            review_posts=("risk_band", lambda values: int((values == "Review").sum())),
            avg_risk=("risk_score", "mean"),
            max_risk=("risk_score", "max"),
        )
        .reset_index()
        .sort_values(["suspicious_posts", "avg_risk"], ascending=False)
    )
    language_platform.to_csv(artifacts_dir / "risk_map_language_platform.csv", index=False)

    language_manipulative = (
        scored.groupby("language", dropna=False)
        .agg(
            total_posts=("risk_score", "size"),
            manipulative_posts=("risk_band", lambda values: int((values == "High").sum())),
            review_posts=("risk_band", lambda values: int((values == "Review").sum())),
            review_high_posts=("risk_band", lambda values: int((values != "Low").sum())),
            avg_risk=("risk_score", "mean"),
        )
        .reset_index()
    )
    language_manipulative["manipulative_rate"] = (
        language_manipulative["manipulative_posts"] / language_manipulative["total_posts"].clip(lower=1)
    )
    language_manipulative["review_high_rate"] = (
        language_manipulative["review_high_posts"] / language_manipulative["total_posts"].clip(lower=1)
    )
    language_manipulative = language_manipulative.sort_values(
        ["manipulative_posts", "manipulative_rate"], ascending=False
    )
    language_manipulative.to_csv(artifacts_dir / "language_manipulative_share.csv", index=False)

    top_segments = (
        suspicious.groupby(["language", "platform_domain", "primary_theme"], dropna=False)
        .agg(
            suspicious_posts=("risk_score", "size"),
            high_posts=("risk_band", lambda values: int((values == "High").sum())),
            review_posts=("risk_band", lambda values: int((values == "Review").sum())),
            avg_risk=("risk_score", "mean"),
            unique_authors=(AUTHOR_COL, "nunique"),
            narratives=("narrative_signature", "nunique"),
        )
        .reset_index()
        .sort_values(["suspicious_posts", "avg_risk"], ascending=False)
    )
    top_segments.to_csv(artifacts_dir / "top_risk_segments.csv", index=False)

    cluster_cols = [
        "narrative_signature",
        "coordination_risk",
        "cluster_confidence_score",
        "cluster_content_similarity",
        "cluster_size",
        "cluster_author_nunique",
        "cluster_platform_nunique",
        "cluster_language_nunique",
        "cluster_window_hours",
        "signature_term_count",
        "cluster_example_text",
    ]
    cluster_cols = [col for col in cluster_cols if col in scored.columns]
    cluster_frame = scored[cluster_cols].drop_duplicates()
    for col, default in [
        ("coordination_risk", 0.0),
        ("cluster_confidence_score", 0.0),
        ("cluster_content_similarity", 0.0),
        ("cluster_size", 0),
        ("cluster_window_hours", float("inf")),
        ("narrative_signature", ""),
    ]:
        if col not in cluster_frame.columns:
            cluster_frame[col] = default
    clusters = (
        cluster_frame.loc[
            lambda frame: frame["coordination_risk"].fillna(0.0).gt(0.0)
            & frame["cluster_window_hours"].fillna(float("inf")).le(72)
            & (~frame["narrative_signature"].apply(is_weak_narrative_signature))
        ]
        .sort_values(["cluster_confidence_score", "coordination_risk", "cluster_size"], ascending=False)
        .head(100)
    )
    clusters.to_csv(artifacts_dir / "top_coordination_clusters.csv", index=False)

    author_cols = [
        AUTHOR_COL,
        "author_risk",
        "author_volume_risk",
        "author_burst_risk",
        "author_repeat_risk",
        "author_low_theme_diversity_risk",
        "author_post_count_sample",
        "author_post_count_full",
        "author_url_nunique_full",
        "author_theme_nunique_full",
        "author_language_nunique_full",
    ]
    author_cols = [col for col in author_cols if col in scored.columns]
    author_source = scored[
        scored[AUTHOR_COL].fillna("").astype(str).str.strip().ne("")
    ].copy()
    if not author_source.empty and author_cols:
        author_summary = (
            author_source.groupby(AUTHOR_COL, dropna=False)
            .agg(
                sample_posts=("risk_score", "size"),
                high_posts=("risk_band", lambda values: int((values == "High").sum())),
                review_posts=("risk_band", lambda values: int((values == "Review").sum())),
                avg_risk=("risk_score", "mean"),
                max_risk=("risk_score", "max"),
                author_risk=("author_risk", "max"),
                author_volume_risk=("author_volume_risk", "max"),
                author_burst_risk=("author_burst_risk", "max"),
                author_repeat_risk=("author_repeat_risk", "max"),
                author_low_theme_diversity_risk=("author_low_theme_diversity_risk", "max"),
                author_post_count_full=("author_post_count_full", "max"),
                platform_nunique=("platform_domain", "nunique"),
                theme_nunique=("primary_theme", "nunique"),
                language_nunique=("language", "nunique"),
                example_text=(TEXT_COL, lambda values: _clean_scalar(values.iloc[0])[:220]),
            )
            .reset_index()
        )
        author_summary["review_high_posts"] = author_summary["high_posts"] + author_summary["review_posts"]
        author_summary["review_high_share"] = author_summary["review_high_posts"] / author_summary["sample_posts"].clip(lower=1)
        author_summary = author_summary.sort_values(
            ["author_risk", "high_posts", "review_high_posts", "avg_risk"],
            ascending=False,
        ).head(100)
    else:
        author_summary = pd.DataFrame(
            columns=[
                AUTHOR_COL,
                "sample_posts",
                "high_posts",
                "review_posts",
                "review_high_share",
                "avg_risk",
                "max_risk",
                "author_risk",
            ]
        )
    author_summary.to_csv(artifacts_dir / "top_risk_authors.csv", index=False)

    global_review_high_share = len(suspicious) / max(len(scored), 1)
    global_high_share = len(high_risk) / max(len(scored), 1)
    platform_normalized = (
        scored.groupby("platform_domain", dropna=False)
        .agg(
            posts=("risk_score", "size"),
            high_posts=("risk_band", lambda values: int((values == "High").sum())),
            review_posts=("risk_band", lambda values: int((values == "Review").sum())),
            avg_risk=("risk_score", "mean"),
            max_risk=("risk_score", "max"),
        )
        .reset_index()
    )
    platform_normalized["review_high_posts"] = platform_normalized["high_posts"] + platform_normalized["review_posts"]
    platform_normalized["review_high_rate"] = platform_normalized["review_high_posts"] / platform_normalized["posts"].clip(lower=1)
    platform_normalized["high_rate"] = platform_normalized["high_posts"] / platform_normalized["posts"].clip(lower=1)
    platform_normalized["platform_normalized_risk_index"] = (
        platform_normalized["review_high_rate"] / max(global_review_high_share, 1e-12)
    )
    platform_normalized["platform_high_risk_index"] = (
        platform_normalized["high_rate"] / max(global_high_share, 1e-12)
    )
    platform_posts_floor = max(50, int(len(scored) * 0.0005))
    platform_normalized["sufficient_sample"] = platform_normalized["posts"] >= platform_posts_floor
    platform_normalized = platform_normalized.sort_values(
        ["sufficient_sample", "platform_normalized_risk_index", "review_high_posts"],
        ascending=False,
    )
    platform_normalized.to_csv(artifacts_dir / "platform_normalized_risk.csv", index=False)

    with_dates = scored.dropna(subset=["date_ts"]).copy()
    if not with_dates.empty:
        with_dates["hour"] = with_dates["date_ts"].dt.floor("h")
        time_spikes = (
            with_dates.groupby("hour")
            .agg(
                posts=("risk_score", "size"),
                suspicious_posts=("risk_band", lambda values: int((values != "Low").sum())),
                high_posts=("risk_band", lambda values: int((values == "High").sum())),
                avg_risk=("risk_score", "mean"),
            )
            .reset_index()
        )
        time_spikes["suspicious_share"] = time_spikes["suspicious_posts"] / time_spikes["posts"].clip(lower=1)
    else:
        time_spikes = pd.DataFrame(columns=["hour", "posts", "suspicious_posts", "high_posts", "avg_risk", "suspicious_share"])
    time_spikes.to_csv(artifacts_dir / "time_spikes.csv", index=False)
    temporal_bursts = build_temporal_burst_windows(time_spikes)
    temporal_bursts.to_csv(artifacts_dir / "temporal_burst_windows.csv", index=False)
    burst_rows = temporal_bursts[temporal_bursts["burst_label"] == "Campaign Burst"] if not temporal_bursts.empty else pd.DataFrame()

    return {
        "suspicious_count": int(len(suspicious)),
        "high_risk_count": int(len(high_risk)),
        "review_count": int((scored["risk_band"] == "Review").sum()),
        "sample_count": int(len(scored)),
        "suspicious_share": float(len(suspicious) / max(len(scored), 1)),
        "high_risk_share": float(len(high_risk) / max(len(scored), 1)),
        "top_language_platform": language_platform.head(10).to_dict(orient="records"),
        "top_language_manipulative": language_manipulative.head(10).to_dict(orient="records"),
        "top_segments": top_segments.head(10).to_dict(orient="records"),
        "top_clusters": clusters.head(10).to_dict(orient="records"),
        "top_authors": author_summary.head(10).to_dict(orient="records"),
        "top_platform_normalized": platform_normalized.head(10).to_dict(orient="records"),
        "temporal_burst_count": int(len(burst_rows)),
        "top_temporal_bursts": burst_rows.sort_values("burst_score", ascending=False).head(10).to_dict(orient="records")
        if not burst_rows.empty
        else [],
    }


def _percentage(value: float) -> str:
    return f"{value:.1%}"


def build_live_inference_benchmark(
    context: Dict[str, pd.DataFrame],
    artifacts_dir: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for case in LIVE_INFERENCE_BENCHMARK_CASES:
        prediction = predict_live(case["text"], context=context)
        expected_behavior = case["expected_behavior"]
        risk_band = str(prediction.get("risk_band", "Low"))
        readiness_check = (
            (expected_behavior == "Low" and risk_band == "Low")
            or (expected_behavior == "Review/High" and risk_band in {"Review", "High"})
        )
        psychological_triggers = prediction.get("used_features", {}).get("psychological_triggers", {})
        active_triggers = [
            name
            for name, count in psychological_triggers.items()
            if isinstance(count, (int, float)) and count > 0
        ]
        rows.append(
            {
                "category": case["category"],
                "expected_behavior": expected_behavior,
                "readiness_check": bool(readiness_check),
                "risk_score": prediction.get("risk_score"),
                "risk_band": risk_band,
                "label": prediction.get("label"),
                "evidence_level": prediction.get("evidence_level"),
                "nlp_text_risk": prediction.get("nlp_text_risk"),
                "nlp_model_used": prediction.get("nlp_model_used"),
                "top_reasons": ";".join(prediction.get("top_reasons", [])),
                "psychological_triggers": ";".join(active_triggers),
                "text_preview": case["text"][:180],
            }
        )
    benchmark = pd.DataFrame(rows)
    benchmark.to_csv(artifacts_dir / "live_inference_benchmark.csv", index=False)
    return benchmark


def build_marketing_scorecard(
    scored: pd.DataFrame,
    benchmark: pd.DataFrame,
    artifacts_dir: Path,
) -> Dict[str, Any]:
    high_risk = scored[scored["risk_band"] == "High"].copy()
    suspicious = scored[scored["risk_band"].isin(["Review", "High"])].copy()
    strong_reason_mask = high_risk["reason_codes"].apply(
        lambda value: has_any_reason(value, STRONG_PRESENTATION_REASONS)
    ) if not high_risk.empty else pd.Series([], dtype=bool)
    high_non_empty_mask = high_risk["stripped_char_len"].fillna(0).gt(0) if not high_risk.empty else pd.Series([], dtype=bool)
    high_non_empty_strong = int((strong_reason_mask & high_non_empty_mask).sum()) if not high_risk.empty else 0

    global_review_high_share = len(suspicious) / max(len(scored), 1)
    global_high_share = len(high_risk) / max(len(scored), 1)
    platform_posts_floor = max(50, int(len(scored) * 0.0005))
    platform_norm = (
        scored.groupby("platform_domain", dropna=False)
        .agg(
            posts=("risk_score", "size"),
            high_posts=("risk_band", lambda values: int((values == "High").sum())),
            review_high_posts=("risk_band", lambda values: int((values != "Low").sum())),
        )
        .reset_index()
    )
    platform_norm["review_high_rate"] = platform_norm["review_high_posts"] / platform_norm["posts"].clip(lower=1)
    platform_norm["high_rate"] = platform_norm["high_posts"] / platform_norm["posts"].clip(lower=1)
    platform_norm["platform_normalized_risk_index"] = (
        platform_norm["review_high_rate"] / max(global_review_high_share, 1e-12)
    )
    platform_norm["platform_high_risk_index"] = platform_norm["high_rate"] / max(global_high_share, 1e-12)
    platform_norm = platform_norm[platform_norm["posts"] >= platform_posts_floor].sort_values(
        ["platform_normalized_risk_index", "review_high_posts"], ascending=False
    )
    top_platform = platform_norm.iloc[0].to_dict() if not platform_norm.empty else {}

    if not suspicious.empty:
        top_segment_row = (
            suspicious.groupby(["language", "platform_domain", "primary_theme"], dropna=False)
            .agg(
                suspicious_posts=("risk_score", "size"),
                high_posts=("risk_band", lambda values: int((values == "High").sum())),
                avg_risk=("risk_score", "mean"),
            )
            .reset_index()
            .sort_values(["suspicious_posts", "avg_risk"], ascending=False)
            .iloc[0]
            .to_dict()
        )
        top_segment = (
            f"{top_segment_row.get('language')} / {top_segment_row.get('platform_domain')} / "
            f"{top_segment_row.get('primary_theme')}"
        )
    else:
        top_segment_row = {}
        top_segment = "none"

    cluster_cols = [
        "narrative_signature",
        "coordination_risk",
        "cluster_confidence_score",
        "cluster_size",
        "cluster_author_nunique",
        "cluster_window_hours",
    ]
    existing_cluster_cols = [col for col in cluster_cols if col in scored.columns]
    clusters = scored[existing_cluster_cols].drop_duplicates() if existing_cluster_cols else pd.DataFrame()
    if not clusters.empty:
        high_conf_clusters = clusters[
            clusters["coordination_risk"].fillna(0.0).ge(0.65)
            & clusters["cluster_confidence_score"].fillna(0.0).ge(0.80)
            & clusters["cluster_window_hours"].fillna(float("inf")).le(72)
            & (~clusters["narrative_signature"].fillna("").apply(is_weak_narrative_signature))
        ]
    else:
        high_conf_clusters = pd.DataFrame()

    benchmark_passed = int(benchmark["readiness_check"].sum()) if "readiness_check" in benchmark else 0
    benchmark_total = int(len(benchmark))
    metrics = [
        {
            "metric": "Skorlanan örnek satır",
            "value": f"{len(scored):,}",
            "detail": "Mevcut artifact üretiminde skorlanan toplam satır sayısı.",
        },
        {
            "metric": "Review + High adet/oran",
            "value": f"{len(suspicious):,} ({_percentage(global_review_high_share)})",
            "detail": "Analist incelemesine ayrılan seçici, düşük olmayan risk havuzu.",
        },
        {
            "metric": "High adet/oran",
            "value": f"{len(high_risk):,} ({_percentage(global_high_share)})",
            "detail": "Canlı inference sırasında Manipulative etiketi üreten yüksek riskli havuz.",
        },
        {
            "metric": "High-risk boş olmayan oran",
            "value": _percentage(float(high_non_empty_mask.mean()) if len(high_non_empty_mask) else 0.0),
            "detail": "Boş metin anomalisi olmayan yüksek riskli satırların oranı.",
        },
        {
            "metric": "Güçlü reason destekli High oranı",
            "value": _percentage(float(strong_reason_mask.mean()) if len(strong_reason_mask) else 0.0),
            "detail": "Güçlü manipülasyon sinyalleriyle açıklanabilen yüksek riskli kararların oranı.",
        },
        {
            "metric": "Sunuma uygun güçlü High örnek",
            "value": f"{high_non_empty_strong:,}",
            "detail": "Kalite filtrelerinden sonra kalan, savunulabilir ve boş olmayan yüksek riskli örnek sayısı.",
        },
        {
            "metric": "En yüksek platform-normalize risk indeksi",
            "value": f"{top_platform.get('platform_domain', 'none')}: {top_platform.get('platform_normalized_risk_index', 0.0):.2f}x",
            "detail": "Yeterli örnek hacmine sahip platformlarda, veri seti ortalamasına göre göreli risk.",
        },
        {
            "metric": "En riskli segment",
            "value": top_segment,
            "detail": f"{int(top_segment_row.get('suspicious_posts', 0)):,} Review+High gönderi.",
        },
        {
            "metric": "Yüksek güvenli koordinasyon cluster",
            "value": f"{len(high_conf_clusters):,}",
            "detail": "Cluster confidence >= 0.80, coordination risk >= 0.65 ve zaman penceresi <= 72 saat.",
        },
        {
            "metric": "Canlı inference hazırlık kontrolü",
            "value": f"{benchmark_passed}/{benchmark_total}",
            "detail": "Sentetik senaryo kontrolleri; etiketli performans metriği değildir.",
        },
    ]

    scorecard = pd.DataFrame(metrics)
    scorecard.to_csv(artifacts_dir / "marketing_scorecard.csv", index=False)
    markdown_lines = [
        "# Pazarlama Kanıt Scorecard",
        "",
        "Bu bölüm gözetimsiz bir kanıt katmanıdır. Proxy metrikler, senaryo kontrolleri ve açıklama kapsamı kullanır; etiketli model başarısı iddiası taşımaz.",
        "",
        "| Metrik | Değer | Açıklama |",
        "|---|---:|---|",
    ]
    for row in metrics:
        markdown_lines.append(f"| {row['metric']} | {row['value']} | {row['detail']} |")
    markdown_lines.append("")
    markdown_lines.extend(
        [
            "## Üretilen Grafikler Ne Anlatıyor?",
            "",
            "| Grafik | Ne ifade ediyor? | Sunumda nasıl kullanılır? |",
            "|---|---|---|",
            "| `risk_score_histogram.png` | Tüm örneklerde risk skorunun dağılımını ve Review/High eşiklerini gösterir. | Modelin seçici davrandığını, her şeyi manipülatif saymadığını anlatmak için kullanılır. |",
            "| `risk_map_language_platform.png` | Dil x platform kırılımında Review + High yoğunluğunu 0-700 sabit renk skalasıyla gösterir. | Manipülasyon haritası ve sıcak bölgeleri göstermek için ana görseldir. |",
            "| `language_manipulative_share.png` | Dillere göre Manipulative satır sayısını ve her dil içindeki Manipulative oranını birlikte gösterir. | Büyük dillerdeki hacmi ve küçük/orta dillerdeki göreli risk yoğunluğunu aynı anda anlatır. |",
            "| `top_suspicious_segments.png` | En çok Review + High üreten dil/platform segmentlerini sıralar. | Hangi platform/dil alanlarında riskin yoğunlaştığını hızlı anlatır. |",
            "| `platform_normalized_risk.png` | Platform içi risk oranını veri seti ortalamasına göre normalize eder. | X gibi büyük platformların ham hacim avantajını dengeleyip adil kıyas sunar. |",
            "| `top_risk_authors.png` | Davranışsal sinyallere göre en riskli author_hash değerlerini gösterir. | Burst, tekrar ve hacim davranışını tekil aktör seviyesinde kanıtlar. |",
            "| `hourly_suspicious_share.png` | Saatlik Review + High oranını zaman üzerinde gösterir. | Kampanya/burst dönemlerini ve zamansal yoğunlaşmayı anlatır. |",
            "| `temporal_burst_windows.png` | Saatlik risk oranındaki z-score sıçramalarını ve Campaign Burst pencerelerini gösterir. | Gerçek dünyada kampanya dönemlerini erken uyarı olarak yakalayabildiğimizi anlatır. |",
            "| `risk_funnel.png` | Tüm satırlardan Review + High, High ve güçlü reason destekli High örneklere daralan huniyi gösterir. | Modelin seçici ve kanıt odaklı karar verdiğini pazarlamak için kullanılır. |",
            "| `reason_code_breakdown.png` | High-risk kararlarını açıklayan reason code dağılımını gösterir. | Kararların kara kutu olmadığını, açıklanabilir sinyallere dayandığını gösterir. |",
            "| `psychological_trigger_breakdown.png` | FOMO, aciliyet, kayıptan kaçınma, sosyal kanıt ve otorite taklidi sinyallerinin kapsamını gösterir. | Spam dilini psikolojik manipülasyon tetikleyicileriyle ilişkilendirmek için kullanılır. |",
            "| `live_inference_benchmark.png` | Sentetik canlı inference senaryolarının risk skorlarını eşiklerle birlikte gösterir. | Jürinin gizli metin testine hazır olduğumuzu göstermek için kullanılır. |",
            "| `coordination_confidence_bubble.png` | Coordination cluster'larının zaman penceresi, güven skoru, boyut ve coordination risk ilişkisini gösterir. | Koordineli davranış tespitinin sadece metin değil, zaman ve author yayılımıyla desteklendiğini anlatır. |",
            "| `evidence_quality_summary.png` | High-risk kararların ne kadarının boş olmayan, güçlü reason destekli, içerik/author/coordination destekli olduğunu gösterir. | High-risk karar kalitesini ve savunulabilirliği özetler. |",
            "| `feature_importance_proxy.png` | Etiket olmadığı için klasik feature importance yerine, skor formülüne göre High-risk satırlarda en çok katkı yapan engineered sinyalleri gösterir. | Hangi feature engineering kararlarının skoru taşıdığını dürüst ve açıklanabilir şekilde anlatır. |",
            "| `risk_component_contribution.png` | Low/Review/High bandlarında content, author, coordination ve rule-floor katkılarının ortalama dağılımını gösterir. | Metadata ve text sinyallerinin final skorda nasıl birleştiğini görselleştirir. |",
            "| `case_studies.md` | Gerçek skorlanmış satırlardan seçilen 3 savunulabilir hikayeyi özetler. | Jürinin 'somut örnek göster' sorusuna doğrudan cevap olarak kullanılır. |",
            "",
            "Önerilen konumlandırma: Etiketsiz sosyal medya verisi için açıklanabilir, seçici ve canlı inference'a hazır risk skorlama sistemi.",
        ]
    )
    (artifacts_dir / "marketing_scorecard.md").write_text("\n".join(markdown_lines), encoding="utf-8")

    return {
        "marketing_scorecard": metrics,
        "live_inference_readiness_passed": benchmark_passed,
        "live_inference_readiness_total": benchmark_total,
        "high_risk_non_empty_share": float(high_non_empty_mask.mean()) if len(high_non_empty_mask) else 0.0,
        "high_risk_strong_reason_share": float(strong_reason_mask.mean()) if len(strong_reason_mask) else 0.0,
        "high_confidence_coordination_cluster_count": int(len(high_conf_clusters)),
    }


def build_marketing_plots(
    scored: pd.DataFrame,
    benchmark: pd.DataFrame,
    artifacts_dir: Path,
) -> List[str]:
    if importlib.util.find_spec("matplotlib") is None:
        return []
    import matplotlib.pyplot as plt

    created: List[str] = []
    high_risk = scored[scored["risk_band"] == "High"].copy()
    suspicious = scored[scored["risk_band"].isin(["Review", "High"])].copy()
    strong_reason_mask = high_risk["reason_codes"].apply(
        lambda value: has_any_reason(value, STRONG_PRESENTATION_REASONS)
    ) if not high_risk.empty else pd.Series([], dtype=bool)
    high_non_empty_mask = high_risk["stripped_char_len"].fillna(0).gt(0) if not high_risk.empty else pd.Series([], dtype=bool)

    funnel_labels = [
        "Total scored",
        "Review + High",
        "High",
        "High + strong reason",
        "High + non-empty + strong",
    ]
    funnel_values = [
        len(scored),
        len(suspicious),
        len(high_risk),
        int(strong_reason_mask.sum()) if len(strong_reason_mask) else 0,
        int((strong_reason_mask & high_non_empty_mask).sum()) if len(strong_reason_mask) else 0,
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#94a3b8", "#f97316", "#dc2626", "#b91c1c", "#7f1d1d"]
    ax.bar(funnel_labels, funnel_values, color=colors)
    ax.set_title("Risk Funnel: Selective Review Pool to Strong High-Risk Evidence")
    ax.set_ylabel("Post count")
    ax.tick_params(axis="x", rotation=18)
    for idx, value in enumerate(funnel_values):
        share = value / max(len(scored), 1)
        ax.text(idx, value, f"{value:,}\n{share:.1%}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path = artifacts_dir / "risk_funnel.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    created.append(str(path))

    reason_counts = (
        high_risk["reason_codes"].fillna("").apply(split_reason_codes).explode().value_counts()
        if not high_risk.empty
        else pd.Series(dtype=int)
    )
    reason_counts = reason_counts[~reason_counts.index.isin(["LOW_CONTENT_RISK", "LOW_OVERALL_RISK"])].head(14)
    fig, ax = plt.subplots(figsize=(11, 6))
    if not reason_counts.empty:
        ax.barh(reason_counts.index.tolist()[::-1], reason_counts.values[::-1], color="#dc2626")
        ax.set_xlabel("High-risk post count")
    else:
        ax.text(0.5, 0.5, "No high-risk reason codes found", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("High-Risk Explanation Coverage by Reason Code")
    fig.tight_layout()
    path = artifacts_dir / "reason_code_breakdown.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    created.append(str(path))

    trigger_labels = {
        "fomo_trigger": "FOMO",
        "urgency_bait": "Urgency",
        "loss_aversion_trigger": "Loss aversion",
        "social_proof_trigger": "Social proof",
        "authority_impersonation": "Authority",
    }
    trigger_counts = []
    for group_name, label in trigger_labels.items():
        col = f"{group_name}_count"
        count = int((suspicious[col].fillna(0) > 0).sum()) if col in suspicious.columns else 0
        trigger_counts.append((label, count))
    fig, ax = plt.subplots(figsize=(10, 6))
    labels, values = zip(*trigger_counts)
    ax.bar(labels, values, color=["#b91c1c", "#ef4444", "#f97316", "#fb7185", "#7c2d12"])
    ax.set_title("Psychological Trigger Coverage in Review + High Content")
    ax.set_ylabel("Post count")
    ax.tick_params(axis="x", rotation=15)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path = artifacts_dir / "psychological_trigger_breakdown.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    created.append(str(path))

    band_colors = {"High": "#dc2626", "Review": "#f97316", "Low": "#94a3b8"}
    benchmark_plot = benchmark.sort_values("risk_score", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(
        benchmark_plot["category"],
        benchmark_plot["risk_score"].astype(float),
        color=[band_colors.get(band, "#94a3b8") for band in benchmark_plot["risk_band"]],
    )
    ax.axvline(REVIEW_THRESHOLD, color="#f97316", linestyle="--", linewidth=1.5, label=f"Review >= {REVIEW_THRESHOLD:.2f}")
    ax.axvline(RISK_THRESHOLD, color="#991b1b", linestyle="--", linewidth=1.5, label=f"High >= {RISK_THRESHOLD:.2f}")
    ax.set_xlim(0, 1)
    ax.set_title("Live Inference Readiness Scenarios")
    ax.set_xlabel("Risk score")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = artifacts_dir / "live_inference_benchmark.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    created.append(str(path))

    cluster_cols = [
        "narrative_signature",
        "coordination_risk",
        "cluster_confidence_score",
        "cluster_size",
        "cluster_author_nunique",
        "cluster_window_hours",
    ]
    existing_cluster_cols = [col for col in cluster_cols if col in scored.columns]
    clusters = scored[existing_cluster_cols].drop_duplicates() if existing_cluster_cols else pd.DataFrame()
    clusters = clusters[
        clusters.get("coordination_risk", pd.Series(dtype=float)).fillna(0.0).gt(0.0)
    ] if not clusters.empty else clusters
    if not clusters.empty:
        clusters = clusters[
            clusters["cluster_window_hours"].fillna(float("inf")).le(72)
            & (~clusters["narrative_signature"].fillna("").apply(is_weak_narrative_signature))
        ].head(100)
    fig, ax = plt.subplots(figsize=(11, 6))
    if not clusters.empty:
        sizes = np.clip(clusters["cluster_size"].fillna(1).astype(float) * 28, 70, 1200)
        scatter = ax.scatter(
            clusters["cluster_window_hours"].fillna(72),
            clusters["cluster_confidence_score"].fillna(0.0),
            s=sizes,
            c=clusters["coordination_risk"].fillna(0.0),
            cmap="Reds",
            alpha=0.75,
            edgecolor="#7f1d1d",
            linewidth=0.5,
        )
        ax.set_xlabel("Cluster time window (hours)")
        ax.set_ylabel("Cluster confidence score")
        ax.set_ylim(0, 1.02)
        fig.colorbar(scatter, ax=ax, label="Coordination risk")
    else:
        ax.text(0.5, 0.5, "No coordination clusters passed the display filters", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("Coordination Confidence: Size, Author Spread, and Time Window")
    fig.tight_layout()
    path = artifacts_dir / "coordination_confidence_bubble.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    created.append(str(path))

    quality_items = [
        ("Non-empty high risk", float(high_non_empty_mask.mean()) if len(high_non_empty_mask) else 0.0),
        ("Strong reason", float(strong_reason_mask.mean()) if len(strong_reason_mask) else 0.0),
        ("Content score >= 0.65", float(high_risk["content_risk"].fillna(0).ge(0.65).mean()) if not high_risk.empty else 0.0),
        ("Author support >= 0.65", float(high_risk["author_risk"].fillna(0).ge(0.65).mean()) if not high_risk.empty else 0.0),
        ("Coordination support >= 0.65", float(high_risk["coordination_risk"].fillna(0).ge(0.65).mean()) if not high_risk.empty else 0.0),
        (
            "Psychological trigger",
            float(
                high_risk["reason_codes"].apply(
                    lambda value: has_any_reason(
                        value,
                        {
                            "FOMO_TRIGGER",
                            "URGENCY_LANGUAGE",
                            "LOSS_AVERSION_TRIGGER",
                            "SOCIAL_PROOF_TRIGGER",
                            "AUTHORITY_IMPERSONATION",
                        },
                    )
                ).mean()
            )
            if not high_risk.empty
            else 0.0,
        ),
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    labels = [item[0] for item in quality_items][::-1]
    values = [item[1] for item in quality_items][::-1]
    ax.barh(labels, values, color="#b91c1c")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of High-risk posts")
    ax.set_title("Evidence Quality Summary for High-Risk Decisions")
    for idx, value in enumerate(values):
        ax.text(value, idx, f" {value:.1%}", va="center", fontsize=9)
    fig.tight_layout()
    path = artifacts_dir / "evidence_quality_summary.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    created.append(str(path))

    return created


def _numeric_column(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def _feature_contribution_rows(
    scored: pd.DataFrame,
    contribution_items: List[Tuple[str, str, str, pd.Series, str]],
) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame(
            columns=[
                "family",
                "feature",
                "contribution_type",
                "active_rows",
                "active_share",
                "mean_contribution_all",
                "mean_contribution_review_high",
                "mean_contribution_high",
                "total_contribution",
                "explanation",
            ]
        )

    review_high = scored["risk_band"].isin(["Review", "High"]) if "risk_band" in scored else pd.Series(False, index=scored.index)
    high = scored["risk_band"].eq("High") if "risk_band" in scored else pd.Series(False, index=scored.index)
    rows: List[Dict[str, Any]] = []
    for family, feature, contribution_type, values, explanation in contribution_items:
        contributions = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0)
        active = contributions > 1e-9
        rows.append(
            {
                "family": family,
                "feature": feature,
                "contribution_type": contribution_type,
                "active_rows": int(active.sum()),
                "active_share": float(active.mean()) if len(active) else 0.0,
                "mean_contribution_all": float(contributions.mean()) if len(contributions) else 0.0,
                "mean_contribution_review_high": float(contributions[review_high].mean()) if review_high.any() else 0.0,
                "mean_contribution_high": float(contributions[high].mean()) if high.any() else 0.0,
                "total_contribution": float(contributions.sum()),
                "explanation": explanation,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_contribution_high", "mean_contribution_review_high", "total_contribution"],
        ascending=False,
    )


def build_feature_contribution_artifacts(
    scored: pd.DataFrame,
    artifacts_dir: Path,
    make_plots: bool = True,
) -> Dict[str, Any]:
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    if scored.empty:
        return {}

    char_len = _numeric_column(scored, "stripped_char_len")
    word_count = _numeric_column(scored, "word_count")
    empty = _numeric_column(scored, "empty_text") >= 1.0
    short_or_tiny = pd.Series(np.where(char_len < 24, _clip01((24.0 - char_len) / 24.0), 0.0), index=scored.index)
    very_long = pd.Series(_clip01((char_len - 700.0) / 1300.0), index=scored.index)
    repetition_raw = pd.Series(
        _clip01(np.maximum(_numeric_column(scored, "repetition_ratio"), _numeric_column(scored, "max_token_share") - 0.15)),
        index=scored.index,
    )
    repetition = pd.Series(np.where(word_count >= 3, repetition_raw, 0.0), index=scored.index)
    link_density = pd.Series(_clip01(_numeric_column(scored, "link_signal_density") * 5.0), index=scored.index)
    caps = pd.Series(_clip01((_numeric_column(scored, "uppercase_ratio") - 0.35) / 0.45), index=scored.index)
    punctuation = pd.Series(_clip01(_numeric_column(scored, "punctuation_density") * 18.0), index=scored.index)
    sentiment_extreme = pd.Series(np.power(_numeric_column(scored, "sentiment_abs"), 1.4), index=scored.index)
    cta = pd.Series(
        _clip01(_numeric_column(scored, "call_to_action_count") / np.maximum(word_count, 1) * 7.0),
        index=scored.index,
    )
    scam_signal = pd.Series(_clip01(_numeric_column(scored, "scam_signal_count") / 4.0), index=scored.index)
    scam_groups = _numeric_column(scored, "scam_signal_groups")

    base_content = (
        0.20 * short_or_tiny
        + 0.10 * very_long
        + 0.20 * repetition
        + 0.17 * link_density
        + 0.10 * caps
        + 0.08 * punctuation
        + 0.10 * sentiment_extreme
        + 0.05 * cta
        + 0.20 * scam_signal
    )
    current = base_content.copy()
    stacked_spam = (link_density >= 0.70) & (punctuation >= 0.50) & (cta >= 0.70)
    after_stacked = pd.Series(np.where(stacked_spam, np.maximum(current, 0.82), current), index=scored.index)
    stacked_lift = after_stacked - current
    current = after_stacked

    high_risk_lure = (
        (_numeric_column(scored, "phishing_threat_count") > 0)
        | (_numeric_column(scored, "prize_fee_count") >= 2)
        | ((_numeric_column(scored, "financial_bait_count") > 0) & ((_numeric_column(scored, "engagement_bait_count") > 0) | (cta >= 0.35)))
        | ((_numeric_column(scored, "engagement_bait_count") > 0) & (cta >= 0.50))
        | ((_numeric_column(scored, "adult_or_leak_bait_count") > 0) & ((_numeric_column(scored, "engagement_bait_count") > 0) | (_numeric_column(scored, "urgency_bait_count") > 0)))
        | ((_numeric_column(scored, "health_miracle_count") > 0) & (cta >= 0.20))
        | ((_numeric_column(scored, "debt_relief_count") > 0) & (cta >= 0.10))
        | (_numeric_column(scored, "wallet_verification_count") > 0)
        | (_numeric_column(scored, "trading_bot_bait_count") > 0)
        | ((_numeric_column(scored, "loss_aversion_trigger_count") > 0) & (_numeric_column(scored, "urgency_bait_count") > 0))
        | (
            (_numeric_column(scored, "authority_impersonation_count") > 0)
            & (
                (_numeric_column(scored, "urgency_bait_count") > 0)
                | (_numeric_column(scored, "loss_aversion_trigger_count") > 0)
                | (_numeric_column(scored, "phishing_threat_count") > 0)
                | (_numeric_column(scored, "wallet_verification_count") > 0)
                | (cta >= 0.20)
            )
        )
        | (
            (_numeric_column(scored, "fomo_trigger_count") > 0)
            & (
                (_numeric_column(scored, "financial_bait_count") > 0)
                | (_numeric_column(scored, "market_manipulation_count") > 0)
                | (_numeric_column(scored, "engagement_bait_count") > 0)
            )
            & ((_numeric_column(scored, "urgency_bait_count") > 0) | (cta >= 0.20))
        )
        | (
            (_numeric_column(scored, "market_manipulation_count") > 0)
            & (
                (_numeric_column(scored, "financial_bait_count") > 0)
                | (_numeric_column(scored, "engagement_bait_count") > 0)
                | (_numeric_column(scored, "urgency_bait_count") > 0)
            )
        )
    )
    after_high_lure = pd.Series(np.where(high_risk_lure, np.maximum(current, 0.72), current), index=scored.index)
    high_lure_lift = after_high_lure - current
    current = after_high_lure

    political_high = (
        ((_numeric_column(scored, "political_mobilization_count") >= 2) & ((_numeric_column(scored, "urgency_bait_count") > 0) | (cta >= 0.20)))
        | ((_numeric_column(scored, "political_mobilization_count") > 0) & (cta >= 0.35))
    )
    after_political = pd.Series(np.where(political_high, np.maximum(current, 0.66), current), index=scored.index)
    political_lift = after_political - current
    current = after_political

    review_lure = (
        (scam_groups >= 2)
        | ((_numeric_column(scored, "engagement_bait_count") > 0) & (cta >= 0.25))
        | (_numeric_column(scored, "political_mobilization_count") > 0)
        | ((_numeric_column(scored, "market_manipulation_count") > 0) & (_numeric_column(scored, "urgency_bait_count") > 0))
        | (_numeric_column(scored, "fomo_trigger_count") > 0)
        | (_numeric_column(scored, "loss_aversion_trigger_count") > 0)
        | (_numeric_column(scored, "social_proof_trigger_count") > 0)
        | (_numeric_column(scored, "authority_impersonation_count") > 0)
    )
    after_review = pd.Series(np.where(review_lure, np.maximum(current, 0.50), current), index=scored.index)
    review_lure_lift = after_review - current
    current = after_review

    after_empty = pd.Series(np.where(empty, np.maximum(current, 0.92), current), index=scored.index)
    empty_lift = after_empty - current
    current = after_empty
    after_repeated_char = pd.Series(
        np.where(_numeric_column(scored, "repeated_char_run") > 0, np.maximum(current, 0.55), current),
        index=scored.index,
    )
    repeated_char_lift = after_repeated_char - current

    content_items: List[Tuple[str, str, str, pd.Series, str]] = [
        ("Content", "Short/tiny text", "weighted additive", 0.20 * short_or_tiny, "Very short text base signal; calibrated so non-empty short replies remain low evidence."),
        ("Content", "Very long text", "weighted additive", 0.10 * very_long, "Extremely long text can indicate copied/promotional payloads."),
        ("Content", "Token repetition", "weighted additive", 0.20 * repetition, "Repeated tokens or one dominant token."),
        ("Content", "Link/hashtag/mention density", "weighted additive", 0.17 * link_density, "Dense external-link, hashtag, or mention usage."),
        ("Content", "Uppercase intensity", "weighted additive", 0.10 * caps, "Excessive uppercase emphasis."),
        ("Content", "Punctuation burst", "weighted additive", 0.08 * punctuation, "Excessive punctuation such as !!! or ???."),
        ("Content", "Extreme sentiment", "weighted additive", 0.10 * sentiment_extreme, "Very high absolute sentiment value from dataset metadata."),
        ("Content", "Call-to-action density", "weighted additive", 0.05 * cta, "Click, join, follow, DM, buy, and similar action language."),
        ("Content", "Scam/manipulation pattern groups", "weighted additive", 0.20 * scam_signal, "Finance, phishing, prize, urgency, FOMO, authority, and related pattern groups."),
        ("Content floor", "Stacked spam floor", "rule floor lift", stacked_lift, "Dense links, punctuation, and CTA together force a high content-risk floor."),
        ("Content floor", "High-risk lure floor", "rule floor lift", high_lure_lift, "Phishing, wallet verification, trading bot, financial bait, or similar high-risk combinations."),
        ("Content floor", "Political amplification floor", "rule floor lift", political_lift, "Political mobilization combined with urgency or CTA."),
        ("Content floor", "Review lure floor", "rule floor lift", review_lure_lift, "Multiple weaker manipulation groups push content into Review territory."),
        ("Content floor", "Empty text floor", "rule floor lift", empty_lift, "Empty/whitespace-only text is treated as a data/content anomaly."),
        ("Content floor", "Repeated character floor", "rule floor lift", repeated_char_lift, "Long repeated character runs such as aaaaa or 11111."),
    ]

    content_summary = _feature_contribution_rows(scored, content_items)
    content_summary_path = artifacts_dir / "feature_contribution_summary.csv"
    content_summary.to_csv(content_summary_path, index=False)

    content_weight = 0.35
    author_weight = 0.30
    coordination_weight = 0.35
    author_sample_count = _numeric_column(scored, "author_post_count_sample")
    author_full_count = _numeric_column(scored, "author_post_count_full")
    author_available = (
        scored.get(AUTHOR_COL, pd.Series("", index=scored.index)).fillna("").astype(str).str.len().gt(0)
        & ((author_sample_count >= 3) | (author_full_count >= 5))
    )
    coord_available = _numeric_column(scored, "coordination_risk") > 0
    denominator = pd.Series(content_weight, index=scored.index, dtype=float)
    denominator += np.where(author_available, author_weight, 0.0)
    denominator += np.where(coord_available, coordination_weight, 0.0)
    content_component = content_weight * _numeric_column(scored, "content_risk") / denominator
    author_component = pd.Series(
        np.where(author_available, author_weight * _numeric_column(scored, "author_risk") / denominator, 0.0),
        index=scored.index,
    )
    coordination_component = pd.Series(
        np.where(coord_available, coordination_weight * _numeric_column(scored, "coordination_risk") / denominator, 0.0),
        index=scored.index,
    )
    combined = content_component + author_component + coordination_component
    coord = _numeric_column(scored, "coordination_risk")
    author = _numeric_column(scored, "author_risk")
    content_risk = _numeric_column(scored, "content_risk")
    after_coord_high = pd.Series(np.where(coord >= 0.80, np.maximum(combined, 0.68), combined), index=scored.index)
    coord_high_lift = after_coord_high - combined
    combined = after_coord_high
    after_coord_content = pd.Series(np.where((coord >= 0.65) & (content_risk >= 0.35), np.maximum(combined, 0.66), combined), index=scored.index)
    coord_content_lift = after_coord_content - combined
    combined = after_coord_content
    after_author_floor = pd.Series(np.where((author >= 0.80) & ((content_risk >= 0.35) | (coord >= 0.35)), np.maximum(combined, 0.66), combined), index=scored.index)
    author_floor_lift = after_author_floor - combined

    component_items: List[Tuple[str, str, str, pd.Series, str]] = [
        ("Final risk component", "Content component", "normalized weighted component", content_component, "Content-risk contribution after metadata availability renormalization."),
        ("Final risk component", "Author component", "normalized weighted component", author_component, "Author behavior contribution when enough author metadata is available."),
        ("Final risk component", "Coordination component", "normalized weighted component", coordination_component, "Narrative cluster contribution when coordination evidence exists."),
        ("Final risk floor", "High coordination floor", "rule floor lift", coord_high_lift, "Coordination risk >= 0.80 lifts final risk to at least 0.68."),
        ("Final risk floor", "Coordination + content floor", "rule floor lift", coord_content_lift, "Coordination risk >= 0.65 with content support lifts final risk to at least 0.66."),
        ("Final risk floor", "Author + support floor", "rule floor lift", author_floor_lift, "High author risk with content or coordination support lifts final risk to at least 0.66."),
    ]
    component_summary = _feature_contribution_rows(scored, component_items)
    component_summary_path = artifacts_dir / "risk_component_contribution_summary.csv"
    component_summary.to_csv(component_summary_path, index=False)

    md_path = artifacts_dir / "feature_contribution_summary.md"
    top_rows = content_summary.head(12).copy()
    component_rows = component_summary.copy()
    md_lines = [
        "# Feature Contribution Summary",
        "",
        "This is a formula-based proxy contribution report, not supervised feature importance. There are no ground-truth labels, so the numbers summarize how much each engineered signal contributes to the current risk scoring formula.",
        "",
        "## Top Feature Contributions Among High-Risk Rows",
        *_markdown_table(
            [
                {
                    "feature": row["feature"],
                    "family": row["family"],
                    "mean_high": f"{row['mean_contribution_high']:.4f}",
                    "active_share": f"{row['active_share']:.2%}",
                    "type": row["contribution_type"],
                }
                for _, row in top_rows.iterrows()
            ],
            ["feature", "family", "mean_high", "active_share", "type"],
        ),
        "",
        "## Final Risk Component Contributions",
        *_markdown_table(
            [
                {
                    "feature": row["feature"],
                    "mean_all": f"{row['mean_contribution_all']:.4f}",
                    "mean_high": f"{row['mean_contribution_high']:.4f}",
                    "active_share": f"{row['active_share']:.2%}",
                }
                for _, row in component_rows.iterrows()
            ],
            ["feature", "mean_all", "mean_high", "active_share"],
        ),
        "",
        "## Interpretation Note",
        "High values mean the feature frequently and materially contributes to the rule-based risk score. They do not prove causal predictive power against ground-truth labels.",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    plot_paths: List[str] = []
    if make_plots and importlib.util.find_spec("matplotlib") is not None:
        import matplotlib.pyplot as plt

        plot_data = content_summary.sort_values("mean_contribution_high", ascending=False).head(14)
        fig, ax = plt.subplots(figsize=(11, 7))
        if not plot_data.empty:
            labels = plot_data["feature"].tolist()[::-1]
            values = plot_data["mean_contribution_high"].tolist()[::-1]
            ax.barh(labels, values, color="#7c3aed")
            ax.set_xlabel("Mean contribution among High-risk rows")
        else:
            ax.text(0.5, 0.5, "No feature contributions available", ha="center", va="center")
            ax.set_axis_off()
        ax.set_title("Proxy Feature Importance: Formula Contribution to High-Risk Scores")
        fig.tight_layout()
        path = artifacts_dir / "feature_importance_proxy.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plot_paths.append(str(path))

        component_plot = pd.DataFrame(
            {
                "risk_band": scored.get("risk_band", pd.Series("Unknown", index=scored.index)).fillna("Unknown"),
                "Content": content_component,
                "Author": author_component,
                "Coordination": coordination_component,
                "Rule floors": coord_high_lift + coord_content_lift + author_floor_lift,
            }
        )
        component_plot = component_plot.groupby("risk_band")[["Content", "Author", "Coordination", "Rule floors"]].mean()
        component_plot = component_plot.reindex(["Low", "Review", "High"]).dropna(how="all")
        fig, ax = plt.subplots(figsize=(10, 6))
        if not component_plot.empty:
            bottom = np.zeros(len(component_plot))
            colors = {
                "Content": "#ef4444",
                "Author": "#7c3aed",
                "Coordination": "#2563eb",
                "Rule floors": "#f97316",
            }
            x = np.arange(len(component_plot.index))
            for column in component_plot.columns:
                values = component_plot[column].fillna(0.0).values
                ax.bar(x, values, bottom=bottom, label=column, color=colors[column])
                bottom += values
            ax.set_xticks(x)
            ax.set_xticklabels(component_plot.index.tolist())
            ax.set_ylabel("Mean final risk contribution")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No component contributions available", ha="center", va="center")
            ax.set_axis_off()
        ax.set_title("Final Risk Component Contribution by Risk Band")
        fig.tight_layout()
        path = artifacts_dir / "risk_component_contribution.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plot_paths.append(str(path))

    return {
        "feature_contribution_summary_csv": str(content_summary_path),
        "risk_component_contribution_summary_csv": str(component_summary_path),
        "feature_contribution_summary_md": str(md_path),
        "feature_contribution_plots": plot_paths,
    }


def build_marketing_artifacts(
    scored: pd.DataFrame,
    context: Dict[str, pd.DataFrame],
    artifacts_dir: Path,
    make_plots: bool = True,
) -> Tuple[Dict[str, Any], List[str]]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    benchmark = build_live_inference_benchmark(context, artifacts_dir)
    scorecard_summary = build_marketing_scorecard(scored, benchmark, artifacts_dir)
    plot_paths = build_marketing_plots(scored, benchmark, artifacts_dir) if make_plots else []
    return scorecard_summary, plot_paths


def build_plots(scored: pd.DataFrame, artifacts_dir: Path) -> List[str]:
    if importlib.util.find_spec("matplotlib") is None:
        return []
    import matplotlib.pyplot as plt

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    created: List[str] = []
    suspicious = scored[scored["risk_band"].isin(["Review", "High"])].copy()

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(scored["risk_score"].fillna(0.0), bins=np.linspace(0, 1, 41), color="#ef4444", alpha=0.82)
    ax.axvline(REVIEW_THRESHOLD, color="#f97316", linestyle="--", linewidth=1.8, label=f"Review >= {REVIEW_THRESHOLD:.2f}")
    ax.axvline(RISK_THRESHOLD, color="#991b1b", linestyle="--", linewidth=1.8, label=f"High >= {RISK_THRESHOLD:.2f}")
    ax.set_title("Risk Score Distribution")
    ax.set_xlabel("Risk score")
    ax.set_ylabel("Post count")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = artifacts_dir / "risk_score_histogram.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    created.append(str(path))

    if not suspicious.empty:
        top_langs = suspicious["language"].value_counts().head(12).index
        top_platforms = suspicious["platform_domain"].value_counts().head(12).index
        pivot = pd.pivot_table(
            suspicious[
                suspicious["language"].isin(top_langs)
                & suspicious["platform_domain"].isin(top_platforms)
            ],
            index="language",
            columns="platform_domain",
            values="risk_score",
            aggfunc="count",
            fill_value=0,
        )
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(12, 7))
            reds = plt.get_cmap("Reds").copy()
            reds.set_over("#67000d")
            image = ax.imshow(pivot.values, aspect="auto", cmap=reds, vmin=0, vmax=700)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_title("Review + High Risk Content Heatmap: Language x Platform (0-700)")
            ax.set_xlabel("Platform")
            ax.set_ylabel("Language")
            fig.colorbar(image, ax=ax, label="Review + High post count, fixed 0-700 scale", extend="max")
            fig.tight_layout()
            path = artifacts_dir / "risk_map_language_platform.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            created.append(str(path))

        language_summary = (
            scored.groupby("language", dropna=False)
            .agg(
                total_posts=("risk_score", "size"),
                manipulative_posts=("risk_band", lambda values: int((values == "High").sum())),
            )
            .reset_index()
        )
        language_summary["manipulative_rate"] = (
            language_summary["manipulative_posts"] / language_summary["total_posts"].clip(lower=1)
        )
        language_plot = (
            language_summary[language_summary["total_posts"] >= max(100, int(len(scored) * 0.0005))]
            .sort_values(["manipulative_posts", "manipulative_rate"], ascending=False)
            .head(15)
        )
        if not language_plot.empty:
            fig, ax_count = plt.subplots(figsize=(12, 6))
            labels = language_plot["language"].fillna("unknown").astype(str).tolist()
            x = np.arange(len(labels))
            ax_count.bar(x, language_plot["manipulative_posts"], color="#dc2626", alpha=0.86, label="Manipulative count")
            ax_count.set_ylabel("Manipulative post count")
            ax_count.set_xlabel("Language")
            ax_count.set_xticks(x)
            ax_count.set_xticklabels(labels, rotation=35, ha="right")
            ax_rate = ax_count.twinx()
            ax_rate.plot(
                x,
                language_plot["manipulative_rate"] * 100.0,
                color="#111827",
                marker="o",
                linewidth=2,
                label="Manipulative rate",
            )
            ax_rate.set_ylabel("Manipulative rate within language (%)")
            ax_count.set_title("Manipulative Posts by Language: Count and Within-Language Share")
            count_handles, count_labels = ax_count.get_legend_handles_labels()
            rate_handles, rate_labels = ax_rate.get_legend_handles_labels()
            ax_count.legend(count_handles + rate_handles, count_labels + rate_labels, loc="upper right")
            ax_count.grid(axis="y", alpha=0.22)
            fig.tight_layout()
            path = artifacts_dir / "language_manipulative_share.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            created.append(str(path))

        segments = (
            suspicious.groupby(["language", "platform_domain"], dropna=False)
            .size()
            .sort_values(ascending=False)
            .head(12)
        )
        if not segments.empty:
            fig, ax = plt.subplots(figsize=(11, 6))
            labels = [f"{lang} | {platform}" for lang, platform in segments.index]
            ax.barh(labels[::-1], segments.values[::-1], color="#3b82f6")
            ax.set_title("Top Review + High Risk Language/Platform Segments")
            ax.set_xlabel("Review + High post count")
            fig.tight_layout()
            path = artifacts_dir / "top_suspicious_segments.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            created.append(str(path))

        global_review_high_share = len(suspicious) / max(len(scored), 1)
        platform_norm = (
            scored.groupby("platform_domain", dropna=False)
            .agg(
                posts=("risk_score", "size"),
                review_high_posts=("risk_band", lambda values: int((values != "Low").sum())),
            )
            .reset_index()
        )
        platform_norm["review_high_rate"] = platform_norm["review_high_posts"] / platform_norm["posts"].clip(lower=1)
        platform_norm["platform_normalized_risk_index"] = (
            platform_norm["review_high_rate"] / max(global_review_high_share, 1e-12)
        )
        min_posts = max(50, int(len(scored) * 0.0005))
        top_platform_norm = (
            platform_norm[platform_norm["posts"] >= min_posts]
            .sort_values("platform_normalized_risk_index", ascending=False)
            .head(12)
        )
        if not top_platform_norm.empty:
            fig, ax = plt.subplots(figsize=(11, 6))
            labels = top_platform_norm["platform_domain"].tolist()[::-1]
            values = top_platform_norm["platform_normalized_risk_index"].tolist()[::-1]
            ax.barh(labels, values, color="#dc2626")
            ax.axvline(1.0, color="#111827", linestyle="--", linewidth=1.3, label="Dataset average")
            ax.set_title("Platform-Normalized Review + High Risk Index")
            ax.set_xlabel("Risk index vs dataset average")
            ax.legend()
            fig.tight_layout()
            path = artifacts_dir / "platform_normalized_risk.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            created.append(str(path))

        author_source = scored[scored[AUTHOR_COL].fillna("").astype(str).str.strip().ne("")].copy()
        if not author_source.empty and "author_risk" in author_source.columns:
            author_plot = (
                author_source.groupby(AUTHOR_COL, dropna=False)
                .agg(author_risk=("author_risk", "max"), sample_posts=("risk_score", "size"))
                .reset_index()
                .sort_values(["author_risk", "sample_posts"], ascending=False)
                .head(12)
            )
            if not author_plot.empty:
                fig, ax = plt.subplots(figsize=(11, 6))
                labels = [f"{str(value)[:8]}..." for value in author_plot[AUTHOR_COL].tolist()][::-1]
                values = author_plot["author_risk"].tolist()[::-1]
                ax.barh(labels, values, color="#7c3aed")
                ax.set_xlim(0, 1)
                ax.set_title("Top Risk Authors")
                ax.set_xlabel("Author risk score")
                fig.tight_layout()
                path = artifacts_dir / "top_risk_authors.png"
                fig.savefig(path, dpi=160)
                plt.close(fig)
                created.append(str(path))

    with_dates = scored.dropna(subset=["date_ts"]).copy()
    if not with_dates.empty:
        with_dates["hour"] = with_dates["date_ts"].dt.floor("h")
        hourly = (
            with_dates.groupby("hour")
            .agg(
                posts=("risk_score", "size"),
                avg_risk=("risk_score", "mean"),
                suspicious_posts=("risk_band", lambda values: int((values != "Low").sum())),
                high_posts=("risk_band", lambda values: int((values == "High").sum())),
            )
            .reset_index()
        )
        hourly["suspicious_share"] = hourly["suspicious_posts"] / hourly["posts"].clip(lower=1)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(hourly["hour"], hourly["suspicious_share"], color="#ef4444", linewidth=1.8)
        ax.set_title("Hourly Review + High Risk Share")
        ax.set_xlabel("Time")
        ax.set_ylabel("Suspicious share")
        ax.grid(alpha=0.25)
        fig.autofmt_xdate()
        fig.tight_layout()
        path = artifacts_dir / "hourly_suspicious_share.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        created.append(str(path))

        temporal_bursts = build_temporal_burst_windows(hourly)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(
            temporal_bursts["hour"],
            temporal_bursts["suspicious_share"],
            color="#ef4444",
            linewidth=1.8,
            label="Review + High share",
        )
        burst_points = temporal_bursts[temporal_bursts["burst_label"] == "Campaign Burst"]
        if not burst_points.empty:
            ax.scatter(
                burst_points["hour"],
                burst_points["suspicious_share"],
                color="#7f1d1d",
                s=np.clip(burst_points["burst_score"].astype(float) * 42, 70, 260),
                zorder=3,
                label="Campaign Burst",
            )
        ax.set_title("Temporal Burst Windows: Review + High Share Z-Score")
        ax.set_xlabel("Time")
        ax.set_ylabel("Suspicious share")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.autofmt_xdate()
        fig.tight_layout()
        path = artifacts_dir / "temporal_burst_windows.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        created.append(str(path))

    return created


def _json_safe(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def write_pitch_deck(summary: Dict[str, Any], artifacts_dir: Path) -> Path:
    suspicious_share = summary.get("suspicious_share", 0.0)
    high_risk_share = summary.get("high_risk_share", 0.0)
    top_segments = summary.get("top_segments", [])[:5]
    top_clusters = summary.get("top_clusters", [])[:3]
    top_authors = summary.get("top_authors", [])[:3]
    top_platforms = summary.get("top_platform_normalized", [])[:3]
    marketing_scorecard = summary.get("marketing_scorecard", [])[:6]

    segment_lines = "\n".join(
        f"- {row.get('language', 'unknown')} / {row.get('platform_domain', 'unknown')} / "
        f"{row.get('primary_theme', 'unknown')}: {row.get('suspicious_posts', 0)} Review/High gönderi "
        f"({row.get('high_posts', 0)} High)"
        for row in top_segments
    ) or "- Üst segmentleri doldurmak için pipeline'ı çalıştırın."

    cluster_lines = "\n".join(
        f"- {row.get('narrative_signature', '')[:90]}: boyut={row.get('cluster_size', 0)}, "
        f"author={row.get('cluster_author_nunique', 0)}, "
        f"pencere={row.get('cluster_window_hours', 0):.1f} saat, "
        f"güven={row.get('cluster_confidence_score', 0):.2f}, risk={row.get('coordination_risk', 0):.2f}"
        for row in top_clusters
    ) or "- Koordinasyon cluster'larını doldurmak için pipeline'ı çalıştırın."

    author_lines = "\n".join(
        f"- {str(row.get(AUTHOR_COL, ''))[:10]}...: author risk={row.get('author_risk', 0):.2f}, "
        f"gönderi={row.get('sample_posts', 0)}, Review/High oranı={row.get('review_high_share', 0):.1%}"
        for row in top_authors
    ) or "- Riskli author özetini doldurmak için pipeline'ı çalıştırın."

    platform_lines = "\n".join(
        f"- {row.get('platform_domain', 'unknown')}: risk indeksi={row.get('platform_normalized_risk_index', 0):.2f}, "
        f"Review/High oranı={row.get('review_high_rate', 0):.1%}"
        for row in top_platforms
    ) or "- Platform-normalize risk özetini doldurmak için pipeline'ı çalıştırın."

    marketing_lines = "\n".join(
        f"- {row.get('metric', 'Metric')}: {row.get('value', '')}"
        for row in marketing_scorecard
    ) or "- Pazarlama scorecard'ını doldurmak için pipeline'ı çalıştırın."

    live_ready = summary.get("live_inference_readiness_passed", 0)
    live_total = summary.get("live_inference_readiness_total", 0)
    temporal_burst_count = summary.get("temporal_burst_count", 0)

    nlp_meta = summary.get("nlp_text_model", {})
    nlp_docs = nlp_meta.get("train_docs_per_class", 0)

    content = f"""# DataLeague Datathon Pitch Deck

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
- Skorlanan örnek satır: {summary.get('sample_count', 0):,}
- Risk bantları: High >= {RISK_THRESHOLD:.2f}, Review >= {REVIEW_THRESHOLD:.2f}, aksi halde Low.
- Örneklemde Review + High oranı: {suspicious_share:.1%}
- Örneklemde High-risk oranı: {high_risk_share:.1%}
- Ana görsel: `artifacts/risk_map_language_platform.png`
- Temporal kampanya pencereleri: {temporal_burst_count} Campaign Burst; görsel `artifacts/temporal_burst_windows.png`

---

## 6. En Şüpheli Segmentler
{segment_lines}
- Platform-normalize risk tablosu: `artifacts/platform_normalized_risk.csv`

---

## 7. Koordinasyon Cluster Örnekleri
{cluster_lines}
- Somut şüpheli gönderi örnekleri: `artifacts/presentation_examples.csv`
- Jüriye anlatılacak hikayeler: `artifacts/case_studies.md`
- En riskli author özetleri: `artifacts/top_risk_authors.csv`

---

## 8. Normalize Platform ve Author Sinyalleri
{platform_lines}
{author_lines}

---

## 9. Kanıt Scorecard
{marketing_lines}
- Ek görseller: `risk_funnel.png`, `reason_code_breakdown.png`, `psychological_trigger_breakdown.png`, `evidence_quality_summary.png`, `feature_importance_proxy.png`, `risk_component_contribution.png`.
- Kampanya dönemi kanıtı: `temporal_burst_windows.csv` ve `temporal_burst_windows.png`.
- Etiket sağlanmadığı için bunlar proxy kanıt metrikleri ve senaryo kontrolleridir; etiketli performans metriği iddiası değildir.

---

## 10. Canlı Demo ve Sınırlar
- Demo fonksiyonu: `predict_live(text, language=None, url=None, author_hash=None, date=None, english_keywords=None)`.
- Dönen alanlar: label, risk score, organic score, nlp_text_risk, top reasons, used features ve psychological triggers.
- Canlı inference hazırlık senaryoları: {live_ready}/{live_total} başarılı.
- Benchmark artifact'i: `artifacts/live_inference_benchmark.csv`.
- NLP text model artifact'i: `artifacts/nlp_text_model.npz`; eğitim özeti: `artifacts/nlp_text_model_metadata.json`.
- NLP eğitim verisi: yüksek güvenli pseudo-positive ve pseudo-negative örneklerden sınıf başına {nlp_docs:,} metin.
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
"""
    path = artifacts_dir / "pitch_deck.md"
    path.write_text(content, encoding="utf-8")
    return path


def save_context(context: Dict[str, Any], artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in context.items():
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(artifacts_dir / f"{name}.csv", index=False)


def load_context(artifacts_dir: Path = ARTIFACT_DIR) -> Dict[str, Any]:
    context: Dict[str, Any] = {}
    for name in ["author_stats", "coordination_stats"]:
        path = artifacts_dir / f"{name}.csv"
        if path.exists():
            context[name] = pd.read_csv(path)
    nlp_model = load_nlp_text_model(artifacts_dir)
    if nlp_model is not None:
        context["nlp_text_model"] = nlp_model
    return context


def run_pipeline(
    data_path: Path = DATA_PATH,
    artifacts_dir: Path = ARTIFACT_DIR,
    sample_size: int = 200_000,
    use_full_author_stats: bool = True,
    make_plots: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Any]]:
    require_runtime_dependencies(include_plotting=make_plots)
    artifacts_dir = Path(artifacts_dir)
    sample = load_sample(data_path=data_path, sample_size=sample_size)
    full_stats = build_full_author_stats(data_path) if use_full_author_stats else None
    scored, context = score_dataframe(sample, full_author_stats=full_stats)
    nlp_model = train_nlp_text_model(scored, artifacts_dir)
    if nlp_model is not None:
        context["nlp_text_model"] = nlp_model

    save_context(context, artifacts_dir)
    table_summary = build_tables(scored, artifacts_dir)
    plot_paths = build_plots(scored, artifacts_dir) if make_plots else []
    marketing_summary, marketing_plot_paths = build_marketing_artifacts(
        scored,
        context,
        artifacts_dir,
        make_plots=make_plots,
    )
    plot_paths.extend(marketing_plot_paths)
    feature_contribution_summary = build_feature_contribution_artifacts(
        scored,
        artifacts_dir,
        make_plots=make_plots,
    )
    plot_paths.extend(feature_contribution_summary.get("feature_contribution_plots", []))
    case_studies_path = build_case_studies(scored, artifacts_dir)

    summary: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "sample_size_requested": int(sample_size),
        "risk_threshold": RISK_THRESHOLD,
        "review_threshold": REVIEW_THRESHOLD,
        "plots": plot_paths,
        "case_studies_path": str(case_studies_path),
        **table_summary,
        **marketing_summary,
        **feature_contribution_summary,
    }
    if nlp_model is not None:
        summary["nlp_text_model"] = nlp_model.get("metadata", {})
    (artifacts_dir / "run_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_pitch_deck(summary, artifacts_dir)
    return scored, context, summary


def _lookup_component(
    frame: Optional[pd.DataFrame],
    key_col: str,
    key_value: str,
    score_col: str,
) -> Tuple[Optional[float], Optional[pd.Series]]:
    if frame is None or frame.empty or not key_value or key_col not in frame.columns:
        return None, None
    matches = frame[frame[key_col].astype(str) == str(key_value)]
    if matches.empty or score_col not in matches.columns:
        return None, None
    row = matches.iloc[0]
    try:
        return float(row[score_col]), row
    except (TypeError, ValueError):
        return None, row


def predict_live(
    text: str,
    language: Optional[str] = None,
    url: Optional[str] = None,
    author_hash: Optional[str] = None,
    date: Optional[str] = None,
    english_keywords: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    artifacts_dir: Path = ARTIFACT_DIR,
) -> Dict[str, Any]:
    if context is None:
        context = load_context(artifacts_dir)

    row = pd.DataFrame(
        [
            {
                TEXT_COL: text or "",
                KEYWORDS_COL: english_keywords or "",
                SENTIMENT_COL: 0.0,
                "main_emotion": "",
                "primary_theme": "",
                "language": language or "unknown",
                URL_COL: url or "",
                AUTHOR_COL: author_hash or "",
                DATE_COL: date or "",
            }
        ]
    )
    prepared = prepare_input_dataframe(row)
    features = compute_text_features(prepared)
    content_risk = float(score_content_features(features).iloc[0])
    content_reasons = content_reason_codes(features.iloc[0])
    nlp_raw_text_risk = predict_nlp_text_risk(text or "", context.get("nlp_text_model"))
    nlp_model_used = nlp_raw_text_risk is not None
    nlp_support_allowed = bool(
        content_risk >= 0.20
        or features.iloc[0].get("call_to_action_count", 0) >= 1
        or features.iloc[0].get("link_signal_density", 0.0) > 0
        or features.iloc[0].get("punctuation_density", 0.0) >= 0.015
        or features.iloc[0].get("uppercase_ratio", 0.0) >= 0.12
        or features.iloc[0].get("scam_signal_count", 0) > 0
    )
    nlp_text_risk = nlp_raw_text_risk if nlp_support_allowed else None
    if nlp_text_risk is not None:
        if nlp_support_allowed and nlp_text_risk >= 0.90:
            content_risk = max(content_risk, 0.72)
        elif nlp_support_allowed and nlp_text_risk >= 0.80:
            content_risk = max(content_risk, 0.66)
        elif nlp_support_allowed and nlp_text_risk >= 0.70:
            content_risk = max(content_risk, 0.50)
        if nlp_support_allowed and nlp_text_risk >= 0.70:
            content_reasons.append("NLP_TEXT_MODEL_SUPPORT")

    author_risk, author_row = _lookup_component(
        context.get("author_stats"), AUTHOR_COL, author_hash or "", "author_risk"
    )
    coord_risk, coord_row = _lookup_component(
        context.get("coordination_stats"),
        "narrative_signature",
        prepared["narrative_signature"].iloc[0],
        "coordination_risk",
    )
    coord_supported = coord_risk is not None and coord_risk > 0

    components: List[Tuple[str, float, float]] = [("content", 0.35, content_risk)]
    reasons = list(content_reasons)
    if author_risk is not None:
        components.append(("author", 0.30, author_risk))
        reasons.extend(author_reason_codes(author_row if author_row is not None else pd.Series()))
    elif author_hash:
        reasons.append("AUTHOR_NOT_SEEN_IN_CONTEXT")

    if coord_supported:
        components.append(("coordination", 0.35, coord_risk))
        reasons.extend(coordination_reason_codes(coord_row if coord_row is not None else pd.Series()))
    elif english_keywords:
        reasons.append("NARRATIVE_NOT_SEEN_IN_CONTEXT")

    if author_risk is None and not coord_supported and not any([language, url, author_hash, date, english_keywords]):
        reasons.append("TEXT_ONLY_METADATA_UNAVAILABLE")

    total_weight = sum(weight for _, weight, _ in components)
    risk_score = sum(weight * score for _, weight, score in components) / max(total_weight, 1e-9)
    risk_score = float(_clip01(risk_score))
    if coord_supported and coord_risk >= 0.80:
        risk_score = max(risk_score, 0.68)
    if coord_supported and coord_risk >= 0.65 and content_risk >= 0.35:
        risk_score = max(risk_score, 0.66)
    if author_risk is not None and author_risk >= 0.80 and (content_risk >= 0.35 or (coord_risk or 0.0) >= 0.35):
        risk_score = max(risk_score, 0.66)
    organic_score = 1.0 - risk_score
    label = "Manipulative" if risk_score >= RISK_THRESHOLD else "Organic"
    risk_band = risk_band_from_score(pd.Series([risk_score])).iloc[0]
    if coord_supported:
        evidence_level = "coordination_supported"
    elif author_risk is not None or any([language, url, author_hash, date, english_keywords]):
        evidence_level = "metadata_supported"
    else:
        evidence_level = "text_only"

    used_features = {
        "content_risk": round(content_risk, 4),
        "nlp_text_risk": None if nlp_text_risk is None else round(float(nlp_text_risk), 4),
        "nlp_raw_text_risk": None if nlp_raw_text_risk is None else round(float(nlp_raw_text_risk), 4),
        "nlp_model_used": bool(nlp_model_used),
        "nlp_support_allowed": bool(nlp_support_allowed),
        "author_risk": None if author_risk is None else round(float(author_risk), 4),
        "coordination_risk": None if coord_risk is None else round(float(coord_risk), 4),
        "narrative_signature": prepared["narrative_signature"].iloc[0],
        "text_features": {
            key: _json_safe(features.iloc[0][key])
            for key in [
                "char_len",
                "word_count",
                "repetition_ratio",
                "link_signal_density",
                "uppercase_ratio",
                "punctuation_density",
                "call_to_action_count",
                "scam_signal_count",
                "scam_signal_groups",
                "url_only_or_link_only",
            ]
        },
        "psychological_triggers": {
            group_name: int(features.iloc[0].get(f"{group_name}_count", 0))
            for group_name in PSYCHOLOGICAL_TRIGGER_GROUPS
        },
    }

    return {
        "model_version": MODEL_VERSION,
        "label": label,
        "risk_score": round(risk_score, 4),
        "risk_band": risk_band,
        "organic_score": round(organic_score, 4),
        "evidence_level": evidence_level,
        "nlp_text_risk": None if nlp_text_risk is None else round(float(nlp_text_risk), 4),
        "nlp_model_used": bool(nlp_model_used),
        "top_reasons": list(dict.fromkeys(reasons))[:8],
        "used_features": used_features,
    }


def _resolve_column(
    df: pd.DataFrame,
    requested_column: Optional[str] = None,
    candidate_names: Optional[Iterable[str]] = None,
) -> Optional[Any]:
    if requested_column:
        if requested_column in df.columns:
            return requested_column
        requested_normalized = normalize_column_name(requested_column)
        for column in df.columns:
            if normalize_column_name(column) == requested_normalized:
                return column
        available = ", ".join(str(column) for column in df.columns)
        raise ValueError(f"Column '{requested_column}' was not found. Available columns: {available}")

    if not candidate_names:
        return None
    normalized_candidates = {normalize_column_name(name) for name in candidate_names}
    for column in df.columns:
        if normalize_column_name(column) in normalized_candidates:
            return column
    return None


def _read_live_inference_csv(
    input_path: Path,
    no_header: bool = False,
    csv_sep: str = ",",
) -> pd.DataFrame:
    input_path = Path(input_path)
    header = None if no_header else "infer"
    df = pd.read_csv(
        input_path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
        sep=None if csv_sep == "auto" else csv_sep,
        engine="python",
        header=header,
    )
    if no_header or df.empty:
        return df

    auto_text_column = _resolve_column(df, candidate_names=TEXT_COLUMN_CANDIDATES)
    if auto_text_column is None and len(df.columns) == 1:
        only_column = str(df.columns[0])
        looks_like_text_value = len(only_column) >= 24 or len(only_column.split()) >= 4
        if looks_like_text_value:
            df = pd.read_csv(
                input_path,
                dtype=str,
                keep_default_na=False,
                encoding="utf-8-sig",
                sep=None if csv_sep == "auto" else csv_sep,
                engine="python",
                header=None,
            )
    return df


def _cell_value(row: pd.Series, column: Optional[Any], fallback: Optional[str] = None) -> Optional[str]:
    if column is None:
        return fallback
    value = _clean_scalar(row.get(column, ""))
    return value if value else fallback


def _count_semicolon_values(values: pd.Series) -> pd.Series:
    counts: Counter[str] = Counter()
    for value in values.fillna("").astype(str):
        for item in split_reason_codes(value):
            counts[item] += 1
    return pd.Series(counts, dtype=int).sort_values(ascending=False)


def _markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> List[str]:
    if not rows:
        return ["No rows available."]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            text = "" if value is None else str(value)
            values.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _build_prediction_batch_plots(predictions: pd.DataFrame, output_csv: Path) -> List[str]:
    if predictions.empty or importlib.util.find_spec("matplotlib") is None:
        return []
    import matplotlib.pyplot as plt

    created: List[str] = []
    output_csv = Path(output_csv)
    output_dir = output_csv.parent
    stem = output_csv.stem
    risk_scores = pd.to_numeric(predictions.get("risk_score", pd.Series(dtype=float)), errors="coerce").fillna(0.0)

    band_counts = (
        predictions.get("risk_band", pd.Series(dtype=str))
        .fillna("Unknown")
        .value_counts()
        .reindex(["Low", "Review", "High"], fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Low": "#94a3b8", "Review": "#f97316", "High": "#dc2626"}
    ax.bar(band_counts.index.tolist(), band_counts.values, color=[colors.get(idx, "#64748b") for idx in band_counts.index])
    ax.set_title("Jury CSV Risk Band Distribution")
    ax.set_xlabel("Risk band")
    ax.set_ylabel("Text count")
    for idx, value in enumerate(band_counts.values):
        share = value / max(len(predictions), 1)
        ax.text(idx, value, f"{int(value):,}\n{share:.1%}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = output_dir / f"{stem}_risk_band_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    created.append(str(path))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(risk_scores, bins=np.linspace(0, 1, 31), color="#ef4444", alpha=0.82)
    ax.axvline(REVIEW_THRESHOLD, color="#f97316", linestyle="--", linewidth=1.7, label=f"Review >= {REVIEW_THRESHOLD:.2f}")
    ax.axvline(RISK_THRESHOLD, color="#991b1b", linestyle="--", linewidth=1.7, label=f"High >= {RISK_THRESHOLD:.2f}")
    ax.set_title("Jury CSV Risk Score Histogram")
    ax.set_xlabel("Risk score")
    ax.set_ylabel("Text count")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = output_dir / f"{stem}_risk_score_histogram.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    created.append(str(path))

    reason_counts = _count_semicolon_values(predictions.get("top_reasons", pd.Series(dtype=str))).head(12)
    fig, ax = plt.subplots(figsize=(10, 6))
    if not reason_counts.empty:
        ax.barh(reason_counts.index.tolist()[::-1], reason_counts.values[::-1], color="#dc2626")
        ax.set_xlabel("Text count")
    else:
        ax.text(0.5, 0.5, "No reason codes found", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("Jury CSV Top Reason Codes")
    fig.tight_layout()
    path = output_dir / f"{stem}_reason_code_breakdown.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    created.append(str(path))

    trigger_labels = {
        "fomo_trigger": "FOMO",
        "urgency_bait": "Urgency",
        "loss_aversion_trigger": "Loss aversion",
        "social_proof_trigger": "Social proof",
        "authority_impersonation": "Authority",
    }
    trigger_counts_raw = _count_semicolon_values(predictions.get("psychological_triggers", pd.Series(dtype=str)))
    trigger_items = [
        (trigger_labels.get(name, name), int(trigger_counts_raw.get(name, 0)))
        for name in trigger_labels
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [item[0] for item in trigger_items]
    values = [item[1] for item in trigger_items]
    ax.bar(labels, values, color=["#b91c1c", "#ef4444", "#f97316", "#fb7185", "#7c2d12"])
    ax.set_title("Jury CSV Psychological Trigger Coverage")
    ax.set_ylabel("Text count")
    ax.tick_params(axis="x", rotation=15)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path = output_dir / f"{stem}_psychological_trigger_breakdown.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    created.append(str(path))

    return created


def _build_prediction_batch_artifacts(
    predictions: pd.DataFrame,
    input_csv: Path,
    output_csv: Path,
    rows_read: int,
) -> Dict[str, Any]:
    output_csv = Path(output_csv)
    output_dir = output_csv.parent
    stem = output_csv.stem
    total = int(len(predictions))
    risk_scores = pd.to_numeric(predictions.get("risk_score", pd.Series(dtype=float)), errors="coerce")
    content_scores = pd.to_numeric(predictions.get("content_risk", pd.Series(dtype=float)), errors="coerce")
    nlp_scores = pd.to_numeric(predictions.get("nlp_text_risk", pd.Series(dtype=float)), errors="coerce")

    band_series = predictions.get("risk_band", pd.Series(dtype=str)).fillna("Unknown")
    band_counts = band_series.value_counts().reindex(["Low", "Review", "High"], fill_value=0)
    review_high_count = int(band_counts.get("Review", 0) + band_counts.get("High", 0))
    manipulative_count = int((predictions.get("label", pd.Series(dtype=str)) == "Manipulative").sum())

    summary: Dict[str, Any] = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "rows_read": int(rows_read),
        "texts_scored": total,
        "low_count": int(band_counts.get("Low", 0)),
        "review_count": int(band_counts.get("Review", 0)),
        "high_count": int(band_counts.get("High", 0)),
        "review_high_count": review_high_count,
        "manipulative_count": manipulative_count,
        "low_share": float(band_counts.get("Low", 0) / max(total, 1)),
        "review_share": float(band_counts.get("Review", 0) / max(total, 1)),
        "high_share": float(band_counts.get("High", 0) / max(total, 1)),
        "review_high_share": float(review_high_count / max(total, 1)),
        "manipulative_share": float(manipulative_count / max(total, 1)),
        "risk_score_mean": float(risk_scores.mean()) if total else 0.0,
        "risk_score_median": float(risk_scores.median()) if total else 0.0,
        "risk_score_p90": float(risk_scores.quantile(0.90)) if total else 0.0,
        "risk_score_max": float(risk_scores.max()) if total else 0.0,
        "content_risk_mean": float(content_scores.mean()) if total else 0.0,
        "nlp_text_risk_mean": float(nlp_scores.mean()) if nlp_scores.notna().any() else None,
    }

    evidence_counts = predictions.get("evidence_level", pd.Series(dtype=str)).fillna("unknown").value_counts()
    for level, count in evidence_counts.items():
        summary[f"evidence_{level}_count"] = int(count)
        summary[f"evidence_{level}_share"] = float(count / max(total, 1))

    reason_counts = _count_semicolon_values(predictions.get("top_reasons", pd.Series(dtype=str)))
    trigger_counts = _count_semicolon_values(predictions.get("psychological_triggers", pd.Series(dtype=str)))
    summary["top_reason_codes"] = reason_counts.head(8).to_dict()
    summary["top_psychological_triggers"] = trigger_counts.head(8).to_dict()

    summary_rows = [
        {"metric": key, "value": value}
        for key, value in summary.items()
        if not isinstance(value, dict)
    ]
    summary_csv = output_dir / f"{stem}_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    top_examples_cols = [
        "test_id",
        "source_row",
        "source_column",
        "risk_score",
        "risk_band",
        "label",
        "top_reasons",
        "psychological_triggers",
        TEXT_COL,
    ]
    existing_top_cols = [col for col in top_examples_cols if col in predictions.columns]
    top_examples = (
        predictions.sort_values("risk_score", ascending=False)
        .head(20)[existing_top_cols]
        .copy()
        if existing_top_cols
        else pd.DataFrame()
    )
    if TEXT_COL in top_examples.columns:
        top_examples["text_preview"] = top_examples[TEXT_COL].fillna("").astype(str).str.slice(0, 220)
        top_examples = top_examples.drop(columns=[TEXT_COL])
    top_examples_csv = output_dir / f"{stem}_top_risk_examples.csv"
    top_examples.to_csv(top_examples_csv, index=False)

    md_path = output_dir / f"{stem}_summary.md"
    overview_rows = [
        {"metric": "Rows read", "value": f"{summary['rows_read']:,}"},
        {"metric": "Texts scored", "value": f"{summary['texts_scored']:,}"},
        {"metric": "Average risk score", "value": f"{summary['risk_score_mean']:.4f}"},
        {"metric": "Median risk score", "value": f"{summary['risk_score_median']:.4f}"},
        {"metric": "P90 risk score", "value": f"{summary['risk_score_p90']:.4f}"},
        {"metric": "Review + High share", "value": f"{summary['review_high_share']:.2%}"},
        {"metric": "High share", "value": f"{summary['high_share']:.2%}"},
        {"metric": "Manipulative share", "value": f"{summary['manipulative_share']:.2%}"},
    ]
    band_rows = [
        {"risk_band": band, "count": int(count), "share": f"{int(count) / max(total, 1):.2%}"}
        for band, count in band_counts.items()
    ]
    reason_rows = [
        {"reason_code": reason, "count": int(count), "share": f"{int(count) / max(total, 1):.2%}"}
        for reason, count in reason_counts.head(12).items()
    ]
    trigger_rows = [
        {"trigger": trigger, "count": int(count), "share": f"{int(count) / max(total, 1):.2%}"}
        for trigger, count in trigger_counts.head(12).items()
    ]
    top_rows = top_examples.head(10).to_dict("records") if not top_examples.empty else []
    md_lines = [
        "# Jury CSV Batch Inference Summary",
        "",
        f"- Input CSV: `{input_csv}`",
        f"- Submission CSV: `{output_csv}`",
        f"- Detailed CSV: `{output_csv.with_name(f'{output_csv.stem}_detailed.csv')}`",
        "",
        "## Overview",
        *_markdown_table(overview_rows, ["metric", "value"]),
        "",
        "## Risk Band Distribution",
        *_markdown_table(band_rows, ["risk_band", "count", "share"]),
        "",
        "## Top Reason Codes",
        *_markdown_table(reason_rows, ["reason_code", "count", "share"]),
        "",
        "## Psychological Triggers",
        *_markdown_table(trigger_rows, ["trigger", "count", "share"]),
        "",
        "## Top Risk Examples",
        *_markdown_table(
            top_rows,
            [col for col in ["test_id", "source_row", "risk_score", "risk_band", "label", "top_reasons", "psychological_triggers", "text_preview"] if col in top_examples.columns],
        ),
        "",
        "## Interpretation Note",
        "These are batch-level inference diagnostics, not ground-truth accuracy metrics. The CSV has no verified labels, so the outputs summarize model selectivity, explanation coverage, and review workload.",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    plot_paths = _build_prediction_batch_plots(predictions, output_csv)
    summary.update(
        {
            "summary_csv": str(summary_csv),
            "summary_markdown": str(md_path),
            "top_risk_examples_csv": str(top_examples_csv),
            "batch_plot_paths": plot_paths,
        }
    )
    return summary


def _build_submission_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    fallback_ids = pd.Series(range(1, len(predictions) + 1), index=predictions.index)
    submission = pd.DataFrame(
        {
            "test_id": predictions.get("test_id", predictions.get("source_row", fallback_ids)),
            "text": predictions.get(TEXT_COL, pd.Series("", index=predictions.index)),
            "label": pd.to_numeric(
                predictions.get("risk_score", pd.Series(0.0, index=predictions.index)),
                errors="coerce",
            )
            .fillna(0.0)
            .clip(0.0, 1.0)
            .round(4),
        }
    )
    return submission[["test_id", "text", "label"]]


def _prediction_to_flat_row(
    prediction: Dict[str, Any],
    source_row: int,
    source_column: str,
    text: str,
    source_id: Optional[str] = None,
) -> Dict[str, Any]:
    used_features = prediction.get("used_features", {})
    text_features = used_features.get("text_features", {})
    trigger_counts = used_features.get("psychological_triggers", {})
    active_triggers = [
        name
        for name, count in trigger_counts.items()
        if isinstance(count, (int, float)) and count > 0
    ]
    row = {
        "source_row": source_row,
        "source_column": source_column,
        TEXT_COL: text,
        "label": prediction.get("label"),
        "risk_band": prediction.get("risk_band"),
        "risk_score": prediction.get("risk_score"),
        "organic_score": prediction.get("organic_score"),
        "evidence_level": prediction.get("evidence_level"),
        "nlp_text_risk": prediction.get("nlp_text_risk"),
        "nlp_model_used": prediction.get("nlp_model_used"),
        "top_reasons": ";".join(prediction.get("top_reasons", [])),
        "psychological_triggers": ";".join(active_triggers),
        "content_risk": used_features.get("content_risk"),
        "author_risk": used_features.get("author_risk"),
        "coordination_risk": used_features.get("coordination_risk"),
        "nlp_raw_text_risk": used_features.get("nlp_raw_text_risk"),
        "nlp_support_allowed": used_features.get("nlp_support_allowed"),
        "narrative_signature": used_features.get("narrative_signature"),
        "char_len": text_features.get("char_len"),
        "word_count": text_features.get("word_count"),
        "link_signal_density": text_features.get("link_signal_density"),
        "call_to_action_count": text_features.get("call_to_action_count"),
        "scam_signal_count": text_features.get("scam_signal_count"),
        "model_version": prediction.get("model_version"),
    }
    if source_id is not None:
        return {
            "test_id": source_id,
            **row,
        }
    return row


def predict_dataframe(
    df: pd.DataFrame,
    text_column: Optional[str] = None,
    all_cells: bool = False,
    language: Optional[str] = None,
    url: Optional[str] = None,
    author_hash: Optional[str] = None,
    date: Optional[str] = None,
    english_keywords: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    artifacts_dir: Path = ARTIFACT_DIR,
) -> pd.DataFrame:
    if context is None:
        context = load_context(artifacts_dir)

    resolved_text_column = None if all_cells else _resolve_column(
        df,
        requested_column=text_column,
        candidate_names=TEXT_COLUMN_CANDIDATES,
    )
    if resolved_text_column is None and not all_cells and len(df.columns) == 1:
        resolved_text_column = df.columns[0]

    id_column = _resolve_column(df, candidate_names=ID_COLUMN_CANDIDATES)
    metadata_columns = {
        field: _resolve_column(df, candidate_names=candidates)
        for field, candidates in METADATA_COLUMN_CANDIDATES.items()
    }

    rows: List[Dict[str, Any]] = []
    if all_cells or resolved_text_column is None:
        metadata_column_values = {column for column in metadata_columns.values() if column is not None}
        if id_column is not None:
            metadata_column_values.add(id_column)
        candidate_columns = [column for column in df.columns if column not in metadata_column_values]
        for row_idx, row in df.reset_index(drop=True).iterrows():
            source_id = _cell_value(row, id_column, None)
            for column in candidate_columns:
                text = _clean_scalar(row.get(column, ""))
                if not text:
                    continue
                prediction = predict_live(
                    text,
                    language=language,
                    url=url,
                    author_hash=author_hash,
                    date=date,
                    english_keywords=english_keywords,
                    context=context,
                    artifacts_dir=artifacts_dir,
                )
                rows.append(_prediction_to_flat_row(prediction, row_idx + 1, str(column), text, source_id=source_id))
    else:
        for row_idx, row in df.reset_index(drop=True).iterrows():
            text = _clean_scalar(row.get(resolved_text_column, ""))
            source_id = _cell_value(row, id_column, None)
            prediction = predict_live(
                text,
                language=_cell_value(row, metadata_columns["language"], language),
                url=_cell_value(row, metadata_columns[URL_COL], url),
                author_hash=_cell_value(row, metadata_columns[AUTHOR_COL], author_hash),
                date=_cell_value(row, metadata_columns[DATE_COL], date),
                english_keywords=_cell_value(row, metadata_columns[KEYWORDS_COL], english_keywords),
                context=context,
                artifacts_dir=artifacts_dir,
            )
            rows.append(_prediction_to_flat_row(prediction, row_idx + 1, str(resolved_text_column), text, source_id=source_id))

    return pd.DataFrame(rows)


def predict_csv_file(
    input_csv: Path,
    output_csv: Optional[Path] = None,
    text_column: Optional[str] = None,
    all_cells: bool = False,
    no_header: bool = False,
    csv_sep: str = ",",
    language: Optional[str] = None,
    url: Optional[str] = None,
    author_hash: Optional[str] = None,
    date: Optional[str] = None,
    english_keywords: Optional[str] = None,
    artifacts_dir: Path = ARTIFACT_DIR,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    artifacts_dir = Path(artifacts_dir)
    input_csv = Path(input_csv)
    if output_csv is None:
        output_csv = artifacts_dir / "live_predictions.csv"
    output_csv = Path(output_csv)

    df = _read_live_inference_csv(input_csv, no_header=no_header, csv_sep=csv_sep)
    predictions = predict_dataframe(
        df,
        text_column=text_column,
        all_cells=all_cells,
        language=language,
        url=url,
        author_hash=author_hash,
        date=date,
        english_keywords=english_keywords,
        artifacts_dir=artifacts_dir,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    submission_predictions = _build_submission_predictions(predictions)
    submission_predictions.to_csv(output_csv, index=False)
    detailed_output_csv = output_csv.with_name(f"{output_csv.stem}_detailed.csv")
    predictions.to_csv(detailed_output_csv, index=False)

    summary = _build_prediction_batch_artifacts(
        predictions=predictions,
        input_csv=input_csv,
        output_csv=output_csv,
        rows_read=len(df),
    )
    summary["submission_output_csv"] = str(output_csv)
    summary["detailed_output_csv"] = str(detailed_output_csv)
    summary["submission_columns"] = ["test_id", "text", "label"]
    summary["label_definition"] = "0-1 manipulative risk score; 0 means more organic, 1 means more manipulative."
    summary_path = output_csv.with_suffix(".summary.json")
    summary["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    return predictions, summary


def smoke_test() -> None:
    toy = pd.DataFrame(
        [
            {
                TEXT_COL: "Local volunteers met for a community clean-up event on Sunday.",
                KEYWORDS_COL: "community, volunteers, cleanup",
                SENTIMENT_COL: 0.2,
                "main_emotion": "joy",
                "primary_theme": "Community",
                "language": "en",
                URL_COL: "reddit.com",
                AUTHOR_COL: "human_1",
                DATE_COL: "2024-11-24T10:00:00Z",
            },
            {
                TEXT_COL: "BUY NOW!!! FREE FREE FREE #deal #promo https://spam.example.com",
                KEYWORDS_COL: "buy, free, promo, deal",
                SENTIMENT_COL: 0.95,
                "main_emotion": "excitement",
                "primary_theme": "Marketing",
                "language": "en",
                URL_COL: "x.com",
                AUTHOR_COL: "bot_1",
                DATE_COL: "2024-11-24T10:01:00Z",
            },
            {
                TEXT_COL: "BUY NOW!!! FREE FREE FREE #deal #promo https://spam.example.com",
                KEYWORDS_COL: "buy, free, promo, deal",
                SENTIMENT_COL: 0.95,
                "main_emotion": "excitement",
                "primary_theme": "Marketing",
                "language": "en",
                URL_COL: "x.com",
                AUTHOR_COL: "bot_2",
                DATE_COL: "2024-11-24T10:02:00Z",
            },
            {
                TEXT_COL: "BUY NOW!!! FREE FREE FREE #deal #promo https://spam.example.com",
                KEYWORDS_COL: "buy, free, promo, deal",
                SENTIMENT_COL: 0.95,
                "main_emotion": "excitement",
                "primary_theme": "Marketing",
                "language": "en",
                URL_COL: "reddit.com",
                AUTHOR_COL: "bot_3",
                DATE_COL: "2024-11-24T10:03:00Z",
            },
            {
                TEXT_COL: "BUY NOW!!! FREE FREE FREE #deal #promo https://spam.example.com",
                KEYWORDS_COL: "buy, free, promo, deal",
                SENTIMENT_COL: 0.95,
                "main_emotion": "excitement",
                "primary_theme": "Marketing",
                "language": "en",
                URL_COL: "bsky.app",
                AUTHOR_COL: "bot_4",
                DATE_COL: "2024-11-24T10:04:00Z",
            },
            {
                TEXT_COL: "Happy birthday!",
                KEYWORDS_COL: "happy birthday, birthday, happy",
                SENTIMENT_COL: 0.4,
                "main_emotion": "joy",
                "primary_theme": "People",
                "language": "en",
                URL_COL: "x.com",
                AUTHOR_COL: "human_2",
                DATE_COL: "2024-11-24T11:00:00Z",
            },
            {
                TEXT_COL: "Happy birthday!",
                KEYWORDS_COL: "happy birthday, birthday, happy",
                SENTIMENT_COL: 0.4,
                "main_emotion": "joy",
                "primary_theme": "People",
                "language": "en",
                URL_COL: "reddit.com",
                AUTHOR_COL: "human_3",
                DATE_COL: "2024-11-24T11:01:00Z",
            },
            {
                TEXT_COL: "Happy birthday!",
                KEYWORDS_COL: "happy birthday, birthday, happy",
                SENTIMENT_COL: 0.4,
                "main_emotion": "joy",
                "primary_theme": "People",
                "language": "en",
                URL_COL: "youtube.com",
                AUTHOR_COL: "human_4",
                DATE_COL: "2024-11-24T11:02:00Z",
            },
            {
                TEXT_COL: "Happy birthday!",
                KEYWORDS_COL: "happy birthday, birthday, happy",
                SENTIMENT_COL: 0.4,
                "main_emotion": "joy",
                "primary_theme": "People",
                "language": "en",
                URL_COL: "bsky.app",
                AUTHOR_COL: "human_5",
                DATE_COL: "2024-11-24T11:03:00Z",
            },
            {
                TEXT_COL: "",
                KEYWORDS_COL: "",
                SENTIMENT_COL: 0.0,
                "main_emotion": "",
                "primary_theme": "",
                "language": "unknown",
                URL_COL: "",
                AUTHOR_COL: "",
                DATE_COL: "",
            },
        ]
    )
    scored, context = score_dataframe(toy)
    assert {"risk_score", "organic_score", "label", "reason_codes"}.issubset(scored.columns)
    assert scored["risk_score"].between(0, 1).all()
    assert scored.iloc[0]["label"] == "Organic"
    assert scored.iloc[1:5]["coordination_risk"].max() > 0.5
    weak_rows = scored[scored["narrative_signature"] == "kw:birthday|happy|happy birthday"]
    assert weak_rows["coordination_risk"].max() == 0.0
    empty_prediction = predict_live("", context=context)
    assert empty_prediction["label"] == "Manipulative"
    assert "EMPTY_TEXT" in empty_prediction["top_reasons"]
    spam_prediction = predict_live(
        "BUY NOW!!! FREE FREE FREE #deal #promo https://spam.example.com",
        english_keywords="buy, free, promo, deal",
        context=context,
    )
    assert spam_prediction["risk_score"] >= 0.65
    normal_prediction = predict_live(
        "The city council published the meeting notes and invited residents to comment.",
        context=context,
    )
    assert normal_prediction["risk_score"] < 0.65
    for short_text in ["yes", "lol", "No"]:
        short_prediction = predict_live(short_text, context=context)
        assert short_prediction["label"] == "Organic"
        assert short_prediction["risk_score"] < REVIEW_THRESHOLD
        assert "SHORT_TEXT_LOW_EVIDENCE" in short_prediction["top_reasons"]
    emoji_prediction = predict_live("🔥🔥🔥", context=context)
    assert emoji_prediction["label"] == "Organic"
    assert emoji_prediction["risk_score"] < REVIEW_THRESHOLD
    assert "SHORT_TEXT_LOW_EVIDENCE" in emoji_prediction["top_reasons"]
    url_only_prediction = predict_live("https://spam.example.com", context=context)
    assert url_only_prediction["label"] == "Organic"
    assert "EMPTY_TEXT" not in url_only_prediction["top_reasons"]
    assert (
        "LINK_HASHTAG_MENTION_DENSE" in url_only_prediction["top_reasons"]
        or "URL_ONLY_OR_LINK_ONLY" in url_only_prediction["top_reasons"]
    )
    repeated_char_prediction = predict_live("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", context=context)
    assert repeated_char_prediction["risk_band"] == "Review"
    assert repeated_char_prediction["label"] == "Organic"
    assert "REPEATED_CHARACTER_RUN" in repeated_char_prediction["top_reasons"]
    turkish_scam_prediction = predict_live(
        "Günde 5000 TL kazanmak ister misiniz? Ücretsiz VIP kripto grubumuza katılmak için profilimdeki linke tıklayın!",
        context=context,
    )
    assert turkish_scam_prediction["label"] == "Manipulative"
    assert "FINANCIAL_OR_CRYPTO_BAIT" in turkish_scam_prediction["top_reasons"]
    phishing_prediction = predict_live(
        "Your account will be suspended in 24 hours due to copyright infringement. Click the link to fill out the appeal form.",
        context=context,
    )
    assert phishing_prediction["label"] == "Manipulative"
    assert "PHISHING_URGENCY_THREAT" in phishing_prediction["top_reasons"]
    multilingual_cases = [
        (
            "Su cuenta sera suspendida en 24 horas. Verifique su billetera ahora.",
            "PHISHING_URGENCY_THREAT",
        ),
        (
            "Sua conta sera suspensa em 24 horas. Verifique sua carteira agora.",
            "PHISHING_URGENCY_THREAT",
        ),
        (
            "Akun Anda ditangguhkan dalam 24 jam. Verifikasi dompet sekarang.",
            "PHISHING_URGENCY_THREAT",
        ),
        (
            "आपका खाता निलंबित होगा 24 घंटे में। वॉलेट सत्यापन अभी पूरा करें।",
            "PHISHING_URGENCY_THREAT",
        ),
        (
            "حسابك سيتم تعليقه خلال 24 ساعة. تحقق من محفظة MetaMask الآن.",
            "PHISHING_URGENCY_THREAT",
        ),
        (
            "Votre compte sera suspendu dans 24 heures. Verification du portefeuille requise.",
            "PHISHING_URGENCY_THREAT",
        ),
        (
            "Ihr Konto wird gesperrt in 24 Stunden. Wallet Verifizierung ist erforderlich.",
            "PHISHING_URGENCY_THREAT",
        ),
        (
            "アカウントが停止されます。24時間以内にウォレット認証を完了してください。",
            "PHISHING_URGENCY_THREAT",
        ),
        (
            "계정이 정지됩니다. 24시간 내에 지갑 인증을 완료하세요.",
            "PHISHING_URGENCY_THREAT",
        ),
        (
            "Ваш аккаунт будет заблокирован через 24 часа. Пройдите проверка кошелька.",
            "PHISHING_URGENCY_THREAT",
        ),
        (
            "Il tuo conto sara sospeso entro 24 ore. Verifica il portafoglio wallet ora.",
            "PHISHING_URGENCY_THREAT",
        ),
        (
            "账号将暂停。请在24小时内完成钱包验证。",
            "PHISHING_URGENCY_THREAT",
        ),
    ]
    for multilingual_text, expected_reason in multilingual_cases:
        multilingual_prediction = predict_live(multilingual_text, context=context)
        assert multilingual_prediction["risk_score"] >= RISK_THRESHOLD, multilingual_prediction
        assert expected_reason in multilingual_prediction["top_reasons"], multilingual_prediction
    batch_predictions = predict_dataframe(
        pd.DataFrame(
            {
                TEXT_COL: [
                    "The city council published the meeting notes and invited residents to comment.",
                    "BUY NOW!!! FREE FREE FREE #deal #promo https://spam.example.com",
                    "yes",
                ]
            }
        ),
        context=context,
    )
    assert len(batch_predictions) == 3
    assert batch_predictions.iloc[0]["label"] == "Organic"
    assert batch_predictions.iloc[1]["label"] == "Manipulative"
    wide_predictions = predict_dataframe(
        pd.DataFrame(
            {
                "cell_a": ["BUY NOW!!! FREE FREE FREE #deal #promo"],
                "cell_b": ["Thanks for sharing the meeting notes."],
            }
        ),
        all_cells=True,
        context=context,
    )
    assert len(wide_predictions) == 2
    assert {"source_row", "source_column", TEXT_COL, "risk_score", "label"}.issubset(wide_predictions.columns)
    no_header_path = Path("/tmp/datathon_smoke_no_header.csv")
    no_header_output = Path("/tmp/datathon_smoke_no_header_predictions.csv")
    no_header_path.write_text(
        "The city council published the meeting notes. \n"
        "BUY NOW!!! FREE FREE FREE #deal #promo https://spam.example.com\n",
        encoding="utf-8",
    )
    no_header_predictions, no_header_summary = predict_csv_file(
        no_header_path,
        output_csv=no_header_output,
        no_header=True,
    )
    assert no_header_summary["texts_scored"] == 2
    assert len(no_header_predictions) == 2
    assert Path(no_header_summary["summary_csv"]).exists()
    assert Path(no_header_summary["summary_markdown"]).exists()
    assert Path(no_header_summary["top_risk_examples_csv"]).exists()
    assert "risk_score_mean" in no_header_summary
    semicolon_path = Path("/tmp/datathon_smoke_semicolon.csv")
    semicolon_output = Path("/tmp/datathon_smoke_semicolon_predictions.csv")
    semicolon_path.write_text(
        "original_text;language\n"
        "The city council published the meeting notes.;en\n"
        "BUY NOW!!! FREE FREE FREE #deal #promo https://spam.example.com;en\n",
        encoding="utf-8",
    )
    semicolon_predictions, semicolon_summary = predict_csv_file(
        semicolon_path,
        output_csv=semicolon_output,
        csv_sep=";",
    )
    assert semicolon_summary["texts_scored"] == 2
    assert len(semicolon_predictions) == 2
    jury_format_path = Path("/tmp/datathon_smoke_jury_format.csv")
    jury_format_output = Path("/tmp/datathon_smoke_jury_format_predictions.csv")
    jury_format_path.write_text(
        "test_id,text\n"
        "case-001,The city council published the meeting notes.\n"
        "case-002,BUY NOW!!! FREE FREE FREE #deal #promo https://spam.example.com\n",
        encoding="utf-8",
    )
    jury_predictions, jury_summary = predict_csv_file(
        jury_format_path,
        output_csv=jury_format_output,
    )
    assert jury_summary["texts_scored"] == 2
    assert "test_id" in jury_predictions.columns
    assert jury_predictions["test_id"].tolist() == ["case-001", "case-002"]
    assert jury_predictions["source_column"].nunique() == 1
    assert jury_predictions["source_column"].iloc[0] == "text"
    jury_submission = pd.read_csv(jury_format_output)
    assert jury_submission.columns.tolist() == ["test_id", "text", "label"]
    assert jury_submission["test_id"].tolist() == ["case-001", "case-002"]
    assert jury_submission["label"].between(0, 1).all()
    assert Path(jury_summary["detailed_output_csv"]).exists()
    print("Smoke tests passed.")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="DataLeague Datathon manipulation scoring pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Score a parquet sample and build artifacts")
    run_parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    run_parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACT_DIR)
    run_parser.add_argument("--sample-size", type=int, default=200_000)
    run_parser.add_argument("--skip-full-author-stats", action="store_true")
    run_parser.add_argument("--no-plots", action="store_true")

    predict_parser = subparsers.add_parser("predict", help="Run live prediction for one text")
    predict_parser.add_argument("text")
    predict_parser.add_argument("--language", default=None)
    predict_parser.add_argument("--url", default=None)
    predict_parser.add_argument("--author-hash", default=None)
    predict_parser.add_argument("--date", default=None)
    predict_parser.add_argument("--english-keywords", default=None)
    predict_parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACT_DIR)

    predict_csv_parser = subparsers.add_parser(
        "predict-csv",
        aliases=["predict_csv", "batch-predict"],
        help="Run live prediction for a CSV file",
    )
    predict_csv_parser.add_argument("input_csv", type=Path)
    predict_csv_parser.add_argument("--output", type=Path, default=None)
    predict_csv_parser.add_argument("--text-column", default=None)
    predict_csv_parser.add_argument("--all-cells", action="store_true")
    predict_csv_parser.add_argument("--no-header", action="store_true")
    predict_csv_parser.add_argument("--sep", default=",", help="CSV delimiter. Use 'auto' to enable delimiter sniffing.")
    predict_csv_parser.add_argument("--language", default=None)
    predict_csv_parser.add_argument("--url", default=None)
    predict_csv_parser.add_argument("--author-hash", default=None)
    predict_csv_parser.add_argument("--date", default=None)
    predict_csv_parser.add_argument("--english-keywords", default=None)
    predict_csv_parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACT_DIR)

    subparsers.add_parser("smoke-test", help="Run lightweight tests without reading parquet")

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "run":
        scored, _, summary = run_pipeline(
            data_path=args.data_path,
            artifacts_dir=args.artifacts_dir,
            sample_size=args.sample_size,
            use_full_author_stats=not args.skip_full_author_stats,
            make_plots=not args.no_plots,
        )
        print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False))
        print(f"Top rows written to: {args.artifacts_dir / 'scored_sample.csv'}")
        print(
            scored.sort_values("risk_score", ascending=False)[["risk_score", "label", "reason_codes"]]
            .head(5)
            .to_string(index=False)
        )
    elif args.command == "predict":
        result = predict_live(
            args.text,
            language=args.language,
            url=args.url,
            author_hash=args.author_hash,
            date=args.date,
            english_keywords=args.english_keywords,
            artifacts_dir=args.artifacts_dir,
        )
        print(json.dumps(_json_safe(result), indent=2, ensure_ascii=False))
    elif args.command in {"predict-csv", "predict_csv", "batch-predict"}:
        predictions, summary = predict_csv_file(
            input_csv=args.input_csv,
            output_csv=args.output,
            text_column=args.text_column,
            all_cells=args.all_cells,
            no_header=args.no_header,
            csv_sep=args.sep,
            language=args.language,
            url=args.url,
            author_hash=args.author_hash,
            date=args.date,
            english_keywords=args.english_keywords,
            artifacts_dir=args.artifacts_dir,
        )
        print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False))
        preview_cols = [
            TEXT_COL,
            "label",
            "risk_band",
            "risk_score",
            "top_reasons",
        ]
        preview_cols = [col for col in preview_cols if col in predictions.columns]
        if preview_cols:
            print(predictions[preview_cols].head(10).to_string(index=False))
    elif args.command == "smoke-test":
        smoke_test()


if __name__ == "__main__":
    main()

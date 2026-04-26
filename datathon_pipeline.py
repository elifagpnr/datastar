from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import unicodedata
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
MODEL_VERSION = "text-calibration-v4-psychological-triggers"

TEXT_COL = "original_text"
KEYWORDS_COL = "english_keywords"
SENTIMENT_COL = "sentiment"
AUTHOR_COL = "author_hash"
DATE_COL = "date"
URL_COL = "url"

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

SCAM_PATTERN_GROUPS = {
    "financial_bait": [
        r"\b(ek gelir|extra income|ayda|month extra income|gunde|daily|gunluk)\b",
        r"\b(kazanmak|kazanmak ister|earn|make|profit|guaranteed profit)\b",
        r"\b(\d[\d.,]*\s*(tl|usd|dolar|dollar|\$))\b",
        r"\b(crypto|kripto|vip crypto|vip kripto|forex|signal|sinyal)\b",
        r"\b(trading bot|trade bot|bot|portfolio|portfoy|piyasa|piyasalar)\b",
        r"\b(100x|x5|gem alert|dogemoon|moon|binance|liquidity locked|doxxed)\b",
    ],
    "phishing_threat": [
        r"\b(account|hesabiniz|hesabin).{0,60}\b(suspend|suspended|kapat|kapatilacaktir|closed)\b",
        r"\b(copyright|telif|ihlali|infringement|appeal|itiraz)\b",
        r"\b(24 hours|24 saat|formu doldur|fill out the form)\b",
    ],
    "engagement_bait": [
        r"\b(giveaway|cekilis|hediye ceki|gift card|follow|takip|begen|like|tag|etiketle)\b",
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
        r"\b(hemen|right now|now|urgent|limited|silinmeden|before it gets deleted|24 saat|24 hours)\b",
        r"\b(acil|cokmeden|son \d+ kisi|immediately|required|before it hits)\b",
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
        r"\b(x5|portfoy|portfolio|piyasalar|telegram grubu|vip telegram|vip)\b",
    ],
    "political_mobilization": [
        r"\b(tag'?e destek|tag e destek|destek versin|rt yap|retweet yap|sesimizi duyur)\b",
        r"\b(linki kopyalayip yayin|linki kopyala|kopyalayip yayin|yayin)\b",
        r"\b(gizli belgeler|belgeler sizdi|skandal|ana akim medya|asla gostermez|yalan haber|susturamazsiniz)\b",
        r"\b(uyan|oyun buyuk|gercekleri gormek|herkes bu tag)\b",
        r"\b(everyone needs to see|vote them out|see this video)\b",
    ],
}

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


def fold_text_for_match(value: Any) -> str:
    text = _clean_scalar(value).casefold().translate(TURKISH_CHAR_MAP)
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


def _pattern_count(text: str, patterns: List[str]) -> int:
    return sum(1 for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE))


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
        features[f"{group_name}_count"] = folded_texts.apply(lambda value, pats=patterns: _pattern_count(value, pats)).astype(float)
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
        | ((features["engagement_bait_count"] > 0) & (cta >= 0.50))
        | ((features["adult_or_leak_bait_count"] > 0) & ((features["engagement_bait_count"] > 0) | (features["urgency_bait_count"] > 0)))
        | ((features["health_miracle_count"] > 0) & (cta >= 0.20))
        | ((features["debt_relief_count"] > 0) & (cta >= 0.10))
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
            "| `risk_funnel.png` | Tüm satırlardan Review + High, High ve güçlü reason destekli High örneklere daralan huniyi gösterir. | Modelin seçici ve kanıt odaklı karar verdiğini pazarlamak için kullanılır. |",
            "| `reason_code_breakdown.png` | High-risk kararlarını açıklayan reason code dağılımını gösterir. | Kararların kara kutu olmadığını, açıklanabilir sinyallere dayandığını gösterir. |",
            "| `psychological_trigger_breakdown.png` | FOMO, aciliyet, kayıptan kaçınma, sosyal kanıt ve otorite taklidi sinyallerinin kapsamını gösterir. | Spam dilini psikolojik manipülasyon tetikleyicileriyle ilişkilendirmek için kullanılır. |",
            "| `live_inference_benchmark.png` | Sentetik canlı inference senaryolarının risk skorlarını eşiklerle birlikte gösterir. | Jürinin gizli metin testine hazır olduğumuzu göstermek için kullanılır. |",
            "| `coordination_confidence_bubble.png` | Coordination cluster'larının zaman penceresi, güven skoru, boyut ve coordination risk ilişkisini gösterir. | Koordineli davranış tespitinin sadece metin değil, zaman ve author yayılımıyla desteklendiğini anlatır. |",
            "| `evidence_quality_summary.png` | High-risk kararların ne kadarının boş olmayan, güçlü reason destekli, içerik/author/coordination destekli olduğunu gösterir. | High-risk karar kalitesini ve savunulabilirliği özetler. |",
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

---

## 6. En Şüpheli Segmentler
{segment_lines}
- Platform-normalize risk tablosu: `artifacts/platform_normalized_risk.csv`

---

## 7. Koordinasyon Cluster Örnekleri
{cluster_lines}
- Somut şüpheli gönderi örnekleri: `artifacts/presentation_examples.csv`
- En riskli author özetleri: `artifacts/top_risk_authors.csv`

---

## 8. Normalize Platform ve Author Sinyalleri
{platform_lines}
{author_lines}

---

## 9. Kanıt Scorecard
{marketing_lines}
- Ek görseller: `risk_funnel.png`, `reason_code_breakdown.png`, `psychological_trigger_breakdown.png`, `evidence_quality_summary.png`.
- Etiket sağlanmadığı için bunlar proxy kanıt metrikleri ve senaryo kontrolleridir; etiketli performans metriği iddiası değildir.

---

## 10. Canlı Demo ve Sınırlar
- Demo fonksiyonu: `predict_live(text, language=None, url=None, author_hash=None, date=None, english_keywords=None)`.
- Dönen alanlar: label, risk score, organic score, top reasons, used features ve psychological triggers.
- Canlı inference hazırlık senaryoları: {live_ready}/{live_total} başarılı.
- Benchmark artifact'i: `artifacts/live_inference_benchmark.csv`.
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
- `risk_funnel.png`: Tüm veriden güçlü reason destekli High örneklere daralan seçim hunisini gösterir.
- `reason_code_breakdown.png`: High-risk kararların hangi açıklanabilir reason code'lara dayandığını gösterir.
- `psychological_trigger_breakdown.png`: FOMO, aciliyet, kayıptan kaçınma, sosyal kanıt ve otorite taklidi sinyallerini gösterir.
- `live_inference_benchmark.png`: Sentetik canlı test senaryolarında modelin risk skorlarını eşiklerle birlikte gösterir.
- `coordination_confidence_bubble.png`: Cluster güveni, zaman penceresi, cluster boyutu ve coordination risk ilişkisini gösterir.
- `evidence_quality_summary.png`: High-risk kararların boş olmayan metin, güçlü reason, içerik/author/coordination desteği gibi kalite boyutlarını özetler.
"""
    path = artifacts_dir / "pitch_deck.md"
    path.write_text(content, encoding="utf-8")
    return path


def save_context(context: Dict[str, pd.DataFrame], artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in context.items():
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(artifacts_dir / f"{name}.csv", index=False)


def load_context(artifacts_dir: Path = ARTIFACT_DIR) -> Dict[str, pd.DataFrame]:
    context: Dict[str, pd.DataFrame] = {}
    for name in ["author_stats", "coordination_stats"]:
        path = artifacts_dir / f"{name}.csv"
        if path.exists():
            context[name] = pd.read_csv(path)
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

    summary: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "sample_size_requested": int(sample_size),
        "risk_threshold": RISK_THRESHOLD,
        "review_threshold": REVIEW_THRESHOLD,
        "plots": plot_paths,
        **table_summary,
        **marketing_summary,
    }
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
    context: Optional[Dict[str, pd.DataFrame]] = None,
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
        "top_reasons": list(dict.fromkeys(reasons))[:8],
        "used_features": used_features,
    }


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
    elif args.command == "smoke-test":
        smoke_test()


if __name__ == "__main__":
    main()

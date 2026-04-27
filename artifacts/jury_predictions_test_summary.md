# Jury CSV Batch Inference Summary

- Input CSV: `jury_inputs/test.csv`
- Submission CSV: `artifacts/jury_predictions_test.csv`
- Detailed CSV: `artifacts/jury_predictions_test_detailed.csv`

## Overview
| metric | value |
| --- | --- |
| Rows read | 8 |
| Texts scored | 8 |
| Average risk score | 0.6975 |
| Median risk score | 0.7200 |
| P90 risk score | 0.7200 |
| Review + High share | 100.00% |
| High share | 100.00% |
| Manipulative share | 100.00% |

## Risk Band Distribution
| risk_band | count | share |
| --- | --- | --- |
| Low | 0 | 0.00% |
| Review | 0 | 0.00% |
| High | 8 | 100.00% |

## Top Reason Codes
| reason_code | count | share |
| --- | --- | --- |
| TEXT_ONLY_METADATA_UNAVAILABLE | 7 | 87.50% |
| CALL_TO_ACTION_LANGUAGE | 6 | 75.00% |
| NLP_TEXT_MODEL_SUPPORT | 4 | 50.00% |
| URGENCY_LANGUAGE | 4 | 50.00% |
| COORDINATED_AMPLIFICATION_LANGUAGE | 4 | 50.00% |
| FINANCIAL_OR_CRYPTO_BAIT | 3 | 37.50% |
| ENGAGEMENT_OR_LINK_BAIT | 3 | 37.50% |
| FOMO_TRIGGER | 2 | 25.00% |
| MARKET_MANIPULATION_BAIT | 2 | 25.00% |
| TRADING_BOT_BAIT | 2 | 25.00% |
| SOCIAL_PROOF_TRIGGER | 2 | 25.00% |
| WALLET_VERIFICATION_BAIT | 1 | 12.50% |

## Psychological Triggers
| trigger | count | share |
| --- | --- | --- |
| urgency_bait | 4 | 50.00% |
| fomo_trigger | 2 | 25.00% |
| social_proof_trigger | 2 | 25.00% |

## Top Risk Examples
| test_id | source_row | risk_score | risk_band | label | top_reasons | psychological_triggers | text_preview |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.72 | High | Manipulative | CALL_TO_ACTION_LANGUAGE;FINANCIAL_OR_CRYPTO_BAIT;ENGAGEMENT_OR_LINK_BAIT;URGENCY_LANGUAGE;FOMO_TRIGGER;TRADING_BOT_BAIT;MARKET_MANIPULATION_BAIT;NLP_TEXT_MODEL_SUPPORT | fomo_trigger;urgency_bait | 🚀💸 100x GEM ALERT! Buy $DOGEMOON now before it hits Binance! Dev is fully doxxed. Liquidity locked! Link in bio! #Crypto #Moon #Airdrop |
| 2 | 2 | 0.72 | High | Manipulative | CALL_TO_ACTION_LANGUAGE;FINANCIAL_OR_CRYPTO_BAIT;ENGAGEMENT_OR_LINK_BAIT;TRADING_BOT_BAIT;NLP_TEXT_MODEL_SUPPORT;TEXT_ONLY_METADATA_UNAVAILABLE |  | Just made $10,000 in one week using this trading bot! 😱 Ask me how in DM. 100% legit. 👇👇 linktr.ee/scamlink |
| 3 | 3 | 0.72 | High | Manipulative | CALL_TO_ACTION_LANGUAGE;FINANCIAL_OR_CRYPTO_BAIT;URGENCY_LANGUAGE;FOMO_TRIGGER;SOCIAL_PROOF_TRIGGER;MARKET_MANIPULATION_BAIT;TEXT_ONLY_METADATA_UNAVAILABLE | fomo_trigger;urgency_bait;social_proof_trigger | ACİL! Piyasalar çökmeden önce portföyünü x5 yapmak isteyenler profilimdeki VIP Telegram grubuna gelsin. Kontenjan son 5 kişi! 🚨📉 |
| 4 | 4 | 0.72 | High | Manipulative | CALL_TO_ACTION_LANGUAGE;URGENCY_LANGUAGE;WALLET_VERIFICATION_BAIT;NLP_TEXT_MODEL_SUPPORT;TEXT_ONLY_METADATA_UNAVAILABLE | urgency_bait | Wallet verification required. Click here to verify your MetaMask account and claim your 500 USDT bonus immediately. |
| 5 | 5 | 0.72 | High | Manipulative | CALL_TO_ACTION_LANGUAGE;ENGAGEMENT_OR_LINK_BAIT;SOCIAL_PROOF_TRIGGER;COORDINATED_AMPLIFICATION_LANGUAGE;TEXT_ONLY_METADATA_UNAVAILABLE | social_proof_trigger | Bu gidişata dur demenin vakti geldi de geçiyor!!! Gerçekleri görmek için uyanın artık, oyun büyük. Herkes bu tag'e destek versin! #GerçeklerOrtada |
| 6 | 6 | 0.66 | High | Manipulative | CALL_TO_ACTION_LANGUAGE;COORDINATED_AMPLIFICATION_LANGUAGE;TEXT_ONLY_METADATA_UNAVAILABLE |  | Hükümetin/Muhalefetin bu skandalını ana akım medya asla göstermez! Gizli belgeler sızdı, linki kopyalayıp yayın! 😡🔥 |
| 7 | 7 | 0.66 | High | Manipulative | URGENCY_LANGUAGE;COORDINATED_AMPLIFICATION_LANGUAGE;NLP_TEXT_MODEL_SUPPORT;TEXT_ONLY_METADATA_UNAVAILABLE | urgency_bait | Absolutely disgusting behavior from the mayor today. Everyone needs to see this video right now! Vote them out! #Election2026 |
| 8 | 8 | 0.66 | High | Manipulative | COORDINATED_AMPLIFICATION_LANGUAGE;TEXT_ONLY_METADATA_UNAVAILABLE |  | Yalan haberlere inanmayın, asıl mağdur edilen biziz. Bizi susturamazsınız, sesimizi duyurana kadar RT yapın! |

## Interpretation Note
These are batch-level inference diagnostics, not ground-truth accuracy metrics. The CSV has no verified labels, so the outputs summarize model selectivity, explanation coverage, and review workload.

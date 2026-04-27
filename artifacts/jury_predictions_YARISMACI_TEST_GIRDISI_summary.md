# Jury CSV Batch Inference Summary

- Input CSV: `jury_inputs/YARISMACI_TEST_GIRDISI.csv`
- Submission CSV: `artifacts/jury_predictions_YARISMACI_TEST_GIRDISI.csv`
- Detailed CSV: `artifacts/jury_predictions_YARISMACI_TEST_GIRDISI_detailed.csv`

## Overview
| metric | value |
| --- | --- |
| Rows read | 443 |
| Texts scored | 443 |
| Average risk score | 0.1863 |
| Median risk score | 0.1397 |
| P90 risk score | 0.5000 |
| Review + High share | 13.54% |
| High share | 8.13% |
| Manipulative share | 8.13% |

## Risk Band Distribution
| risk_band | count | share |
| --- | --- | --- |
| Low | 383 | 86.46% |
| Review | 24 | 5.42% |
| High | 36 | 8.13% |

## Top Reason Codes
| reason_code | count | share |
| --- | --- | --- |
| TEXT_ONLY_METADATA_UNAVAILABLE | 443 | 100.00% |
| LOW_CONTENT_RISK | 266 | 60.05% |
| LINK_HASHTAG_MENTION_DENSE | 98 | 22.12% |
| CALL_TO_ACTION_LANGUAGE | 28 | 6.32% |
| ENGAGEMENT_OR_LINK_BAIT | 23 | 5.19% |
| CREDENTIAL_OR_PAYMENT_DATA_PHISHING | 13 | 2.93% |
| WALLET_VERIFICATION_BAIT | 12 | 2.71% |
| AUTHORITY_IMPERSONATION | 12 | 2.71% |
| FINANCIAL_OR_CRYPTO_BAIT | 12 | 2.71% |
| NLP_TEXT_MODEL_SUPPORT | 10 | 2.26% |
| PHISHING_URGENCY_THREAT | 6 | 1.35% |
| SOCIAL_PROOF_TRIGGER | 4 | 0.90% |

## Psychological Triggers
| trigger | count | share |
| --- | --- | --- |
| authority_impersonation | 12 | 2.71% |
| social_proof_trigger | 4 | 0.90% |
| urgency_bait | 3 | 0.68% |
| fomo_trigger | 1 | 0.23% |

## Top Risk Examples
| test_id | source_row | risk_score | risk_band | label | top_reasons | psychological_triggers | text_preview |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TEST_0000 | 1 | 0.72 | High | Manipulative | CALL_TO_ACTION_LANGUAGE;FINANCIAL_OR_CRYPTO_BAIT;TRADING_BOT_BAIT;MARKET_MANIPULATION_BAIT;NLP_TEXT_MODEL_SUPPORT;TEXT_ONLY_METADATA_UNAVAILABLE |  | I got a likely phishing email from "service@paypal.com". Even the mailed-by and signed-by fields were from https://t.co/a9YJ8xgLKw. No capital "i" to replace the "l".  Contained a link to pay an invoice for $479 in Binan |
| TEST_0440 | 441 | 0.72 | High | Manipulative | WALLET_VERIFICATION_BAIT;TEXT_ONLY_METADATA_UNAVAILABLE |  | @BonitasMedical Hi Bonitas, there was a fraudulent claim in my medical aid. I wrote emails and called the fraudulent line, no one has or willing to assist me. |
| TEST_0414 | 415 | 0.72 | High | Manipulative | CREDENTIAL_OR_PAYMENT_DATA_PHISHING;TEXT_ONLY_METADATA_UNAVAILABLE |  | Receiving reports of a sophisticated phishing scam in which "Coinbase support" calls you about a "suspicious withdrawal request." Next they walk you through "resetting" your password via a malicious phishing site.  Stay  |
| TEST_0179 | 180 | 0.72 | High | Manipulative | PHISHING_URGENCY_THREAT;CREDENTIAL_OR_PAYMENT_DATA_PHISHING;TEXT_ONLY_METADATA_UNAVAILABLE |  | Phishing is a cyber-attack that uses fraudulent emails or websites to trick users into revealing sensitive information, such as login credentials or financial information. Attackers often masquerade as a trusted entity t |
| TEST_0029 | 30 | 0.72 | High | Manipulative | FINANCIAL_OR_CRYPTO_BAIT;HEALTH_MIRACLE_CLAIM;WALLET_VERIFICATION_BAIT;TEXT_ONLY_METADATA_UNAVAILABLE |  | 23 rd February N1 London. Opportunity for a dungeon slave or maid to serve @Deva_De_Vi and I. Clean, tidy, fetch and make tea in between sessions, wait quietly in your cage whilst we work. Deposit required. Email to outl |
| TEST_0398 | 399 | 0.72 | High | Manipulative | CALL_TO_ACTION_LANGUAGE;AUTHORITY_IMPERSONATION;TEXT_ONLY_METADATA_UNAVAILABLE | authority_impersonation | #Phishing is when someone tries to trick you into giving them sensitive information by pretending to be a trusted entity, like a bank or a government agency.   They may send you an #email or #SMS that looks legitimate an |
| TEST_0024 | 25 | 0.72 | High | Manipulative | WALLET_VERIFICATION_BAIT;TEXT_ONLY_METADATA_UNAVAILABLE |  | Replacement VPS  registered user verification email activation is normal |
| TEST_0378 | 379 | 0.72 | High | Manipulative | CALL_TO_ACTION_LANGUAGE;ENGAGEMENT_OR_LINK_BAIT;TEXT_ONLY_METADATA_UNAVAILABLE |  | By end of month, we’ll be able to just input into Airtable   &gt; email copy &gt; plain sales nav link &gt; clients name  &amp; it’ll auto  &gt; scrape LinkedIn &gt; enrich with emails &gt; clean emails &gt; clean catch- |
| TEST_0394 | 395 | 0.72 | High | Manipulative | WALLET_VERIFICATION_BAIT;TEXT_ONLY_METADATA_UNAVAILABLE |  | @BCBSTX Were yall hacked last night? Many employees getting fraudulent Spam claim emails, and your website and App appear to not let anyone in today. |
| TEST_0049 | 50 | 0.72 | High | Manipulative | CREDENTIAL_OR_PAYMENT_DATA_PHISHING;TEXT_ONLY_METADATA_UNAVAILABLE |  | I see the SNP's very own James Bond has opened his data up to Foreign Operatives by putting his govt provided email password into a phishing mail.  He sat on a committee and will have been through at least one security b |

## Interpretation Note
These are batch-level inference diagnostics, not ground-truth accuracy metrics. The CSV has no verified labels, so the outputs summarize model selectivity, explanation coverage, and review workload.

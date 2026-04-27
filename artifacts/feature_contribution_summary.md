# Feature Contribution Summary

This is a formula-based proxy contribution report, not supervised feature importance. There are no ground-truth labels, so the numbers summarize how much each engineered signal contributes to the current risk scoring formula.

## Top Feature Contributions Among High-Risk Rows
| feature | family | mean_high | active_share | type |
| --- | --- | --- | --- | --- |
| High-risk lure floor | Content floor | 0.4904 | 2.67% | rule floor lift |
| Scam/manipulation pattern groups | Content | 0.0994 | 15.98% | weighted additive |
| Token repetition | Content | 0.0294 | 66.02% | weighted additive |
| Link/hashtag/mention density | Content | 0.0285 | 13.62% | weighted additive |
| Extreme sentiment | Content | 0.0262 | 98.98% | weighted additive |
| Call-to-action density | Content | 0.0132 | 8.91% | weighted additive |
| Empty text floor | Content floor | 0.0123 | 0.08% | rule floor lift |
| Punctuation burst | Content | 0.0056 | 24.16% | weighted additive |
| Very long text | Content | 0.0051 | 1.70% | weighted additive |
| Short/tiny text | Content | 0.0050 | 7.95% | weighted additive |
| Repeated character floor | Content floor | 0.0022 | 0.81% | rule floor lift |
| Uppercase intensity | Content | 0.0016 | 3.89% | weighted additive |

## Final Risk Component Contributions
| feature | mean_all | mean_high | active_share |
| --- | --- | --- | --- |
| Content component | 0.0961 | 0.6775 | 99.78% |
| Author component | 0.0200 | 0.0440 | 16.03% |
| Coordination component | 0.0002 | 0.0031 | 0.05% |
| High coordination floor | 0.0000 | 0.0004 | 0.01% |
| Coordination + content floor | 0.0000 | 0.0002 | 0.01% |
| Author + support floor | 0.0000 | 0.0000 | 0.00% |

## Interpretation Note
High values mean the feature frequently and materially contributes to the rule-based risk score. They do not prove causal predictive power against ground-truth labels.

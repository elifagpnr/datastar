# Feature Contribution Summary

This is a formula-based proxy contribution report, not supervised feature importance. There are no ground-truth labels, so the numbers summarize how much each engineered signal contributes to the current risk scoring formula.

## Top Feature Contributions Among High-Risk Rows
| feature | family | mean_high | active_share | type |
| --- | --- | --- | --- | --- |
| High-risk lure floor | Content floor | 0.4810 | 2.16% | rule floor lift |
| Scam/manipulation pattern groups | Content | 0.0904 | 8.38% | weighted additive |
| Link/hashtag/mention density | Content | 0.0337 | 13.62% | weighted additive |
| Token repetition | Content | 0.0264 | 66.02% | weighted additive |
| Extreme sentiment | Content | 0.0263 | 98.98% | weighted additive |
| Call-to-action density | Content | 0.0156 | 9.59% | weighted additive |
| Empty text floor | Content floor | 0.0152 | 0.08% | rule floor lift |
| Short/tiny text | Content | 0.0062 | 7.95% | weighted additive |
| Punctuation burst | Content | 0.0061 | 24.16% | weighted additive |
| Very long text | Content | 0.0039 | 1.70% | weighted additive |
| Repeated character floor | Content floor | 0.0027 | 0.81% | rule floor lift |
| Review lure floor | Content floor | 0.0026 | 1.00% | rule floor lift |

## Final Risk Component Contributions
| feature | mean_all | mean_high | active_share |
| --- | --- | --- | --- |
| Content component | 0.0875 | 0.6693 | 99.77% |
| Author component | 0.0200 | 0.0525 | 16.03% |
| Coordination component | 0.0002 | 0.0038 | 0.05% |
| High coordination floor | 0.0000 | 0.0005 | 0.01% |
| Coordination + content floor | 0.0000 | 0.0002 | 0.01% |
| Author + support floor | 0.0000 | 0.0000 | 0.00% |

## Interpretation Note
High values mean the feature frequently and materially contributes to the rule-based risk score. They do not prove causal predictive power against ground-truth labels.

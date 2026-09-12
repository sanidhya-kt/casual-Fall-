# Paper vs Implementation

## Paper Table 4 reference

| Dataset | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| SisFall | 97.73 | 96.85 | 96.85 | 96.73 |
| KFall | 98.87 | 98.69 | 98.66 | 98.72 |

## Previous repository run

| Dataset | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| SisFall | 91.24 | 87.47 | 91.11 | 89.25 |
| KFall | 94.29 | 92.63 | 95.03 | 93.82 |

These values were selected using the former test-set-best-F1 procedure and are not directly comparable to a clean validation-selected final test evaluation. New results must be written to `reports/runs/` and `reports/experiment_results.csv` with split mode, seed, normalization, lag, validation epoch, and confusion matrix.

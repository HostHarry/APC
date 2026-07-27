# MMMU boxed-first offline rescore

Rules: last `\boxed{}` (incl. unclosed) → answer-is → `(A)` → bare letter → option text; unparseable = wrong (no random).

| Run | Split | n | Official all | Boxed all | Δ | Official MCQ | Boxed MCQ |
|---|---|---:|---:|---:|---:|---:|---:|
| swd | dev | 150 | 36.00% | 34.67% | -1.33 | 38.30% | 36.88% |
| swd | val | 900 | 39.67% | 33.33% | -6.34 | 41.68% | 34.95% |
| swd | weighted | 1050 | 39.14% | 33.52% | -5.62 | 41.19% | 35.22% |
| psp | dev | 150 | 40.67% | 36.00% | -4.67 | 43.26% | 38.30% |
| psp | val | 900 | 38.67% | 35.89% | -2.78 | 40.61% | 37.66% |
| psp | weighted | 1050 | 38.95% | 35.90% | -3.05 | 40.99% | 37.75% |
| psp_vrg | dev | 150 | 38.00% | 36.67% | -1.33 | 40.43% | 39.01% |
| psp_vrg | val | 900 | 39.78% | 35.67% | -4.11 | 41.68% | 37.31% |
| psp_vrg | weighted | 1050 | 39.52% | 35.81% | -3.71 | 41.50% | 37.55% |

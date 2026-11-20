## TARA test — 500 images (location n=471, date n=432)

| run | date_EM@1 | date_E-F1 | Delta | GREAT_date | loc_EM@1 | loc_E-F1 | CO-delta | GREAT_loc | GREAT | MCQA_ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| retrieval_bi | 3.7 | 43.7 | 43.6 | 59.6 | 22.9 | 52.4 | 23.7 | 45.3 | 52.5 |  |
| retrieval_ce | 4.2 | 43.2 | 43.5 | 60.2 | 25.5 | 56.8 | 26.5 | 49.3 | 54.7 |  |
| retrieval_ce_cluster | 3.5 | 43.6 | 40.9 | 59.6 | 25.5 | 56.8 | 26.5 | 49.3 | 54.4 |  |
| zs_noev | 1.2 | 40.9 | 38.7 | 59.0 | 24.6 | 50.0 | 26.6 | 43.3 | 51.1 |  |
| zs_qa | 3.5 | 46.5 | 44.1 | 57.6 | 31.4 | 61.4 | 33.2 | 56.7 | 57.2 |  |
| zs_mcqa | 3.5 | 42.2 | 41.1 | 57.0 | 25.1 | 57.0 | 27.0 | 51.0 | 54.0 | 23.6 |
| lora_mixed_mcqa | 5.6 | 47.2 | 49.5 | 64.4 | 44.8 | 64.9 | 46.0 | 59.4 | 61.9 | 18.4 |
| lora_mixed_qa | 4.6 | 46.5 | 48.4 | 64.9 | 44.2 | 64.5 | 45.4 | 60.1 | 62.5 |  |
| lora_qaonly_qa | 4.4 | 46.1 | 47.4 | 63.8 | 44.2 | 65.4 | 45.7 | 61.7 | 62.7 |  |
| zs7b_qa | 3.9 | 44.4 | 45.4 | 61.1 | 27.6 | 60.7 | 29.2 | 54.5 | 57.8 |  |

### Flips vs `zs_noev` (EM@1)

| run | loc correct→wrong | loc wrong→correct | date correct→wrong | date wrong→correct |
|---|---|---|---|---|
| retrieval_bi | 68 | 60 | 3 | 14 |
| retrieval_ce | 57 | 61 | 3 | 16 |
| retrieval_ce_cluster | 57 | 61 | 3 | 13 |
| zs_qa | 29 | 61 | 2 | 12 |
| zs_mcqa | 49 | 51 | 4 | 14 |
| lora_mixed_mcqa | 10 | 105 | 2 | 21 |
| lora_mixed_qa | 7 | 99 | 2 | 17 |
| lora_qaonly_qa | 4 | 96 | 2 | 16 |
| zs7b_qa | 42 | 56 | 4 | 16 |

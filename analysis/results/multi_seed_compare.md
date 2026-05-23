# Multi-seed model comparison
Seeds: [42, 1337, 2024, 7, 12345]  |  epochs=30, patience=8, seq_len=60
Short test: 2019-10-14..2019-10-28  (n=11)
Long test : 2019-10-14..end  (n=1622)

## Close MAE (mean +/- std across seeds) — short window
| Model | Open | High | Low | Close | Volume |
|---|---|---|---|---|---|
| Fusion CNN-LSTM | 2.167 ± 1.266 | 3.863 ± 2.339 | 3.860 ± 1.683 | 2.455 ± 2.622 | 18308320.200 ± 5407908.840 |
| Cascade CNN-LSTM | 14.119 ± 2.828 | 11.870 ± 4.684 | 11.340 ± 3.472 | 11.147 ± 2.475 | 46378067.600 ± 9103614.240 |
| LSTM-128 | 2.697 ± 0.625 | 2.384 ± 0.517 | 2.109 ± 0.980 | 2.300 ± 0.984 | 19439892.200 ± 8473801.487 |
| Persistence | 1.397 ± 0.000 | 1.266 ± 0.000 | 1.070 ± 0.000 | 1.202 ± 0.000 | 13779418.182 ± 0.000 |

## Close MAE (mean +/- std across seeds) — long window
| Model | Open | High | Low | Close | Volume |
|---|---|---|---|---|---|
| Fusion CNN-LSTM | 20.839 ± 8.187 | 24.307 ± 10.487 | 24.641 ± 7.725 | 22.108 ± 12.808 | 49558738.400 ± 33237943.893 |
| Cascade CNN-LSTM | 85.426 ± 10.265 | 78.185 ± 8.003 | 72.956 ± 11.754 | 78.994 ± 19.992 | 79210395.200 ± 38465807.912 |
| LSTM-128 | 25.768 ± 4.817 | 24.943 ± 5.000 | 21.424 ± 8.773 | 23.629 ± 7.820 | 31466954.000 ± 16220180.386 |
| Persistence | 3.589 ± 0.000 | 2.872 ± 0.000 | 3.334 ± 0.000 | 3.600 ± 0.000 | 18371538.656 ± 0.000 |

## Stats tests on long-window Close errors (seed-averaged forecasts)
- **Fusion vs LSTM-128**: mean|e_A|=20.935, mean|e_B|=23.474, DM(HLN)=-14.007 (p=0), Wilcoxon p=0.00632
- **Fusion vs Cascade**: mean|e_A|=20.935, mean|e_B|=78.960, DM(HLN)=-28.626 (p=0), Wilcoxon p=5.28e-255
- **Fusion vs Persistence**: mean|e_A|=20.935, mean|e_B|=5.075, DM(HLN)=25.872 (p=0), Wilcoxon p=3.58e-201
- **LSTM-128 vs Persistence**: mean|e_A|=23.474, mean|e_B|=5.075, DM(HLN)=21.445 (p=0), Wilcoxon p=1.53e-184

## Chow tests on SPY log-returns (AR(1))
| Break year | F | p-value |
|---|---|---|
| 2012-01-01 | 1.859 | 0.1558 |
| 2013-01-01 | 2.324 | 0.09796 |
| 2014-01-01 | 2.255 | 0.105 |
| 2015-01-01 | 2.615 | 0.07324 |
| 2016-01-01 | 4.416 | 0.01211 |
| 2017-01-01 | 4.439 | 0.01184 |
| 2018-01-01 | 4.202 | 0.015 |

# SOAP-H tuned aux + lower V LR

Relative to `20260518_soaph`, this uses tuned aux Adam hyperparameters and a
separate lower LR for attention V weights.

Target: mean pass if `(3.28 - mean_loss) * sqrt(7) >= 0.004`.

| seed | loss@3025 | final step | final loss |
| ---: | ---: | ---: | ---: |
| 1 | 3.27751 | 3025 | 3.27751 |
| 2 | 3.27988 | 3025 | 3.27988 |
| 3 | 3.27896 | 3025 | 3.27896 |
| 4 | 3.27918 | 3025 | 3.27918 |
| 5 | 3.27705 | 3025 | 3.27705 |
| 6 | 3.27841 | 3025 | 3.27841 |
| 7 | 3.27736 | 3025 | 3.27736 |

Mean loss: 3.27834.
Range: 3.27705-3.27988.
Significance statistic: 0.004403.
Passes: True.

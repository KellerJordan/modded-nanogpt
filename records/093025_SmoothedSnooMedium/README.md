# Update Smoothing + Snoo

TL;DR: Combining #128 and #129, decreases iters to 5590 (-20 from #129, and p-values are much more robust.)


This PR combines #128 (Snoo optimizer) and #128 (EMA on top of Muon). Both PRs are in some way “smoothing out” the updates: #129 smoothes the Muon update, and #128 applies a lookahead smoothing wrapper to the entire optimizer. Here, we just apply #128 to #129. After combining the two, the total iterations decreases to 5590.

## More detail on method
#129 smooths out the Muon updates:
```
muon_update = NS(EMA(grads))
final_update = EMA(muon_update)
```
Here, unlike in #129, we use a constant ema coefficient of 0.2. 

#128 applies a lookahead step to the updates: run an inner optimizer for K iterations, and treat the parameter displacement as a “gradient” for an inner SGD optimizer. Note that if K=1 and the SGD optimizer does not employ Nesterov momentum, then I think the two are equivalent - with the exception that #128 works on every parameter rather than just the Muon parameters.

Here, we simply use the smoothed Muon updates of #129 as the inner optimizer for #128, in addition to importing some learning rate tuning from #129. 

Overall, the total iterations can be decreased to 5590 (from 5610 in #129 or 5640 in #128). I also was more stringent with the p-value criterion, so that it’s likely there is a bit more “slack” in this submission than in either #128 or #129.


## Baselines (80 runs each)

I have noticed that there is substantial variance in the p-values for these runs, so I ran 80 runs of each baseline, and then created 1000 bootstrap samples of size 40 to compute the fraction of times the p-value was less than 0.01. I’m not a real statistician, but I feel better about this methodology than the one employed in #129 to estimate the probability of seeing a p-value below 0.01.

#129:
```
--- Val Loss Stats ---
mean: 	2.919815
std:  	0.000751
val loss 99% confidence interval: (2.919594 - 2.920037)
val_loss t-test p=0.015461 (small means <2.92)
--- Bootstrap p-value analysis --- (1000 samples of size 40)
Mean p-value: 0.139028
Variance of p-values: 0.029849
Percentage of p-values below 0.01: 21.00%
--- Training Time Stats ---
train time (minutes): mean=23.4811, std=0.1983
train time 99% confidence interval: (23.4227 - 23.5396)
avg ms per iteration: 251.1352. 99%% confidence interval: (250.5097 - 251.7608)
```

#128 (here I use the current configuration with 5640 steps)
```
--- Val Loss Stats ---
mean: 	2.919738
std:  	0.000884
val loss 99% confidence interval: (2.919477 - 2.919999)
val_loss t-test p=0.004818 (small means <2.92)

--- Bootstrap p-value analysis --- (1000 samples of size 40)
Mean p-value: 0.092580
Variance of p-values: 0.018915
Percentage of p-values below 0.01: 32.10%

--- Training Time Stats ---
train time (minutes): mean=23.6421, std=0.1916
train time 99% confidence interval: (23.5856 - 23.6986)
avg ms per iteration: 251.5118. 99%% confidence interval: (250.9105 - 252.1131)
```

So, from this we see that there both of these runs have a reasonable chance of hitting the required p-value in 40 samples. The “mean p-value” for the bootstrap analysis is very high because the mean is disproportionately favoring larger numbers.

## This PR
I ran 160 runs for the new changes in order to have more data, and from these again created 1000 bootstrapped samples of size 40 each to get an idea for the variance in the p-value calculation. Over these samples, we see:

```
--- Val Loss stats over all 160 runs --- 
mean: 	2.919547
std:  	0.000798
val loss 99% confidence interval: (2.919383 - 2.919712)
val_loss t-test p=0.000000 (small means <2.92)

--- Bootstrap p-value analysis (1000 samples of size 40 each) ---
Mean p-value: 0.006984
Max p-value: 0.262882
Variance of p-values: 0.000433
Percentage of p-values below 0.01: 85.40%

--- Training Time Stats ---
train time (minutes): mean=23.4283, std=0.1866
train time 99% confidence interval: (23.3899 - 23.4668)
avg ms per iteration: 251.4670. 99%% confidence interval: (251.0542 - 251.8799)
```


## More Aggressive run with 5580 iterations:

I also checked 120 runs of 5580 iterations. As expected, this still hits the target, but the p-value is a bit less robust.

```
--- Val loss stats over all 120 runs ---
mean: 	2.919583
std:  	0.000897
val loss 99% confidence interval: (2.919368 - 2.919797)
val_loss t-test p=0.000001 (small means <2.92)

--- Bootstrap p-value analysis (1000 samples of size 40 each) ---
Mean p-value: 0.018333
Max p-value: 0.376981
Variance of p-values: 0.001668
Percentage of p-values below 0.01: 67.90%

--- Training time stats ---
train time (minutes): mean=23.4492, std=0.2011
train time 99% confidence interval: (23.4012 - 23.4973)
avg ms per iteration: 252.1423. 99%% confidence interval: (251.6256 - 252.6591)
```


## Ablation

To make sure that the improvement over 128 is not just from the new LR tuning, I turned off the update smoothing, but kept the LR tuning. I also increase the number of number of iterations to 5600, which I guessed would more than make up for any improved time-per-step:
```
--- Val Loss Stats ---
mean: 	2.920357
std:  	0.000802
val loss 99% confidence interval: (2.920120 - 2.920593)
val_loss t-test p=0.999924 (small means <2.92)
--- Bootstrap p-value analysis --- (1000 samples of size 40)
Mean p-value: 0.973404
Max p-value: 1.000000
Variance of p-values: 0.003703
Percentage of p-values below 0.01: 0.00%
--- Training Time Stats ---
train time (minutes): mean=23.5413, std=0.2211
train time 99% confidence interval: (23.4761 - 23.6065)
avg ms per iteration: 252.2285. 99%% confidence interval: (251.5297 - 252.9272)
```
So, it does not seem to hit the target without the smoothing.


I also tried tuning the LR cooldown fraction a bit (both with and without smoothing) as suggested by @YouJiacheng in a comment on #129, but also did not find any improvement from this.




A list of all 120 validation losses:
```
2.919161
2.920945
2.91878
2.91945
2.920088
2.918436
2.919751
2.919509
2.919121
2.920388
2.920208
2.920169
2.920938
2.918948
2.919245
2.919653
2.918682
2.918916
2.919926
2.920458
2.918769
2.918555
2.91991
2.919425
2.922082
2.919508
2.920449
2.919091
2.921161
2.919444
2.920434
2.918194
2.919289
2.919533
2.9209
2.918483
2.919002
2.919399
2.920047
2.920363
2.918821
2.920426
2.920432
2.918828
2.918984
2.918681
2.918769
2.918822
2.919352
2.919853
2.919699
2.919783
2.918965
2.919565
2.918902
2.919225
2.920187
2.919625
2.921371
2.919239
2.919902
2.918071
2.919462
2.918726
2.920078
2.918884
2.919408
2.920146
2.919939
2.920311
2.920426
2.919574
2.919629
2.921047
2.918987
2.918633
2.918057
2.919441
2.920069
2.921082
2.920105
2.920009
2.918286
2.919617
2.920899
2.919312
2.919833
2.918901
2.920027
2.919553
2.918713
2.920759
2.919725
2.91843
2.919194
2.920136
2.919102
2.920179
2.919613
2.919428
2.920121
2.918931
2.918599
2.919027
2.918768
2.920173
2.91906
2.919343
2.921149
2.919538
2.919927
2.919984
2.920188
2.919886
2.918576
2.919965
2.919993
2.919684
2.918075
2.920297
2.920482
2.920536
2.919626
2.919845
2.919099
2.919832
2.918258
2.920294
2.920837
2.918292
2.918897
2.917934
2.919626
2.919178
2.918989
2.919164
2.918687
2.918274
2.918378
2.920714
2.920003
2.919554
2.918437
2.919514
2.920284
2.918734
2.920206
2.919427
2.918294
2.920774
2.918721
2.918992
2.919474
2.920078
2.918853
2.917999
2.919675
2.91946
2.920768
2.920036
```


A list of all 160 timings:
```
1404459
1395516
1417959
1400905
1389809
1416569
1402959
1414998
1397510
1403570
1398782
1425294
1395871
1415545
1396661
1408827
1396578
1413940
1385842
1412084
1410787
1389944
1395654
1392580
1400412
1392653
1404503
1395197
1400111
1397917
1433637
1395602
1415162
1397540
1404055
1403213
1418843
1415625
1406476
1416766
1430944
1396947
1395281
1405680
1415661
1405027
1420947
1403780
1401405
1385218
1397050
1397721
1395464
1400580
1391885
1406463
1404080
1434562
1399548
1397347
1393351
1401256
1425077
1395779
1399501
1405212
1401776
1403159
1386425
1408381
1414500
1406785
1395996
1404508
1418131
1396965
1416503
1418343
1434041
1414912
1409539
1404085
1398737
1401971
1403821
1403490
1417833
1407924
1403386
1414721
1406877
1399508
1386470
1416935
1407935
1397397
1398797
1427337
1397961
1400335
1392212
1403485
1398353
1407128
1396716
1401178
1398133
1402724
1402052
1414994
1420134
1435721
1399607
1432821
1403566
1403252
1397278
1396447
1427209
1417239
1385722
1386063
1405059
1402496
1407869
1399238
1393754
1422199
1403489
1405654
1419264
1396958
1395686
1416636
1418610
1401210
1403842
1404138
1404863
1426769
1408366
1398821
1399002
1399979
1403091
1405100
1424880
1388376
1405475
1416867
1403473
1403733
1419152
1416690
1386661
1402672
1432678
1399397
1419047
1417291
```


# Smoothing Muon Updates (includes PR 124)

This submission adds a small EMA filter to the Muon update:
```
original_muon_update =  NS(EMA(gradients)))
final_update = EMA(original_muon_update)
```
Various notions of update smoothing are common in optimization literature; this submission tests the most minimal version I could think of that can be easily "tuned" to be close to a baseline of no-smoothing.

The EMA weight is rather small, starting at 0.5 and decaying to 0.2 over training.

Other changes to learning rates:
* Decay to 0.01 * peak lr, rather than 0.0.
* Increase muon lr to 0.03.

The motivation behind this lr tuning (besides "I tried it and it was better"):
* Increasing final LR: In initial experiments, update smoothing appeared to hit the baseline at iteration T-X when run on T iterations, where T is the baseline iteration count, but would fail when run with T-X iterations.
* Increasing Muon LR: The smoothing "improves" Muon's updates, so intuitively we can trust them more.

This method decreases the number of iterations to 5610, (80 less than PR124, 20 less than PR128)
 
Overall final improvement (on my machine) is 23.52/23.72 = 99.1% of PR124's time and marginally faster than PR128's time (23.52 vs 23.59). PR128's validation loss was not sufficiently low on my replication (see stats below).
Each step requires on average roughly ~1% more time compared to PR124

Note that the times on my machine are slightly slower than those reported by the previous baselines.
I don't know why this is, but my final reported time is slightly faster than the reported baseline time, so I believe the speed improvement should be robust.

Side notes:
* When I first started investigating this method I based it off of the main branch, then off of PR 119, and finally off of PR 124. The relative improvement off of PR 124 is much smaller: for previous baselines just turning on smoothing without LR changes was enough to see evidence of improvement. In fact, I might not have continued working on this if I had started from PR 124.
Of course, one should expect diminishing returns, but I also suspect that part of this falloff is due to the new value embedding layers not being optimized by Muon so that Muon itself contributes less to the training. It's possible that using some form of this update on all the parameters would be reasonable - indeed, PR 128 seems to do something similar (if one does a lookahead every step rather than every K steps, it reduces to the same method).
Perhaps there is some reasonable way to combine them.
* I experimented briefly with adding a "Nesterov-like" extra EMA step to the new smoothing. It didn't seem clearly helpful, so I removed it. But it's possible that I just didn't hit on the right EMA values.

Simple stats:

I ran 80 trials for baselines and ablations. For the update smoothing change, I run four sets of only 40 runs each to check that the p-value has a reasonable chance of being small after a moderate number of runs.
That said, while the p-value has a good chance of being <0.01, after 40 trials, it's not an extremely high chance. I don't know what the standards should be for making reproducability easy here.

To compensate for this, I tested decreasing the iteration count only to 5630 (so a decrease of 60 iterations from PR124). After 40 trials the 

## Baselines:

PR124:
```
Processed 80 files.
val loss:
mean: 	2.919458
std:  	0.000737
median:	2.91946
min:   	2.917731
max:   	2.921479
val loss 99% confidence interval: (2.919241 - 2.919676)
val_loss t-test p=0.000000 (small means <2.92)
train time (minutes): mean=23.7152, std=0.1746
train time 99% confidence interval: (23.6637 - 23.7667)
``` 

PR128:
```
Processed 80 files.
val loss:
mean: 	2.919946
std:  	0.000839
median:	2.9198864999999996
min:   	2.917602
max:   	2.921847
val loss 99% confidence interval: (2.919698 - 2.920193)
val_loss t-test p=0.282210 (small means <2.92)
train time (minutes): mean=23.5897, std=0.1843
train time 99% confidence interval: (23.5353 - 23.6441)
```
This p-value is pretty high. I'm not sure what's wrong and I haven't investigated. The method itself seems reasonable and somewhat similar to the one proposed here.

## Update smoothing data
with EMA update smoothing (4 replicates of 40 runs each to ensure p-value has reasonable chance of being small):
```
Replicate 1:
Processed 40 files.
val loss:
mean: 	2.919700
std:  	0.000786
median:	2.9196095
min:   	2.918108
max:   	2.921116
val loss 99% confidence interval: (2.919364 - 2.920036)
val_loss t-test p=0.010251 (small means <2.92)
train time (minutes): mean=23.4630, std=0.1956
train time 99% confidence interval: (23.3793 - 23.5466)

Replication 2:
Processed 40 files.
val loss:
mean: 	2.919769
std:  	0.000719
median:	2.9197115
min:   	2.918409
max:   	2.92128
val loss 99% confidence interval: (2.919462 - 2.920076)
val_loss t-test p=0.024538 (small means <2.92)
train time (minutes): mean=23.5620, std=0.1834
train time 99% confidence interval: (23.4835 - 23.6404)

Replication 3:
Processed 40 files.
val loss:
mean: 	2.919879
std:  	0.000684
median:	2.9197715000000004
min:   	2.918745
max:   	2.921742
val loss 99% confidence interval: (2.919587 - 2.920172)
val_loss t-test p=0.135301 (small means <2.92)
train time (minutes): mean=23.5111, std=0.1905
train time 99% confidence interval: (23.4296 - 23.5925)

Replicate 4:
Processed 40 files.
val loss:
mean: 	2.919653
std:  	0.000880
median:	2.9197154999999997
min:   	2.917344
max:   	2.921405
val loss 99% confidence interval: (2.919277 - 2.920030)
val_loss t-test p=0.008553 (small means <2.92)
train time (minutes): mean=23.5281, std=0.1638
train time 99% confidence interval: (23.4580 - 23.5981)
```

So, one replicate has a high p-value (0.135), two runs are very close to 0.01 (0.0103 and 0.0086), and one run is moderate value (0.0245)..
Moreover, looking over all 160 replicates, if we compute the p-value for runs X to X+40 for all 120 values of X, we end up with:
```
Mean p-value: 0.028338
Std  p-value: 0.034810
Fraction of p-values less than 0.01: 0.4050
```
Doing the same experiment with (so runs X to X+80 for 80 possible X values):
```
Mean p-value: 0.003372
Std  p-value: 0.003911
Fraction of p-values less than 0.01: 0.9012
```
These "windowed p-value" stats are of course correlated with each other due to overlapping data, so take with a grain of salt, but the 4 replicates of 40 each with detailed stats above are at least independent.


Full stats over all 160 replicates
```
Processed 160 files.
val loss:
mean: 	2.919750
std:  	0.000768
median:	2.9197075
min:   	2.917344
max:   	2.921742
val loss 99% confidence interval: (2.919592 - 2.919909)
val_loss t-test p=0.000032 (small means <2.92)
train time (minutes): mean=23.5160, std=0.1855
train time 99% confidence interval: (23.4778 - 23.5542)
```


## Increasing iters count to get smaller p-values
Increase the iteration count to 5630, and decreasing muon lr back to 0.025.
```
Processed 40 files.
val loss:
mean: 	2.919379
std:  	0.000693
median:	2.9195105000000003
min:   	2.918197
max:   	2.921039
val loss 99% confidence interval: (2.919083 - 2.919675)
val_loss t-test p=0.000001 (small means <2.92)
train time (minutes): mean=23.5919, std=0.1811
train time 99% confidence interval: (23.5144 - 23.6693)
```
so, a lower mean val loss than PR 124 while still having a faster time. Moreover, to make sure this is good even with 20 runs, let's break this 40 runs into two groups of 20:

```
Replication 1:
Processed 20 files.
val loss:
mean: 	2.919324
std:  	0.000655
median:	2.9193825
min:   	2.918197
max:   	2.9203
val loss 99% confidence interval: (2.918907 - 2.919741)
val_loss t-test p=0.000094 (small means <2.92)
train time (minutes): mean=23.6051, std=0.1298
train time 99% confidence interval: (23.5225 - 23.6877)

Replication 2:
Processed 20 files.
val loss:
mean: 	2.919434
std:  	0.000741
median:	2.9195115
min:   	2.918345
max:   	2.921039
val loss 99% confidence interval: (2.918962 - 2.919905)
val_loss t-test p=0.001446 (small means <2.92)
train time (minutes): mean=23.5786, std=0.2239
train time 99% confidence interval: (23.4361 - 23.7210)
```


## Simple Ablation
increase iters to 5940, remove update smoothing, keep other changes the same:
```
Processed 80 files.
val loss:
mean: 	2.920077
std:  	0.000808
median:	2.9200865
min:   	2.918031
max:   	2.922638
val loss 99% confidence interval: (2.919839 - 2.920316)
val_loss t-test p=0.802983 (small means <2.92)
train time (minutes): mean=23.60, std=0.19
train time 99% confidence interval: (23.55 - 23.66)
```
So, seems a little slower and doesn't hit the baseline. Not necessarily conclusive (better lr tuning might fix it), but at least this is suggestive that smoothing is helpful.


## Pytorch/CUDA info
as copied from output file:
```
Running Python 3.13.5 (main, Jul 23 2025, 00:37:22) [Clang 20.1.4 ]
Running PyTorch 2.8.0+cu128 compiled for CUDA 12.8
Tue Sep 16 02:32:34 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.172.08             Driver Version: 570.172.08     CUDA Version: 12.8 
```
  

Here are all 160 training times (overall mean 1410960.48125 ms)


```
1391391
1408491
1401832
1404752
1392552
1420409
1405047
1412869
1409450
1415498
1414957
1401424
1410718
1422997
1419292
1416086
1419258
1405641
1414497
1421766
1424494
1409541
1393819
1417834
1414885
1402838
1408898
1392595
1418211
1411401
1397595
1407447
1400228
1399698
1413712
1409826
1421124
1418892
1414915
1405831
1400860
1391113
1408995
1394252
1424756
1406710
1389815
1403059
1406396
1401571
1404389
1409773
1398976
1421501
1396972
1388204
1411405
1416186
1419580
1421979
1418319
1409912
1402709
1433182
1413680
1402302
1404565
1410057
1434883
1422305
1410500
1439471
1401498
1431033
1432998
1421552
1412808
1404751
1412865
1403706
1417577
1421942
1395378
1414067
1423624
1422974
1402284
1427130
1405501
1414874
1418051
1440454
1402307
1417316
1413494
1413734
1401694
1418043
1408612
1406333
1420139
1404261
1414312
1410916
1410233
1403200
1429008
1392578
1393487
1409165
1404297
1397671
1418106
1417320
1402624
1405509
1400737
1400298
1432556
1387058
1410781
1419512
1413382
1408942
1407961
1413075
1418502
1417505
1403760
1408104
1433911
1428427
1406158
1391180
1408216
1397702
1410098
1414692
1394623
1389378
1405825
1394569
1422446
1406564
1408059
1411215
1415311
1415047
1403396
1420240
1391145
1430792
1418766
1413581
1418699
1417753
1419534
1430704
1409010
1433949
```


And here are the observed validation losses (nth line in this list corresponds to the nth line in the timing list above)
```
2.919652
2.919344
2.919402
2.920542
2.920727
2.919708
2.921071
2.920085
2.920328
2.919185
2.920703
2.920551
2.918806
2.920843
2.919849
2.920598
2.919405
2.918958
2.921742
2.919007
2.91958
2.917344
2.919342
2.919716
2.918745
2.920624
2.919545
2.918998
2.919271
2.920993
2.919976
2.92045
2.918666
2.91947
2.919101
2.9203
2.919622
2.91893
2.920574
2.918999
2.918302
2.919394
2.918831
2.919326
2.920257
2.920548
2.918899
2.920383
2.920293
2.918931
2.91878
2.919701
2.921197
2.920343
2.920113
2.919629
2.920949
2.920174
2.919398
2.919762
2.918351
2.919457
2.920061
2.919773
2.919091
2.919187
2.919629
2.920961
2.919422
2.919889
2.919575
2.920425
2.920125
2.921204
2.919423
2.92128
2.918409
2.920283
2.921116
2.919538
2.919657
2.92012
2.919803
2.918861
2.919502
2.92021
2.919618
2.91856
2.919367
2.920173
2.919882
2.9194
2.91987
2.920285
2.91977
2.91949
2.919791
2.919801
2.919398
2.918108
2.919847
2.920134
2.920015
2.91959
2.919689
2.921048
2.919629
2.920577
2.920112
2.92013
2.920295
2.920196
2.91835
2.919592
2.919837
2.921405
2.918981
2.920307
2.920855
2.919849
2.918772
2.919276
2.918409
2.919707
2.919387
2.91831
2.920839
2.919292
2.920145
2.918613
2.919679
2.918958
2.920652
2.919317
2.919928
2.919774
2.920213
2.919079
2.919508
2.919373
2.918248
2.918815
2.920206
2.918921
2.920149
2.919315
2.919749
2.921427
2.921272
2.919723
2.919558
2.919604
2.919822
2.920779
2.919902
2.920701
2.918823
2.919916
2.919132
2.919508
```


# Muown + NorMuon + Contra in 2995 Steps

The base is from the [Muown](https://arxiv.org/abs/2605.10797) [PR #288](https://github.com/KellerJordan/modded-nanogpt/pull/288). On top of it we port three mechanisms:

| Mechanism | Author |
|---|---|
| **NorMuon** | [PR #274](https://github.com/KellerJordan/modded-nanogpt/pull/274) by [@kumarkrishna](https://github.com/kumarkrishna)%7C
| **Contra-Muon** | [PR #275](https://github.com/KellerJordan/modded-nanogpt/pull/275) by [@nilin](https://github.com/nilin) |
| **Power-law cooldown LR** | [PR #287](https://github.com/KellerJordan/modded-nanogpt/pull/287) by [@yash-oai](https://github.com/yash-oai) (deleted; mentioned by merged [PR #291](https://github.com/KellerJordan/modded-nanogpt/pull/291)). |

| seed     | step 2990 | step 2995 | step 3000 |
|---------:|----------:|----------:|----------:|
|        0 |   3.27749 |   3.27718 |   3.27690 |
|        1 |   3.27940 |   3.27914 |   3.27885 |
|        2 |   3.27900 |   3.27871 |   3.27838 |
|        3 |   3.28004 |   3.27975 |   3.27945 |
|        4 |   3.28034 |   3.28007 |   3.27973 |
|        5 |   3.27863 |   3.27837 |   3.27804 |
|        6 |   3.27976 |   3.27950 |   3.27919 |
|        7 |   3.27869 |   3.27843 |   3.27814 |
|        8 |   3.27987 |   3.27959 |   3.27931 |
|        9 |   3.27926 |   3.27900 |   3.27862 |
|       10 |   3.27748 |   3.27723 |   3.27692 |
|       11 |   3.27849 |   3.27821 |   3.27795 |
|       12 |   3.27983 |   3.27961 |   3.27930 |
|       13 |   3.27922 |   3.27900 |   3.27866 |
|       14 |   3.28003 |   3.27974 |   3.27943 |
|       15 |   3.27959 |   3.27933 |   3.27907 |
|       16 |   3.27796 |   3.27768 |   3.27741 |
|       17 |   3.27946 |   3.27915 |   3.27886 |
|       18 |   3.28006 |   3.27975 |   3.27940 |
|       19 |   3.27856 |   3.27827 |   3.27800 |
| **mean** | **3.27916** | **3.27889** | **3.27858** |

We pass the margin after **2995** steps with `(3.28 - 3.27889) * math.sqrt(20) = 0.00496 > 0.004`
This is a log produced by running the current version of Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c), as of October 13th 2024.

It was run on a node with 8x H100 HBM3 according to the instructions [here](https://github.com/karpathy/llm.c/discussions/481).
The mean per-step time was 140ms. The total number of training tokens is 10.26B. The final validation loss was **3.2722**.

This is (significantly) better than the quoted result of **3.29** val loss in
[Andrej Karpathy's May 28th GPT-2 replication discussion](https://github.com/karpathy/llm.c/discussions/481#:~:text=By%20the%20end%20of%20the%20optimization%20we%27ll%20get%20to%20about%203.29).
So it appears that there have been some improvements to the training algorithm used by llm.c since then.

Note that the set of examples which llm.c uses for validation appears to be the same as what we do in this repo, i.e., the first `10 * 2**20` tokens of the val set.


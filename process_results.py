import os
import sys

import numpy as np
import scipy.stats


def load_text_files(dir: str):
    files = [os.path.join(dir, f) for f in os.listdir(dir)]
    text_files = [f for f in files if f.endswith('.txt')]
    text = []
    for f in text_files:
        with open(f, 'r') as file:
            text.append(file.readlines())
    return text


if __name__ == '__main__':
    dir = sys.argv[1]

    text_list = load_text_files(dir)
    val_losses, runtimes = [], []
    for text in text_list:
        last_log = text[-2]
        val_loss = float(text[-2][24:24+6])
        runtime = int(text[-2][42:42+6])
        val_losses.append(val_loss)
        runtimes.append(runtime)

    print(f"{np.mean(val_losses) = }")
    print('p=%.4f' % scipy.stats.ttest_1samp(val_losses, 3.28, alternative='less').pvalue)
    print(f"{np.mean(runtimes) = }")

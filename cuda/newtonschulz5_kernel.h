#ifndef NEWTONSCHULZ5_KERNEL_H
#define NEWTONSCHULZ5_KERNEL_H

#include <torch/extension.h>

// CUDA kernel declaration
torch::Tensor newtonschulz5_cuda(
    torch::Tensor G,
    int steps,
    float a = 3.4445f,
    float b = -4.7750f,
    float c = 2.0315f
);

#endif // NEWTONSCHULZ5_KERNEL_H
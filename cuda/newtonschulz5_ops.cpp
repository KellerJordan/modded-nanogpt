#include <torch/extension.h>
#include "newtonschulz5_kernel.h"

// C++ interface
torch::Tensor newtonschulz5(
    torch::Tensor G,
    int64_t steps,
    double a = 3.4445,
    double b = -4.7750,
    double c = 2.0315
) {
    // Input validation
    TORCH_CHECK(G.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(G.ndimension() >= 2, "Input tensor must be at least 2D");
    TORCH_CHECK(G.is_contiguous(), "Input tensor must be contiguous");
    
    // Call CUDA kernel
    return newtonschulz5_cuda(G, steps, static_cast<float>(a), static_cast<float>(b), static_cast<float>(c));
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("newtonschulz5", &newtonschulz5, "Newton-Schulz iteration for matrix orthogonalization (CUDA)",
          py::arg("G"),
          py::arg("steps"),
          py::arg("a") = 3.4445,
          py::arg("b") = -4.7750,
          py::arg("c") = 2.0315);
}
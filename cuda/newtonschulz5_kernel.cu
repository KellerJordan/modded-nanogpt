#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>

#define BLOCK_SIZE 16
#define WARP_SIZE 32
#define MAX_THREADS 1024

// Kernel for matrix-vector multiplication: y = A * x
template <typename scalar_t>
__global__ void matvec_kernel(
    const scalar_t* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols,
    int batch_idx
) {
    extern __shared__ float shared_x[];
    
    const int tid = threadIdx.x;
    const int row = blockIdx.x * blockDim.x + tid;
    const int batch_offset = batch_idx * rows * cols;
    
    // Load x into shared memory
    for (int i = tid; i < cols; i += blockDim.x) {
        shared_x[i] = x[i];
    }
    __syncthreads();
    
    if (row < rows) {
        float sum = 0.0f;
        for (int col = 0; col < cols; col++) {
            sum += static_cast<float>(A[batch_offset + row * cols + col]) * shared_x[col];
        }
        y[row] = sum;
    }
}

// Kernel for vector normalization and norm computation
__global__ void normalize_vector_kernel(
    float* __restrict__ v,
    float* __restrict__ norm,
    int size
) {
    extern __shared__ float shared_sum[];
    
    const int tid = threadIdx.x;
    float local_sum = 0.0f;
    
    // Compute local sum of squares
    for (int i = tid; i < size; i += blockDim.x) {
        float val = v[i];
        local_sum += val * val;
    }
    
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    // Compute norm and normalize
    if (tid == 0) {
        float vec_norm = sqrtf(shared_sum[0]);
        if (norm != nullptr) {
            *norm = vec_norm;
        }
        shared_sum[0] = vec_norm;
    }
    __syncthreads();
    
    float vec_norm = shared_sum[0];
    
    // Normalize vector
    for (int i = tid; i < size; i += blockDim.x) {
        v[i] /= (vec_norm + 1e-7f);
    }
}

// Power iteration method for spectral norm computation
template <typename scalar_t>
__global__ void power_iteration_kernel(
    const scalar_t* __restrict__ X,
    float* __restrict__ spectral_norms,
    float* __restrict__ workspace,  // Size: batch_size * max(rows, cols) * 2
    int batch_size,
    int rows,
    int cols,
    int num_iters = 3  // Number of power iterations
) {
    const int batch_idx = blockIdx.z;
    if (batch_idx >= batch_size) return;
    
    const int max_dim = max(rows, cols);
    float* u = workspace + batch_idx * max_dim * 2;
    float* v = workspace + batch_idx * max_dim * 2 + max_dim;
    
    // Initialize random vector (using threadIdx as seed for determinism)
    if (blockIdx.x == 0 && blockIdx.y == 0) {
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            v[i] = 1.0f / sqrtf(static_cast<float>(cols));
        }
        __syncthreads();
    }
    
    // Power iteration
    for (int iter = 0; iter < num_iters; iter++) {
        // u = X @ v
        if (blockIdx.y == 0 && threadIdx.y == 0) {
            dim3 mv_block(256);
            dim3 mv_grid((rows + 255) / 256);
            matvec_kernel<<<mv_grid, mv_block, cols * sizeof(float)>>>(
                X, v, u, rows, cols, batch_idx
            );
        }
        __syncthreads();
        
        // Normalize u
        if (blockIdx.x == 0 && blockIdx.y == 0) {
            normalize_vector_kernel<<<1, 256, 256 * sizeof(float)>>>(
                u, nullptr, rows
            );
        }
        __syncthreads();
        
        // v = X^T @ u
        if (blockIdx.y == 0 && threadIdx.y == 0) {
            // Transpose matrix-vector multiplication
            for (int col = threadIdx.x; col < cols; col += blockDim.x) {
                float sum = 0.0f;
                for (int row = 0; row < rows; row++) {
                    sum += static_cast<float>(X[batch_idx * rows * cols + row * cols + col]) * u[row];
                }
                v[col] = sum;
            }
        }
        __syncthreads();
        
        // Normalize v and get norm (which is the spectral norm)
        if (blockIdx.x == 0 && blockIdx.y == 0 && iter == num_iters - 1) {
            normalize_vector_kernel<<<1, 256, 256 * sizeof(float)>>>(
                v, &spectral_norms[batch_idx], cols
            );
        } else if (blockIdx.x == 0 && blockIdx.y == 0) {
            normalize_vector_kernel<<<1, 256, 256 * sizeof(float)>>>(
                v, nullptr, cols
            );
        }
        __syncthreads();
    }
}

// Fused kernel for computing A = X @ X.T and B = b * A + c * A @ A
template <typename scalar_t>
__global__ void newtonschulz_fused_iteration_kernel(
    const scalar_t* __restrict__ X_in,
    scalar_t* __restrict__ X_out,
    scalar_t* __restrict__ workspace,  // For intermediate results
    int batch_size,
    int rows,
    int cols,
    float a,
    float b,
    float c,
    bool transposed
) {
    extern __shared__ float shared_mem[];
    
    const int batch_idx = blockIdx.z;
    if (batch_idx >= batch_size) return;
    
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    
    // Determine actual dimensions based on transpose flag
    const int M = transposed ? cols : rows;
    const int N = transposed ? rows : cols;
    const int K = N;  // For X @ X.T
    
    // Pointers to current batch
    const int batch_offset = batch_idx * rows * cols;
    const scalar_t* X = X_in + batch_offset;
    scalar_t* A = workspace + batch_idx * M * M;  // Store A = X @ X.T
    
    // Tile indices
    const int row = block_y * BLOCK_SIZE + tid_y;
    const int col = block_x * BLOCK_SIZE + tid_x;
    
    // Shared memory for tiles
    float* tile_X = shared_mem;
    float* tile_XT = &shared_mem[BLOCK_SIZE * BLOCK_SIZE];
    
    // Compute A = X @ X.T using tiled matrix multiplication
    float acc_A = 0.0f;
    
    for (int k = 0; k < K; k += BLOCK_SIZE) {
        // Load tiles of X and X.T into shared memory
        if (row < M && k + tid_x < K) {
            if (transposed) {
                tile_X[tid_y * BLOCK_SIZE + tid_x] = static_cast<float>(X[tid_x * M + row + k * M]);
            } else {
                tile_X[tid_y * BLOCK_SIZE + tid_x] = static_cast<float>(X[row * N + k + tid_x]);
            }
        } else {
            tile_X[tid_y * BLOCK_SIZE + tid_x] = 0.0f;
        }
        
        if (col < M && k + tid_y < K) {
            if (transposed) {
                tile_XT[tid_y * BLOCK_SIZE + tid_x] = static_cast<float>(X[tid_y * M + col + k * M]);
            } else {
                tile_XT[tid_y * BLOCK_SIZE + tid_x] = static_cast<float>(X[col * N + k + tid_y]);
            }
        } else {
            tile_XT[tid_y * BLOCK_SIZE + tid_x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++) {
            acc_A += tile_X[tid_y * BLOCK_SIZE + i] * tile_XT[i * BLOCK_SIZE + tid_x];
        }
        
        __syncthreads();
    }
    
    // Write A to global memory
    if (row < M && col < M) {
        A[row * M + col] = static_cast<scalar_t>(acc_A);
    }
    
    __syncthreads();
    
    // Now compute B = b * A + c * A @ A
    // This requires another matrix multiplication A @ A
    float acc_AA = 0.0f;
    
    for (int k = 0; k < M; k += BLOCK_SIZE) {
        // Load tiles of A
        if (row < M && k + tid_x < M) {
            tile_X[tid_y * BLOCK_SIZE + tid_x] = static_cast<float>(A[row * M + k + tid_x]);
        } else {
            tile_X[tid_y * BLOCK_SIZE + tid_x] = 0.0f;
        }
        
        if (col < M && k + tid_y < M) {
            tile_XT[tid_y * BLOCK_SIZE + tid_x] = static_cast<float>(A[(k + tid_y) * M + col]);
        } else {
            tile_XT[tid_y * BLOCK_SIZE + tid_x] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++) {
            acc_AA += tile_X[tid_y * BLOCK_SIZE + i] * tile_XT[i * BLOCK_SIZE + tid_x];
        }
        
        __syncthreads();
    }
    
    // Compute B = b * A + c * A @ A
    if (row < M && col < M) {
        float B_val = b * acc_A + c * acc_AA;
        A[row * M + col] = static_cast<scalar_t>(B_val);  // Reuse A buffer for B
    }
    
    __syncthreads();
    
    // Finally compute X_new = a * X + B @ X
    float acc_BX = 0.0f;
    
    for (int k = 0; k < M; k += BLOCK_SIZE) {
        // Load tile of B (stored in A buffer)
        if (row < M && k + tid_x < M) {
            tile_X[tid_y * BLOCK_SIZE + tid_x] = static_cast<float>(A[row * M + k + tid_x]);
        } else {
            tile_X[tid_y * BLOCK_SIZE + tid_x] = 0.0f;
        }
        
        // Load tile of X
        if (k + tid_y < M && col < N) {
            if (transposed) {
                tile_XT[tid_y * BLOCK_SIZE + tid_x] = static_cast<float>(X[col * M + k + tid_y]);
            } else {
                tile_XT[tid_y * BLOCK_SIZE + tid_x] = static_cast<float>(X[(k + tid_y) * N + col]);
            }
        } else {
            tile_XT[tid_y * BLOCK_SIZE + tid_x] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++) {
            acc_BX += tile_X[tid_y * BLOCK_SIZE + i] * tile_XT[i * BLOCK_SIZE + tid_x];
        }
        
        __syncthreads();
    }
    
    // Write final result: X_new = a * X + B @ X
    if (row < M && col < N) {
        float x_val;
        if (transposed) {
            x_val = static_cast<float>(X[col * M + row]);
        } else {
            x_val = static_cast<float>(X[row * N + col]);
        }
        
        float result = a * x_val + acc_BX;
        
        if (transposed) {
            X_out[batch_offset + col * M + row] = static_cast<scalar_t>(result);
        } else {
            X_out[batch_offset + row * N + col] = static_cast<scalar_t>(result);
        }
    }
}

// Kernel to normalize by spectral norm
template <typename scalar_t>
__global__ void normalize_by_norm_kernel(
    scalar_t* __restrict__ X,
    const float* __restrict__ norms,
    int batch_size,
    int size
) {
    const int batch_idx = blockIdx.y;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || tid >= size) return;
    
    const int idx = batch_idx * size + tid;
    X[idx] = static_cast<scalar_t>(static_cast<float>(X[idx]) / (norms[batch_idx] + 1e-7f));
}

// Simple kernel to compute spectral norm using power iteration
template <typename scalar_t>
__global__ void compute_spectral_norm_kernel(
    const scalar_t* __restrict__ X,
    float* __restrict__ spectral_norms,
    float* __restrict__ workspace,
    int batch_size,
    int rows,
    int cols,
    int num_iters = 3
) {
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const int max_dim = max(rows, cols);
    const int batch_offset = batch_idx * rows * cols;
    
    // Workspace for power iteration vectors
    float* u = workspace + batch_idx * max_dim * 2;
    float* v = workspace + batch_idx * max_dim * 2 + max_dim;
    
    // Initialize v with normalized random values
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        v[i] = 1.0f / sqrtf(static_cast<float>(cols));
    }
    __syncthreads();
    
    float sigma = 0.0f;
    
    // Power iteration
    for (int iter = 0; iter < num_iters; iter++) {
        // u = X @ v
        for (int row = threadIdx.x; row < rows; row += blockDim.x) {
            float sum = 0.0f;
            for (int col = 0; col < cols; col++) {
                sum += static_cast<float>(X[batch_offset + row * cols + col]) * v[col];
            }
            u[row] = sum;
        }
        __syncthreads();
        
        // Compute norm of u
        float u_norm_sq = 0.0f;
        for (int i = threadIdx.x; i < rows; i += blockDim.x) {
            u_norm_sq += u[i] * u[i];
        }
        
        // Reduce to get total norm
        __shared__ float shared_norm[256];
        shared_norm[threadIdx.x] = u_norm_sq;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                shared_norm[threadIdx.x] += shared_norm[threadIdx.x + s];
            }
            __syncthreads();
        }
        
        float u_norm = sqrtf(shared_norm[0]);
        
        // Normalize u
        for (int i = threadIdx.x; i < rows; i += blockDim.x) {
            u[i] /= (u_norm + 1e-7f);
        }
        __syncthreads();
        
        // v = X^T @ u
        for (int col = threadIdx.x; col < cols; col += blockDim.x) {
            float sum = 0.0f;
            for (int row = 0; row < rows; row++) {
                sum += static_cast<float>(X[batch_offset + row * cols + col]) * u[row];
            }
            v[col] = sum;
        }
        __syncthreads();
        
        // Compute norm of v (this is the singular value)
        float v_norm_sq = 0.0f;
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            v_norm_sq += v[i] * v[i];
        }
        
        shared_norm[threadIdx.x] = v_norm_sq;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                shared_norm[threadIdx.x] += shared_norm[threadIdx.x + s];
            }
            __syncthreads();
        }
        
        sigma = sqrtf(shared_norm[0]);
        
        // Normalize v
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            v[i] /= (sigma + 1e-7f);
        }
        __syncthreads();
    }
    
    // Write the spectral norm
    if (threadIdx.x == 0) {
        spectral_norms[batch_idx] = sigma;
    }
}

// Main Newton-Schulz iteration function
torch::Tensor newtonschulz5_cuda(
    torch::Tensor G,
    int steps,
    float a,
    float b,
    float c
) {
    // Check input
    TORCH_CHECK(G.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(G.ndimension() >= 2, "Input must be at least 2D");
    TORCH_CHECK(G.is_contiguous(), "Input must be contiguous");
    
    const at::cuda::CUDAGuard device_guard(G.device());
    
    // Get dimensions
    const int64_t batch_dims = G.ndimension() - 2;
    int64_t batch_size = 1;
    for (int i = 0; i < batch_dims; i++) {
        batch_size *= G.size(i);
    }
    const int64_t rows = G.size(-2);
    const int64_t cols = G.size(-1);
    
    // Determine if we need to transpose
    const bool need_transpose = rows > cols;
    const int64_t M = need_transpose ? cols : rows;
    const int64_t N = need_transpose ? rows : cols;
    
    // Create output tensor and workspace
    auto X = G.clone();
    auto options = torch::TensorOptions().dtype(G.dtype()).device(G.device());
    auto workspace = torch::empty({batch_size, M, M}, options);
    auto spectral_norms = torch::empty({batch_size}, options.dtype(torch::kFloat32));
    
    // Allocate workspace for power iteration
    const int max_dim = std::max(rows, cols);
    auto power_iter_workspace = torch::empty({batch_size * max_dim * 2}, options.dtype(torch::kFloat32));
    
    // Flatten batch dimensions
    X = X.view({batch_size, rows, cols});
    
    // Compute spectral norm using power iteration and normalize
    {
        dim3 block(256);
        dim3 grid(batch_size);
        
        // Compute spectral norms using power iteration
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
            G.scalar_type(), "compute_spectral_norm_kernel", [&] {
            compute_spectral_norm_kernel<scalar_t><<<grid, block>>>(
                X.data_ptr<scalar_t>(),
                spectral_norms.data_ptr<float>(),
                power_iter_workspace.data_ptr<float>(),
                batch_size,
                rows,
                cols,
                3  // num_iters for power iteration
            );
        });
        
        // Normalize by spectral norm
        dim3 norm_grid((rows * cols + 255) / 256, batch_size);
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
            G.scalar_type(), "normalize_by_norm_kernel", [&] {
            normalize_by_norm_kernel<scalar_t><<<norm_grid, block>>>(
                X.data_ptr<scalar_t>(),
                spectral_norms.data_ptr<float>(),
                batch_size,
                rows * cols
            );
        });
    }
    
    // Perform Newton-Schulz iterations
    auto X_ping = X;
    auto X_pong = torch::empty_like(X);
    
    for (int iter = 0; iter < steps; iter++) {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (M + BLOCK_SIZE - 1) / BLOCK_SIZE,
            batch_size
        );
        
        size_t shared_size = 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float);
        
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
            G.scalar_type(), "newtonschulz_fused_iteration_kernel", [&] {
            newtonschulz_fused_iteration_kernel<scalar_t><<<grid, block, shared_size>>>(
                X_ping.data_ptr<scalar_t>(),
                X_pong.data_ptr<scalar_t>(),
                workspace.data_ptr<scalar_t>(),
                batch_size,
                rows,
                cols,
                a, b, c,
                need_transpose
            );
        });
        
        // Swap buffers
        std::swap(X_ping, X_pong);
    }
    
    // Reshape back to original dimensions
    auto result = X_ping.view(G.sizes());
    
    // Handle transpose if needed
    if (need_transpose) {
        result = result.transpose(-2, -1);
    }
    
    return result;
}
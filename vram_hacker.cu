#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

// CUDA Kernel: Zero-copy append to KV-cache
// Bypasses torch.cat memory re-allocation by injecting tokens directly into pre-allocated VRAM.
__global__ void kv_cache_append_kernel(
    float* kv_cache,
    const float* new_tokens,
    int current_seq_len,
    int num_new_tokens,
    int hidden_dim,
    int max_seq_len
) {
    int token_idx = blockIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (token_idx < num_new_tokens && dim_idx < hidden_dim) {
        int cache_pos = current_seq_len + token_idx;
        
        // Safety bound check
        if (cache_pos < max_seq_len) {
            kv_cache[cache_pos * hidden_dim + dim_idx] = new_tokens[token_idx * hidden_dim + dim_idx];
        }
    }
}

// C++ Binding function
void append_to_kv_cache(torch::Tensor kv_cache, torch::Tensor new_tokens, torch::Tensor current_pos_tensor) {
    int current_pos = current_pos_tensor.item<int>();
    int num_new_tokens = new_tokens.size(0);
    int hidden_dim = new_tokens.size(1);
    int max_seq_len = kv_cache.size(0);

    // Thread block configuration
    int threads = 256;
    int blocks_x = (hidden_dim + threads - 1) / threads;
    dim3 blocks(blocks_x, num_new_tokens);

    // Launch Kernel
    kv_cache_append_kernel<<<blocks, threads>>>(
        kv_cache.data_ptr<float>(),
        new_tokens.data_ptr<float>(),
        current_pos,
        num_new_tokens,
        hidden_dim,
        max_seq_len
    );

    // Update the position tracker strictly in-place
    current_pos_tensor.index_put_({0}, current_pos + num_new_tokens);
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("append_to_kv_cache", &append_to_kv_cache, "Zero-Copy Direct VRAM Memory Injection for LLMs");
}
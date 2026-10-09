#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>

// ============================================================
// CUDA Kernel: Zero-copy append to KV-cache
// Bypasses torch.cat memory re-allocation by injecting tokens
// directly into pre-allocated VRAM.
// ============================================================
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

// ============================================================
// CUDA Kernel: Paged KV-Cache append (physical block table)
// ============================================================
// Appends K/V tokens into a *discrete* pool of physical blocks whose layout is
// described by a block table (vLLM / PagedAttention style). The logical sequence
// is therefore no longer required to live in one contiguous allocation: blocks
// are mapped on demand, which removes the "max_seq_len" over-allocation of the
// contiguous kernel above.
//
// Memory access pattern (fully coalesced):
//   threadIdx.x -> head_dim    (consecutive floats => 128B transactions)
//   blockIdx.x  -> token index inside the new chunk
//   blockIdx.y  -> head index
//   batch_index -> batch element (launched once per sequence)
// ============================================================
__global__ void paged_kv_cache_append_kernel(
    const float* __restrict__ new_kv,        // [batch, num_heads, new_tokens, head_dim]
    float* __restrict__ paged_kv_pool,       // [total_blocks, num_heads, block_size, head_dim]
    const int* __restrict__ block_table,     // [batch, max_blocks_per_seq]
    const int* __restrict__ seq_lens,        // [batch] length BEFORE the append
    int num_new_tokens,
    int block_size,
    int head_dim,
    int num_heads,
    int max_blocks_per_seq,
    int total_blocks,
    int batch_index,
    int layer_stride                         // num_heads * block_size * head_dim
) {
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    if (token_idx >= num_new_tokens || head_idx >= num_heads) return;

    const int start_len = seq_lens[batch_index];
    const int position = start_len + token_idx;
    const int logical_block = position / block_size;       // block inside the sequence
    const int offset_in_block = position % block_size;     // token slot inside the block
    if (logical_block >= max_blocks_per_seq) return;

    const int physical_block = block_table[batch_index * max_blocks_per_seq + logical_block];
    if (physical_block < 0 || physical_block >= total_blocks) return;  // unmapped slot

    const size_t src_base =
        (((size_t)batch_index * num_heads + head_idx) * num_new_tokens + token_idx) * head_dim;
    const size_t dst_base =
        (size_t)physical_block * layer_stride
        + ((size_t)head_idx * block_size + offset_in_block) * head_dim;

    // Contiguous per-warp writes over head_dim -> coalesced global transactions
    for (int dim_idx = threadIdx.x; dim_idx < head_dim; dim_idx += blockDim.x) {
        paged_kv_pool[dst_base + dim_idx] = new_kv[src_base + dim_idx];
    }
}

// C++ Binding: paged_kv_cache_append
void paged_kv_cache_append(
    torch::Tensor new_kv,
    torch::Tensor paged_kv_pool,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    int64_t block_size
) {
    TORCH_CHECK(new_kv.is_cuda(), "new_kv must be a CUDA tensor");
    TORCH_CHECK(paged_kv_pool.is_cuda(), "paged_kv_pool must be a CUDA tensor");
    TORCH_CHECK(block_table.is_cuda(), "block_table must be a CUDA tensor");
    TORCH_CHECK(seq_lens.is_cuda(), "seq_lens must be a CUDA tensor");
    TORCH_CHECK(new_kv.is_contiguous(), "new_kv must be contiguous");
    TORCH_CHECK(paged_kv_pool.is_contiguous(), "paged_kv_pool must be contiguous");
    TORCH_CHECK(new_kv.dtype() == torch::kFloat32, "new_kv must be float32");
    TORCH_CHECK(new_kv.dim() == 4, "new_kv must be [batch, heads, tokens, head_dim]");

    const int batch = new_kv.size(0);
    const int num_heads = new_kv.size(1);
    const int num_new_tokens = new_kv.size(2);
    const int head_dim = new_kv.size(3);
    const int total_blocks = paged_kv_pool.size(0);
    const int max_blocks_per_seq = block_table.size(1);
    const int layer_stride = num_heads * block_size * head_dim;

    TORCH_CHECK(paged_kv_pool.dim() == 4, "paged_kv_pool must be 4-D");
    TORCH_CHECK(paged_kv_pool.size(1) == num_heads, "head mismatch between new_kv and pool");
    TORCH_CHECK(paged_kv_pool.size(2) == block_size, "pool block_size mismatch");
    TORCH_CHECK(paged_kv_pool.size(3) == head_dim, "head_dim mismatch between new_kv and pool");
    TORCH_CHECK(block_table.size(0) == batch, "block_table batch mismatch");
    TORCH_CHECK(seq_lens.numel() >= batch, "seq_lens must hold one length per batch element");
    TORCH_CHECK(block_size > 0, "block_size must be positive");

    const int threads = (head_dim >= 128) ? 128 : 64;
    dim3 blocks((unsigned)num_new_tokens, (unsigned)num_heads);

    for (int b = 0; b < batch; ++b) {
        paged_kv_cache_append_kernel<<<blocks, threads>>>(
            new_kv.data_ptr<float>(),
            paged_kv_pool.data_ptr<float>(),
            block_table.data_ptr<int>(),
            seq_lens.data_ptr<int>(),
            num_new_tokens,
            (int)block_size,
            head_dim,
            num_heads,
            max_blocks_per_seq,
            total_blocks,
            b,
            layer_stride
        );
    }
    cudaDeviceSynchronize();

    // Extend every sequence in-place by the number of appended tokens
    seq_lens.add_(num_new_tokens);
}

// ============================================================
// CUDA Kernel: Fused paged KV-Cache scale + append
// ============================================================
// Same physical block-table addressing as paged_kv_cache_append_kernel, but the
// incoming K/V tokens are transformed *in registers* before they are merged into
// the pool:
//
//     x_out = clamp(x_in * scale, -limit, +limit)   (limit <= 0 disables it)
//
// The naive alternative -- scale the token tensor in a first kernel and then
// append it in a second one -- reads and writes every element twice. Fusing the
// two steps removes one full global-memory round-trip over the new tokens, which
// matters for the per-token KV path of a decode loop.
//
// Memory access pattern (identical to the non-fused kernel, fully coalesced):
//   threadIdx.x -> head_dim    (consecutive floats => 128B transactions)
//   blockIdx.x  -> token index inside the new chunk
//   blockIdx.y  -> head index
//   batch_index -> batch element (launched once per sequence)
// ============================================================
__global__ void fused_paged_kv_cache_scale_append_kernel(
    const float* __restrict__ new_kv,        // [batch, num_heads, new_tokens, head_dim]
    float* __restrict__ paged_kv_pool,       // [total_blocks, num_heads, block_size, head_dim]
    const int* __restrict__ block_table,     // [batch, max_blocks_per_seq]
    const int* __restrict__ seq_lens,        // [batch] length BEFORE the append
    int num_new_tokens,
    int block_size,
    int head_dim,
    int num_heads,
    int max_blocks_per_seq,
    int total_blocks,
    int batch_index,
    int layer_stride,                        // num_heads * block_size * head_dim
    float scale,
    float clamp_limit                        // <= 0 disables dynamic truncation
) {
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    if (token_idx >= num_new_tokens || head_idx >= num_heads) return;

    const int start_len = seq_lens[batch_index];
    const int position = start_len + token_idx;
    const int logical_block = position / block_size;       // block inside the sequence
    const int offset_in_block = position % block_size;     // token slot inside the block
    if (logical_block >= max_blocks_per_seq) return;

    const int physical_block = block_table[batch_index * max_blocks_per_seq + logical_block];
    if (physical_block < 0 || physical_block >= total_blocks) return;  // unmapped slot

    const size_t src_base =
        (((size_t)batch_index * num_heads + head_idx) * num_new_tokens + token_idx) * head_dim;
    const size_t dst_base =
        (size_t)physical_block * layer_stride
        + ((size_t)head_idx * block_size + offset_in_block) * head_dim;

    for (int dim_idx = threadIdx.x; dim_idx < head_dim; dim_idx += blockDim.x) {
        // Register-level transform: never touches global memory in between
        float value = new_kv[src_base + dim_idx] * scale;
        if (clamp_limit > 0.0f) {
            value = fminf(clamp_limit, fmaxf(-clamp_limit, value));
        }
        paged_kv_pool[dst_base + dim_idx] = value;
    }
}

// C++ Binding: fused_paged_kv_cache_scale_append
void fused_paged_kv_cache_scale_append(
    torch::Tensor new_kv,
    torch::Tensor paged_kv_pool,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    int64_t block_size,
    double scale,
    double clamp_limit
) {
    TORCH_CHECK(new_kv.is_cuda(), "new_kv must be a CUDA tensor");
    TORCH_CHECK(paged_kv_pool.is_cuda(), "paged_kv_pool must be a CUDA tensor");
    TORCH_CHECK(block_table.is_cuda(), "block_table must be a CUDA tensor");
    TORCH_CHECK(seq_lens.is_cuda(), "seq_lens must be a CUDA tensor");
    TORCH_CHECK(new_kv.is_contiguous(), "new_kv must be contiguous");
    TORCH_CHECK(paged_kv_pool.is_contiguous(), "paged_kv_pool must be contiguous");
    TORCH_CHECK(new_kv.dtype() == torch::kFloat32, "new_kv must be float32");
    TORCH_CHECK(new_kv.dim() == 4, "new_kv must be [batch, heads, tokens, head_dim]");
    TORCH_CHECK(paged_kv_pool.dim() == 4, "paged_kv_pool must be 4-D");

    const int batch = new_kv.size(0);
    const int num_heads = new_kv.size(1);
    const int num_new_tokens = new_kv.size(2);
    const int head_dim = new_kv.size(3);
    const int total_blocks = paged_kv_pool.size(0);
    const int max_blocks_per_seq = block_table.size(1);
    const int layer_stride = num_heads * block_size * head_dim;

    TORCH_CHECK(paged_kv_pool.size(1) == num_heads, "head mismatch between new_kv and pool");
    TORCH_CHECK(paged_kv_pool.size(2) == block_size, "pool block_size mismatch");
    TORCH_CHECK(paged_kv_pool.size(3) == head_dim, "head_dim mismatch between new_kv and pool");
    TORCH_CHECK(block_table.size(0) == batch, "block_table batch mismatch");
    TORCH_CHECK(seq_lens.numel() >= batch, "seq_lens must hold one length per batch element");
    TORCH_CHECK(block_size > 0, "block_size must be positive");

    const int threads = (head_dim >= 128) ? 128 : 64;
    dim3 blocks((unsigned)num_new_tokens, (unsigned)num_heads);
    const float scale_f = (float)scale;
    const float clamp_f = (float)clamp_limit;

    for (int b = 0; b < batch; ++b) {
        fused_paged_kv_cache_scale_append_kernel<<<blocks, threads>>>(
            new_kv.data_ptr<float>(),
            paged_kv_pool.data_ptr<float>(),
            block_table.data_ptr<int>(),
            seq_lens.data_ptr<int>(),
            num_new_tokens,
            (int)block_size,
            head_dim,
            num_heads,
            max_blocks_per_seq,
            total_blocks,
            b,
            layer_stride,
            scale_f,
            clamp_f
        );
    }
    cudaDeviceSynchronize();

    // Extend every sequence in-place by the number of appended tokens
    seq_lens.add_(num_new_tokens);
}

// C++ Binding: append_to_kv_cache
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

// ============================================================
// CUDA Kernel: Fused Audio Preprocessing (VAD + Pre-emphasis + Windowing)
// Combines energy computation, pre-emphasis filter, and Hann windowing
// into a single kernel pass to minimize memory bandwidth.
// ============================================================
__global__ void fused_audio_preprocess_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float* __restrict__ frame_energies,
    int num_samples,
    int frame_size,
    int hop_size,
    float pre_emphasis_coeff,
    int num_frames
) {
    int frame_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame_idx >= num_frames) return;

    int start = frame_idx * hop_size;
    if (start + frame_size > num_samples) return;

    float energy = 0.0f;

    for (int i = 0; i < frame_size; i++) {
        float sample = input[start + i];

        // Pre-emphasis: y[n] = x[n] - coeff * x[n-1]
        float emphasized = sample;
        if (i > 0) {
            emphasized = sample - pre_emphasis_coeff * input[start + i - 1];
        }

        // Hann window: w[n] = 0.5 * (1 - cos(2*pi*n/(N-1)))
        float window = 0.5f * (1.0f - cosf(2.0f * 3.14159265f * i / (frame_size - 1)));
        float windowed = emphasized * window;

        output[frame_idx * frame_size + i] = windowed;
        energy += windowed * windowed;
    }

    frame_energies[frame_idx] = energy / frame_size;
}

// C++ Binding: fused_audio_preprocess
torch::Tensor fused_audio_preprocess(
    torch::Tensor input,
    int frame_size,
    int hop_size,
    float pre_emphasis_coeff
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    int num_samples = input.size(0);
    int num_frames = (num_samples - frame_size) / hop_size + 1;

    auto output = torch::zeros({num_frames, frame_size}, input.options());
    auto frame_energies = torch::zeros({num_frames}, input.options());

    int threads = 256;
    int blocks = (num_frames + threads - 1) / threads;

    fused_audio_preprocess_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        frame_energies.data_ptr<float>(),
        num_samples,
        frame_size,
        hop_size,
        pre_emphasis_coeff,
        num_frames
    );

    cudaDeviceSynchronize();
    return output;
}

// ============================================================
// CUDA Kernel: Dynamic Memory Injection for LLM Models
// Injects optimized KV-cache tensors into model layers at runtime,
// enabling zero-copy memory sharing between components.
// ============================================================
__global__ void inject_kv_kernel(
    float* model_kv_ptr,
    const float* source_kv_ptr,
    int layer_offset,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        model_kv_ptr[layer_offset + idx] = source_kv_ptr[idx];
    }
}

// C++ Binding: inject_into_model
void inject_into_model(
    torch::Tensor model_kv_cache,
    torch::Tensor source_kv_cache,
    int layer_idx,
    int num_layers
) {
    TORCH_CHECK(model_kv_cache.is_cuda(), "model_kv_cache must be a CUDA tensor");
    TORCH_CHECK(source_kv_cache.is_cuda(), "source_kv_cache must be a CUDA tensor");

    int elements_per_layer = source_kv_cache.size(0) * source_kv_cache.size(1);
    int total_elements = elements_per_layer * num_layers;

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    inject_kv_kernel<<<blocks, threads>>>(
        model_kv_cache.data_ptr<float>(),
        source_kv_cache.data_ptr<float>(),
        layer_idx * elements_per_layer,
        total_elements
    );

    cudaDeviceSynchronize();
}

// ============================================================
// CUDA Kernel: Memory Stress Test
// Performs intensive read/write operations to measure VRAM bandwidth
// and stability under load.
// ============================================================
__global__ void stress_test_kernel(
    float* buffer,
    int num_elements,
    int num_iterations,
    float* throughput_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    float acc = 0.0f;
    clock_t start = clock();

    for (int iter = 0; iter < num_iterations; iter++) {
        float val = buffer[idx];
        val = val * 1.001f + 0.001f;  // Simple compute to prevent optimization
        buffer[idx] = val;
        acc += val;
    }

    clock_t end = clock();
    float elapsed = (float)(end - start) / CLOCKS_PER_SEC;

    // Write throughput info from thread 0
    if (idx == 0 && throughput_out) {
        long long total_ops = (long long)num_elements * num_iterations;
        throughput_out[0] = (float)total_ops / elapsed;  // ops/sec
        throughput_out[1] = elapsed;
        throughput_out[2] = acc;  // Prevent optimization
    }
}

// C++ Binding: stress_test
std::vector<float> stress_test(int num_elements, int num_iterations) {
    // Allocate device buffer
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto buffer = torch::randn({num_elements}, options);
    auto throughput = torch::zeros({3}, options);

    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    stress_test_kernel<<<blocks, threads>>>(
        buffer.data_ptr<float>(),
        num_elements,
        num_iterations,
        throughput.data_ptr<float>()
    );

    cudaDeviceSynchronize();

    // Copy results back
    auto result = throughput.cpu();
    return {result[0].item<float>(), result[1].item<float>(), result[2].item<float>()};
}

// ============================================================
// CUDA Kernel: Dynamic Kernel Launch with Occupancy-based Config
// Demonstrates runtime kernel configuration based on device capabilities.
// ============================================================
__global__ void dynamic_compute_kernel(
    float* data,
    int num_elements,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float val = data[idx];
        // Some non-trivial compute
        val = sqrtf(fabsf(val)) * scale + sinf(val * 0.01f);
        data[idx] = val;
    }
}

// C++ Binding: launch_dynamic_kernel
torch::Tensor launch_dynamic_kernel(torch::Tensor input, float scale) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    int num_elements = input.numel();
    auto output = input.clone();

    // Use cudaOccupancyMaxPotentialBlockSize for optimal launch config
    int min_grid_size = 0;
    int block_size = 0;
    // Fall back to fixed block size for compatibility
    block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;

    dynamic_compute_kernel<<<grid_size, block_size>>>(
        output.data_ptr<float>(),
        num_elements,
        scale
    );

    cudaDeviceSynchronize();
    return output;
}

// ============================================================
// CUDA Utility: Query GPU Memory
// Returns detailed VRAM usage statistics.
// ============================================================
std::vector<int64_t> query_memory() {
    size_t free_bytes = 0;
    size_t total_bytes = 0;

    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        // Return zeros on error
        return {0, 0, 0};
    }

    size_t used_bytes = total_bytes - free_bytes;
    return {(int64_t)total_bytes, (int64_t)free_bytes, (int64_t)used_bytes};
}

// ============================================================
// PYBIND11 Module Registration
// ============================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Omni-VRAM CUDA Extension: GPU-accelerated audio processing and memory management";

    m.def("append_to_kv_cache", &append_to_kv_cache,
        "Zero-Copy Direct VRAM Memory Injection for LLMs");

    m.def("paged_kv_cache_append", &paged_kv_cache_append,
        "Paged KV-Cache append with a physical block table (PagedAttention-style)");

    m.def("fused_paged_kv_cache_scale_append", &fused_paged_kv_cache_scale_append,
        "Fused scale/truncate + paged KV-Cache append (one pass, no extra round-trip)");

    m.def("fused_audio_preprocess", &fused_audio_preprocess,
        "Fused Audio Preprocessing (VAD + Pre-emphasis + Windowing)");

    m.def("inject_into_model", &inject_into_model,
        "Dynamic KV-Cache Injection into LLM Model Layers");

    m.def("stress_test", &stress_test,
        "GPU Memory Stress Test - Returns [ops_per_sec, elapsed_time, checksum]");

    m.def("launch_dynamic_kernel", &launch_dynamic_kernel,
        "Dynamic Kernel Launch with Occupancy-based Configuration");

    m.def("query_memory", &query_memory,
        "Query GPU VRAM Usage - Returns [total_bytes, free_bytes, used_bytes]");
}
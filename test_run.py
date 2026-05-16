import os
import torch 
import time

# Resolve Windows CUDA DLL directory
cuda_bin_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
if os.path.exists(cuda_bin_path):
    os.add_dll_directory(cuda_bin_path)

import vram_core

def run_benchmark():
    print("[INFO] Omni-VRAM KV-Cache Zero-Copy Benchmark")
    print("-" * 50)

    # Configuration
    hidden_dim = 4096
    max_seq_len = 100_000 
    num_updates = 100
    tokens_per_update = 50 

    print(f"[INFO] Config: hidden_dim={hidden_dim}, max_seq_len={max_seq_len}")
    print(f"[INFO] Simulating {num_updates} updates of {tokens_per_update} tokens each...")
    print("-" * 50)

    # --- Baseline: PyTorch Native (torch.cat) ---
    print("[1] Running PyTorch Baseline (torch.cat)...")
    traditional_cache = torch.zeros((0, hidden_dim), device='cuda', dtype=torch.float32)
    new_voice_feature = torch.randn((tokens_per_update, hidden_dim), device='cuda', dtype=torch.float32)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_updates):
        traditional_cache = torch.cat([traditional_cache, new_voice_feature], dim=0)
    torch.cuda.synchronize()
    native_time = (time.perf_counter() - start) * 1000
    print(f"    -> Baseline Latency: {native_time:.2f} ms")

    # --- Optimized: Omni-VRAM Zero-Copy ---
    print("\n[2] Running Omni-VRAM Optimized (Zero-Copy)...")
    omni_cache = torch.zeros((max_seq_len, hidden_dim), device='cuda', dtype=torch.float32)
    current_pos = torch.tensor([0], device='cuda', dtype=torch.int32)

    # Warmup
    vram_core.append_to_kv_cache(omni_cache, new_voice_feature, current_pos)
    current_pos.fill_(0)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_updates):
        vram_core.append_to_kv_cache(omni_cache, new_voice_feature, current_pos)
    torch.cuda.synchronize()
    omni_time = (time.perf_counter() - start) * 1000

    print(f"    -> Omni-VRAM Latency: {omni_time:.2f} ms")
    print(f"    -> Final sequence length: {current_pos.item()} tokens")

    # --- Results ---
    print("-" * 50)
    if omni_time < native_time:
        speedup = native_time / omni_time
        print(f"[SUCCESS] Speedup achieved: {speedup:.2f}x")
    else:
        print("[WARNING] No speedup observed.")
    print("-" * 50)

if __name__ == "__main__":
    run_benchmark()
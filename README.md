# Fused FlashAttention PoC

This repository demonstrates a **fused** attention kernel versus a **naive** multi-step approach in C++. The code runs on **CPU** by default, even if a GPU runtime is selected in Google Colab. The fused kernel merges three steps:

1. `Q × K^T`  
2. `softmax(scores)`  
3. `scores × V`  

into a **single** pass, reducing intermediate reads/writes. Although performance gains on CPU may be small, this concept can yield significantly larger speedups on specialized AI hardware (like Tenstorrent’s chips) or with GPU-based kernels.

---

## 1. Project Structure

```
.
├── CMakeLists.txt          # CMake build configuration
├── fused_attention.hpp     # Header for fused kernel parameters
├── fused_attention.cpp     # Implementation of the fused kernel
├── naive_attention.hpp     # Header for naive kernel parameters
├── naive_attention.cpp     # Implementation of the naive approach
└── main.cpp                # Benchmark driver (compares naive vs. fused)
```

**Key Files**  
- **`fused_attention.cpp`**: Implements attention (QK^T, softmax, ×V) in one routine.  
- **`naive_attention.cpp`**: Splits the same operations into three distinct steps.  
- **`main.cpp`**: Allocates random Q/K/V data, calls both kernels, and measures performance.

---

## 2. Usage

The project is designed to run in a **CPU** environment, such as a local machine or a Google Colab notebook. In Colab, the “GPU” runtime does **not** automatically provide GPU acceleration for this code, because the code is not written in CUDA or using GPU libraries.

### 2.1. Colab Steps

1. **Clone** or copy the `.cpp`, `.hpp`, and `CMakeLists.txt` files into Colab (e.g., via `%%writefile` cells).  
2. **Install** the required build tools:
   ```bash
   !apt-get update
   !apt-get install -y cmake build-essential
   ```
3. **Build** the project:
   ```bash
   !mkdir build
   %cd build
   !cmake ..
   !make
   ```
4. **Run** the executable:
   ```bash
   !./fused_flash_attention
   ```
5. The console output shows:
   - Execution time for the naive kernel  
   - Execution time for the fused kernel  
   - The computed speedup (`naive_time / fused_time`)  
   - RMSE between naive and fused outputs (reflecting floating-point differences)

---

## 3. Example Results

Using a default setting like `batch=1, heads=8, seq_len=128, d_head=64`, the output might show:

```
Naive: 90.12 ms
Fused: 88.35 ms
Speedup: 1.02x
RMSE: 0.35
```

### 3.1. Increasing Sequence Length

Setting `S=512` or `S=1024` in `main.cpp` can highlight memory-traffic overhead in the naive version. This often leads to a modestly larger speedup, for example 1.1–1.3×, depending on CPU caches and concurrency.

### 3.2. Why the Gain is Small on CPU

- Single-thread CPU code can keep intermediate data in cache, limiting overhead differences between naive and fused  
- No GPU or specialized accelerator instructions are used

---

## 4. Relevance to AI Accelerators

A single fused kernel can be significantly faster on hardware designed for AI workloads, such as **Tenstorrent** chips, because:

- **Less DRAM Traffic**: Multiple steps remain in on-chip buffers rather than saving intermediate data back to memory  
- **Fewer Kernel Launches**: A single fused operation can avoid overhead from multiple kernel calls  
- **Scalable**: Large models (longer sequence lengths) amplify data movement costs, making fusion even more beneficial

---

## 5. Next Steps

- **Increase `S`**: Change `seq_len` in `main.cpp` to 512 or 1024 for a bigger demonstration of memory overhead  
- **Add Parallelism**: Use OpenMP or another threading approach to see if concurrency magnifies the difference  
- **Port to TT-Metal**: Implement the fused logic in Tenstorrent’s TT-Metal environment to evaluate performance on actual AI-optimized hardware  
- **Mixed Precision**: Fusing QK^T, softmax, and multiply-by-V in lower-precision formats (e.g., FP16 or BF16) can further reduce memory usage and yield even larger speedups on specialized hardware

---

## 7. Contact

Feel free to reach out for questions or feedback:

- **LinkedIn**: [mohitraosatya](https://www.linkedin.com/in/mohitraosatya/)  
- **Email**: [saka4331@colorado.edu](mailto:saka4331@colorado.edu)


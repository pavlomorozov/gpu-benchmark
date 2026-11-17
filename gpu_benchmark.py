import time
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# Optional GPU support via CuPy
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

def benchmark_cpu(normalized_windows, window_size, batch_size=5000):
    total_windows = normalized_windows.shape[0]
    start_time = time.perf_counter()

    for batch_start_idx in range(0, total_windows, batch_size):
        batch_end = min(batch_start_idx + batch_size, total_windows)
        window_batch = normalized_windows[batch_start_idx:batch_end]
        correlation_matrix = (window_batch @ normalized_windows.T) / window_size

    elapsed = time.perf_counter() - start_time
    print(f"CPU correlation matrix time: {elapsed:.4f} seconds")

def benchmark_gpu_cupy(normalized_windows, window_size, batch_size=5000):
    if not HAS_CUPY:
        print("CuPy not available. Skipping GPU (cupy) benchmark.")
        return

    # Transfer data to GPU
    cp_windows = cp.asarray(normalized_windows)
    total_windows = cp_windows.shape[0]

    # Warmup
    cp.cuda.Stream.null.synchronize()
    correlation_matrix = (cp_windows[:10] @ cp_windows.T) / window_size
    cp.cuda.Stream.null.synchronize()

    start_time = time.perf_counter()
    for batch_start_idx in range(0, total_windows, batch_size):
        batch_end = min(batch_start_idx + batch_size, total_windows)
        window_batch = cp_windows[batch_start_idx:batch_end]
        correlation_matrix = (window_batch @ cp_windows.T) / window_size

    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start_time
    print(f"GPU (CuPy) correlation matrix time: {elapsed:.4f} seconds")

def main():

    df = pd.read_csv("dataset.csv")
    window_size = 60
    values = df["Data"].values.astype(np.float32)

    # Sliding windows
    windows = sliding_window_view(values, window_shape=window_size)
    window_means = np.nanmean(windows, axis=1, keepdims=True)
    window_stds = np.nanstd(windows, axis=1, keepdims=True)
    normalized_windows = (windows - window_means) / window_stds

    print(f"Total windows: {normalized_windows.shape[0]}, Window size: {window_size}")

    # Benchmark CPU
    benchmark_cpu(normalized_windows, window_size)
    
    # Benchmark GPU (CuPy)
    benchmark_gpu_cupy(normalized_windows, window_size)

if __name__ == "__main__":
    main()

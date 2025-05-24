import numpy as np
import time
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import platform
import psutil
import cpuinfo
import memray
import json
import subprocess
import re
import tempfile
from contextlib import contextmanager, nullcontext
from scipy import fft

from imed.frequency import DCT_by_FFT_ST, DCT_ST, FFT_ST
from imed.legacy import fullMat_ST, sepMat_ST
import imed

imed_version = imed.__version__

# Add the parent directory to the path to import imed functions
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Directory for .bin profiles (set in main)
PROFILE_DIR: str = None  # will be initialized in main

# --- System Info ---
def get_system_info():
    cpu = cpuinfo.get_cpu_info()
    mem = psutil.virtual_memory()
    return {
        "Platform": platform.system(),
        "Platform-Version": platform.version(),
        "Architecture": platform.machine(),
        "CPU": cpu.get("brand_raw", "Unknown"),
        "Physical Cores": psutil.cpu_count(logical=False),
        "Logical CPUs": psutil.cpu_count(logical=True),
        "RAM (GB)": round(mem.total / (1024 ** 3), 2),
        "Python Version": platform.python_version(),
        "IMED version": imed_version,
    }


def get_peak_memory_bytes(profile_file: str) -> int:
    """
    Return the peak memory (in bytes) recorded in a Memray .bin file.
    1) Try `memray stats --json` (Memray ≥1.8). If its output is empty or invalid JSON, skip.
    2) Fallback to `memray summary` and extract bare floats (assumed MB).
    """
    # 1) JSON first
    proc = subprocess.run(
        [sys.executable, "-m", "memray", "stats", "--json", profile_file],
        capture_output=True, text=True
    )
    if proc.stdout:
        try:
            data = json.loads(proc.stdout)
            if "peak_memory" in data:
                return int(data["peak_memory"])
        except json.JSONDecodeError:
            # Not valid JSON → fallback
            pass

    # 2) Fallback: human summary → find all "NNN.NNNMB" or "NNN.NNNKB", convert to bytes
    proc = subprocess.run(
        [sys.executable, "-m", "memray", "summary", profile_file],
        capture_output=True, text=True, check=True
    )
    text = proc.stdout or proc.stderr
    # Match value and unit (KB or MB)
    matches = re.findall(r"([\d\.]+)([KM]B)", text)
    if not matches:
        raise RuntimeError(f"No memory size values in memray summary output:\n{text!r}")
    peak_bytes_list = []
    for val, unit in matches:
        size = float(val)
        if unit == "KB":
            peak_bytes_list.append(size * 1024)
        elif unit == "MB":
            peak_bytes_list.append(size * 1024**2)
    peak_bytes = max(peak_bytes_list)
    return int(peak_bytes)



# --- Helper Functions for Benchmarking ---

ALL_INPUT_SIZES = [
    (32, 32), (64, 64), (128, 128),
    (256, 256), (512, 512), (1024, 1024),(2048, 2048), (4096, 4096),
    (250, 250), (500, 500), (750, 750), (1500, 1500), (3000, 3000),
]

def run_benchmark(func, volume_shape, num_repetitions=5, *args, **kwargs):
    func_name = func.__name__
    execution_times = []
    peak_memory_usages = []

    for i in range(num_repetitions + 1):  # +1 for cold run
        # store .bin in temporary profile directory
        fname = f"benchmark_{func_name}_{i:03d}_{'x'.join(map(str, volume_shape))}.bin"
        profile_file = os.path.join(PROFILE_DIR, fname)
        with memray.Tracker(profile_file):
            t0 = time.perf_counter()
            func(*args, **kwargs)
            t1 = time.perf_counter()

        if i > 0:
            execution_times.append((t1 - t0) * 1000)
            peak_bytes = get_peak_memory_bytes(profile_file)
            peak_memory_usages.append(peak_bytes / (1024 * 1024))  # → MiB

    mean_time = np.mean(execution_times)
    sem_time = np.std(execution_times, ddof=0) / np.sqrt(len(execution_times))
    mean_mem  = np.mean(peak_memory_usages)
    sem_mem   = np.std(peak_memory_usages, ddof=0)  / np.sqrt(len(peak_memory_usages))

    print(f"  Mean Execution Time: {mean_time:.4f} ms ±{sem_time:.4f}")
    print(f"  Mean Peak Memory:    {mean_mem:.4f} MiB ±{sem_mem:.4f}")
    print("-" * 40)

    return {
        "function": func_name,
        "input_shape": volume_shape,
        "mean_time_ms": mean_time,
        "sem_time_ms": sem_time,
        "mean_memory_mb": mean_mem,
        "sem_memory_mb": sem_mem,
    }


# --- Benchmark Definitions ---

def benchmark_fullMat_ST():
    print("\n--- Benchmarking fullMat_ST (Legacy, Inefficient) ---")
    input_sizes = ALL_INPUT_SIZES[:2] # (32, 32), (64, 64)
    sigma = 1.0
    results = []
    for N, M in input_sizes:
        volume = np.random.rand(1, N, M).astype(np.float32)
        results.append(run_benchmark(fullMat_ST, (N, M), imgs=volume, sigma=sigma))
    return results

def benchmark_sepMat_ST():
    print("\n--- Benchmarking sepMat_ST (Legacy, Inefficient) ---")
    input_sizes = ALL_INPUT_SIZES[:5]
    sigma = 1.0
    results = []
    for N, M in input_sizes:
        volume = np.random.rand(1, N, M).astype(np.float32)
        results.append(run_benchmark(sepMat_ST, (N, M), imgs=volume, sigma=sigma))
    return results

def benchmark_DCT_by_FFT_ST():
    print("\n--- Benchmarking DCT_by_FFT_ST ---")
    input_sizes = ALL_INPUT_SIZES
    sigma = 1.0
    results = []
    for N, M in input_sizes:
        volume = np.random.rand(N, M).astype(np.float32)
        results.append(run_benchmark(DCT_by_FFT_ST, (N, M), imgs=volume, sigma=sigma))
    return results

def benchmark_DCT_ST():
    print("\n--- Benchmarking DCT_ST ---")
    input_sizes = ALL_INPUT_SIZES
    sigma = 1.0
    results = []
    for N, M in input_sizes:
        volume = np.random.rand(N, M).astype(np.float32)
        results.append(run_benchmark(DCT_ST, (N, M), imgs=volume, sigma=sigma))
    return results

def benchmark_FFT_ST():
    print("\n--- Benchmarking FFT_ST ---")
    input_sizes = ALL_INPUT_SIZES
    sigma = 1.0
    results = []
    for N, M in input_sizes:
        volume = np.random.rand(N, M).astype(np.float32)
        results.append(run_benchmark(FFT_ST, (N, M), imgs=volume, sigma=sigma))
    return results

# --- Report Generation ---

def generate_report(all_results, backend:str):
    print("\n--- Generating Performance Report ---")
    df = pd.DataFrame(all_results)
    system_info = get_system_info()
    sys_info_html = "<ul>" + "".join(f"<li><strong>{k}:</strong> {v}</li>" for k, v in system_info.items()) + "</ul>"

    output_dir = f"benchmarks/reports/imed-{imed_version}_backend={backend}/"
    os.makedirs(output_dir, exist_ok=True)


    # Plot 1: Execution time
    plt.figure(figsize=(15, 8))
    for func_name in df["function"].unique():
        func_df = df[df["function"] == func_name].sort_values(by="input_shape")
        input_labels = [str(s) for s in func_df["input_shape"]]
        plt.errorbar(input_labels, func_df["mean_time_ms"], yerr=func_df["sem_time_ms"], marker='o', capsize=5, label=func_name)

    plt.xlabel("Input Shape (N, M)")
    plt.ylabel("Mean Execution Time (ms, Log Scale)")
    plt.title("Mean Execution Time vs. Input Size (Logarithmic Y-axis)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path_log = os.path.join(output_dir, "execution_time_comparison_log.png")
    plt.savefig(plot_path_log)
    plt.close()
    print(f"Generated plot: {plot_path_log}")

    # Plot 2: Peak Memory Usage
    plt.figure(figsize=(15, 8))
    for func_name in df["function"].unique():
        func_df = df[df["function"] == func_name].sort_values(by="input_shape")
        input_labels = [str(s) for s in func_df["input_shape"]]
        plt.errorbar(input_labels, func_df["mean_memory_mb"], yerr=func_df["sem_memory_mb"], marker='o', capsize=5, label=func_name)

    plt.xlabel("Input Shape (N, M)")
    plt.ylabel("Mean Peak Memory (MB, Log Scale)")
    plt.title("Mean Peak Memory Usage vs. Input Size for imed Functions (Logarithmic Y-axis with SEM Error Bars)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path_log_memory = os.path.join(output_dir, "peak_memory_comparison_log.png")
    plt.savefig(plot_path_log_memory)
    plt.close()
    print(f"Generated plot: {plot_path_log_memory}")


    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>imed Performance Benchmarks</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #555; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            img {{ max-width: 100%; height: auto; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>imed Library Performance Benchmarks (Version: {imed_version})</h1>
        <p>This report details the execution time and memory benchmarks for key functions in the <code>imed</code> library.</p>

        <h2>System Information</h2>
        {sys_info_html}

        <h2>Benchmark Results</h2>
        {df.to_html(index=False)}

        <h2>Performance Comparison Plots</h2>        
        <h3>Execution Time (Logarithmic Y-axis)</h3>
        <img src="{os.path.basename(plot_path_log)}" alt="Execution Time Log Scale">
        <h3>Peak Memory Usage (Logarithmic Y-axis)</h3>
        <img src="{os.path.basename(plot_path_log_memory)}" alt="Peak Memory Usage Log Scale">

        <h2>Memory Profiling</h2>
        <p>For detailed memory profiling, you can still run this script externally using `memray run`:</p>
        <pre><code>memray run --output benchmarks_report.bin benchmarks/performance_benchmarks.py</code></pre>
    </body>
    </html>
    """
    report_path = os.path.join(output_dir, f"performance_report_imed-{imed_version}_python-{platform.python_version()}.html")
    with open(report_path, "w") as f:
        f.write(html_content)

    print(f"Generated HTML report: {report_path}")
    return report_path

# --- Main Execution ---

@contextmanager
def fft_backend(name: str):
    """
    Context‐manager that (a) enables pyFFTW cache if requested, and
    (b) installs the appropriate FFT backend for scipy.fft.set_backend.
    """
    if name == "pyfftw":
        import pyfftw
        # enable the FFTW wisdom cache
        pyfftw.interfaces.cache.enable()
        # install the FFTW‐powered backend
        with fft.set_backend(pyfftw.interfaces.scipy_fft):
            yield
    else:
        # default scipy backend — no need to set anything
        yield

def run_all_benchmarks():
    all_results = []
    all_results.extend(benchmark_fullMat_ST())
    all_results.extend(benchmark_sepMat_ST())
    all_results.extend(benchmark_DCT_by_FFT_ST())
    all_results.extend(benchmark_DCT_ST())
    all_results.extend(benchmark_FFT_ST())
    return all_results

if __name__ == "__main__":
    print("Starting performance benchmarks for imed library…")
    backend = "pyfftw"  # or "scipy"

    # one unified block:
    with fft_backend(backend):
        with tempfile.TemporaryDirectory() as tmpdir:
            PROFILE_DIR = tmpdir
            results = run_all_benchmarks()
            report = generate_report(results, backend)

            print("Performance benchmarks completed. "
                  "Profiles stored in temporary directory and cleaned up.")
            print(f"Open {report} in your browser to view the full report.")

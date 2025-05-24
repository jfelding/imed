import numpy as np
import time
import os
import sys
import re
import matplotlib.pyplot as plt
import pandas as pd
import platform
import psutil
import cpuinfo
from pathlib import Path
import memray
import json
import subprocess
import tempfile
from contextlib import contextmanager, nullcontext
from scipy import fft
import logging

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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_peak_memory_bytes(profile_file: str) -> float:
    """
    Return the peak memory usage in bytes recorded in a Memray profile file.
    
    1) Attempt `memray stats --json` (Memray ≥1.8). If JSON output is valid or writes to a file,
       load the JSON and look for `peak_memory` (bytes).
    """
    # 1) Try JSON-based stats first
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "memray", "stats", "--json", profile_file],
            capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        logger.warning("`memray stats --json` failed (exit %d): %s",
                       e.returncode, e.stderr or e.stdout)
    else:
        stdout = proc.stdout or ""
        stdout_stripped = stdout.strip()
        data = None

        # Check if memray wrote JSON to a file
        wrote_match = re.match(r"Wrote (.*\.json)$", stdout_stripped)
        if wrote_match:
            json_path = Path(wrote_match.group(1))
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                logger.error("Failed to load JSON from file %s: %s", json_path, e)
                raise ValueError(f"Could not read JSON file {json_path}: {e}") from e
        else:
            # Otherwise, assume JSON was printed to stdout
            if stdout_stripped.startswith("{"):
                try:
                    data = json.loads(stdout)
                except json.JSONDecodeError as e:
                    logger.error("Invalid JSON from memray stats:\n%s", stdout)
                    raise ValueError(
                        f"Invalid JSON from memray stats: {e} — output was:\n{stdout!r}"
                    ) from e

        if data:
            # look in both top-level and metadata
            peak = data.get("peak_memory")
            if peak is None and isinstance(data.get("metadata"), dict):
                peak = data["metadata"].get("peak_memory")

            if peak is not None:
                return peak
            else:
                logger.error("JSON is missing `peak_memory`: %s", data)
                raise ValueError("JSON did not contain `peak_memory` field")


# --- Helper Functions for Benchmarking ---

ALL_INPUT_SIZES = sorted([
    (2**5,  2**5), 
    (2**6,  2**6), 
    (2**7,  2**7),
    (2**8,  2**8),
    (2**9,  2**9),
    (2**10, 2**10),
    (2**11, 2**11),
    (2**12, 2**12),
    (250, 250), (500, 500), (750, 750), (1500, 1500), (3000, 3000),
])

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
            peak_memory_usages.append(peak_bytes)

    mean_time = np.mean(execution_times)
    sem_time = np.std(execution_times, ddof=0) / np.sqrt(len(execution_times))
    mean_mem  = np.mean(peak_memory_usages)
    sem_mem   = np.std(peak_memory_usages, ddof=0)  / np.sqrt(len(peak_memory_usages))

    print(f"  Mean Execution Time: {mean_time:.4f} ms ±{sem_time:.4f}")
    print(f"  Mean Peak Memory:    {mean_mem:.4f} B ±{sem_mem:.4f}")
    print("-" * 40)

    return {
        "function": func_name,
        "input_shape": volume_shape,
        "mean_time_ms": mean_time,
        "sem_time_ms": sem_time,
        "mean_memory_bytes": mean_mem,
        "sem_memory_bytes": sem_mem,
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
    input_sizes = ALL_INPUT_SIZES
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
        plt.errorbar(input_labels, func_df["mean_memory_bytes"], yerr=func_df["sem_memory_bytes"], marker='o', capsize=5, label=func_name)

    plt.xlabel("Input Shape (N, M)")
    plt.ylabel("Mean Peak Memory (B, Log Scale)")
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
        with fft.set_backend(pyfftw.interfaces.scipy_fft) and fft.set_workers(-1):
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
    for backend in "pyfftw", "scipy":

        with fft_backend(backend):
            with tempfile.TemporaryDirectory() as tmpdir:
                PROFILE_DIR = tmpdir
                results = run_all_benchmarks()
                report = generate_report(results, backend)

                print("Performance benchmarks completed. "
                    "Profiles stored in temporary directory and cleaned up.")
                print(f"Open {report} in your browser to view the full report.")

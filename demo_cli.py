#!/usr/bin/env python3
"""PyReceipt Demo CLI Tool.

External demonstration script to test PyReceipt OCR extraction, performance profiling,
and regex parsing on sample dataset images in datasets/.
Displays hardware specifications, stage-by-stage memory/execution metrics, and parsed results.
"""

import argparse
import glob
import logging
import os
import platform
import sys
import time
import tracemalloc
from typing import Dict, Any

from pyreceipt.adapters.tesseract_ocr import TesseractOCRAdapter
from pyreceipt.core.parser import ReceiptParser


def get_hardware_info() -> Dict[str, Any]:
    """Retrieve system hardware details (OS, CPU cores, RAM)."""
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "architecture": platform.machine(),
        "cpu_cores": os.cpu_count() or 1,
        "ram_gb": "Unknown",
    }

    # Attempt to retrieve RAM via psutil
    try:
        import psutil

        mem = psutil.virtual_memory()
        info["ram_gb"] = f"{mem.total / (1024**3):.2f} GB"
        return info
    except ImportError:
        pass

    # Fallback for Linux / Raspberry Pi reading /proc/meminfo
    if os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        kb = int(parts[1])
                        info["ram_gb"] = f"{kb / (1024**2):.2f} GB"
                        break
        except Exception:
            pass

    return info


def main() -> None:
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="PyReceipt Demo CLI - Process receipt images with hardware & performance telemetry"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to receipt image file. If omitted, a sample from datasets/ will be selected.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ExpressExpense", "SROIE2019"],
        default="ExpressExpense",
        help="Dataset directory to sample image from if --image is omitted.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Language code for JSON parser rules (e.g., 'en', 'it').",
    )
    args = parser.parse_args()

    # Determine target image path
    image_path = args.image
    if not image_path:
        dataset_dir = os.path.join("datasets", args.dataset)
        sample_images = sorted(glob.glob(f"{dataset_dir}/**/*.jpg", recursive=True))
        if not sample_images:
            sample_images = sorted(
                glob.glob(f"{dataset_dir}/**/*.png", recursive=True)
            )

        if not sample_images:
            print(f"Error: No image files found in '{dataset_dir}'.")
            sys.exit(1)
        image_path = sample_images[0]

    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' does not exist.")
        sys.exit(1)

    hw_info = get_hardware_info()

    print("=" * 65)
    print("                PyReceipt Performance Demo CLI")
    print("=" * 65)
    print(" [HARDWARE SPECIFICATIONS]")
    print(f"  OS / Kernel  : {hw_info['os']}")
    print(f"  Architecture : {hw_info['architecture']}")
    print(f"  CPU Cores    : {hw_info['cpu_cores']}")
    print(f"  Total RAM    : {hw_info['ram_gb']}")
    print("-" * 65)
    print(f" Target Image  : {image_path}")
    print(f" Parser Lang   : {args.lang}")
    print("=" * 65)

    # Initialize Adapter and Parser
    tess_lang = "eng" if args.lang == "en" else args.lang
    ocr_adapter = TesseractOCRAdapter(lang=tess_lang)
    receipt_parser = ReceiptParser(lang_code=args.lang)

    # Start Overall Tracemalloc & Timer
    tracemalloc.start()
    start_overall = time.perf_counter()

    # --- Step 1: Tesseract OCR ---
    tracemalloc.reset_peak()
    start_tess = time.perf_counter()
    raw_text = ocr_adapter.extract_text(image_path)
    tess_time = time.perf_counter() - start_tess
    _, tess_peak = tracemalloc.get_traced_memory()
    tess_peak_mb = tess_peak / (1024 * 1024)

    # --- Step 2: Regex Parser ---
    tracemalloc.reset_peak()
    start_parse = time.perf_counter()
    receipt = receipt_parser.parse(raw_text)
    parse_time = time.perf_counter() - start_parse
    _, parse_peak = tracemalloc.get_traced_memory()
    parse_peak_mb = parse_peak / (1024 * 1024)

    # --- Overall Metrics ---
    overall_time = time.perf_counter() - start_overall
    _, overall_peak = tracemalloc.get_traced_memory()
    overall_peak_mb = overall_peak / (1024 * 1024)
    tracemalloc.stop()

    print("\n [PERFORMANCE METRICS BREAKDOWN]")
    print(f" {'Stage':<22} | {'Time (s)':<14} | {'Peak RAM (MB)':<15}")
    print(" " + "-" * 22 + "-+-" + "-" * 14 + "-+-" + "-" * 15)
    print(f" 1. Tesseract OCR       | {tess_time:12.4f} s | {tess_peak_mb:13.2f} MB")
    print(f" 2. Regex Parser        | {parse_time:12.4f} s | {parse_peak_mb:13.2f} MB")
    print(" " + "-" * 22 + "-+-" + "-" * 14 + "-+-" + "-" * 15)
    print(
        f" OVERALL PIPELINE       | {overall_time:12.4f} s | {overall_peak_mb:13.2f} MB"
    )
    print("=" * 65)

    print("\n [FULL RAW OCR TEXT OUTPUT]")
    if raw_text:
        for line in raw_text.splitlines():
            print(f"  {line}")
    else:
        print("  [No text detected]")
    print("-" * 65)

    print("\n [PARSED RECEIPT ENTITY]")
    print(f"  Company  : {receipt.company}")
    print(f"  Date     : {receipt.date if receipt.date else 'N/A'}")
    print(f"  Total    : {receipt.total:.2f}")
    print(f"  Category : {receipt.category.name}")
    print("=" * 65)


if __name__ == "__main__":
    main()

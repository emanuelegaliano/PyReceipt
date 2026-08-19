#!/usr/bin/env python3
"""PyReceipt Demo CLI Tool.

Demonstrates receipt processing with:
1. OCR Text & 2D Bounding Box Extraction (Tesseract, RapidOCR, EasyOCR)
2. Configurable Parser Architectures:
   - 'spatial2d': Geometric 2D Row Clustering & Ray-Casting (Method 1)
   - 'layoutlm' : Multi-modal 2D Visual Document Attention (Method 2)
   - 'regex'    : Traditional Linear Regex Parser
3. Layered Accuracy Verification (when ground-truth metadata is available)
4. System Hardware & Telemetry Profiling (Time, RAM)
"""

import argparse
import glob
import json
import logging
import os
from pathlib import Path
import platform
import re
import sys
import time
import tracemalloc
from typing import Dict, Any, Optional

from pyreceipt.core.parser import ReceiptParser
from pyreceipt.adapters.spatial_2d_parser import Spatial2DBoxParser


def get_hardware_info() -> Dict[str, Any]:
    """Retrieve system hardware details (OS, CPU cores, RAM)."""
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "architecture": platform.machine(),
        "cpu_cores": os.cpu_count() or 1,
        "ram_gb": "Unknown",
    }

    try:
        import psutil

        mem = psutil.virtual_memory()
        info["ram_gb"] = f"{mem.total / (1024**3):.2f} GB"
        return info
    except ImportError:
        pass

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


def load_env() -> Dict[str, str]:
    """Load key-value environment variables from .env if present."""
    env_vars: Dict[str, str] = {}
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    env_vars[k.strip()] = v.strip().strip("'\"")
        except Exception:
            pass
    return env_vars


def find_ground_truth(image_path: str) -> Optional[Dict[str, Any]]:
    """Attempt to locate ground truth JSON/TXT entity file for the given image."""
    img_p = Path(image_path)
    stem = img_p.stem

    for parent in [img_p.parent, img_p.parent.parent]:
        ent_candidate = parent / "entities" / f"{stem}.txt"
        if ent_candidate.exists():
            try:
                with open(ent_candidate, "r", encoding="utf-8", errors="ignore") as f:
                    data = json.load(f)
                gt_tot = 0.0
                try:
                    raw_t = str(data.get("total", "0")).replace(",", "").replace("$", "").replace("RM", "").strip()
                    gt_tot = float(raw_t)
                except Exception:
                    pass
                return {
                    "company": data.get("company", "").strip(),
                    "date": data.get("date", "").strip(),
                    "total": gt_tot,
                }
            except Exception:
                pass
    return None


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="PyReceipt Demo CLI - Process receipt images with selectable OCR and Parser models"
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
        "--ocr",
        type=str,
        choices=["tesseract", "rapidocr", "easyocr"],
        default="tesseract",
        help="OCR engine adapter to use (default: 'tesseract').",
    )
    parser.add_argument(
        "--parser",
        type=str,
        choices=["spatial2d", "layoutlm", "regex"],
        default="spatial2d",
        help="Parser architecture to use: 'spatial2d' (Method 1), 'layoutlm' (Method 2), or 'regex' (Baseline).",
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
        env_vars = load_env()
        if args.dataset == "ExpressExpense" and (
            os.environ.get("EXPRESS_EXPENSE_PATH") or env_vars.get("EXPRESS_EXPENSE_PATH")
        ):
            dataset_dir = os.environ.get("EXPRESS_EXPENSE_PATH") or env_vars.get(
                "EXPRESS_EXPENSE_PATH"
            )
        elif args.dataset == "SROIE2019" and (
            os.environ.get("SROIE_PATH") or env_vars.get("SROIE_PATH")
        ):
            dataset_dir = os.environ.get("SROIE_PATH") or env_vars.get("SROIE_PATH")
        else:
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

    print("=" * 68)
    print("                 PyReceipt Performance Demo CLI")
    print("=" * 68)
    print(" [HARDWARE SPECIFICATIONS]")
    print(f"  OS / Kernel  : {hw_info['os']}")
    print(f"  Architecture : {hw_info['architecture']}")
    print(f"  CPU Cores    : {hw_info['cpu_cores']}")
    print(f"  Total RAM    : {hw_info['ram_gb']}")
    print("-" * 68)
    print(f" Target Image  : {image_path}")
    print(f" OCR Engine    : {args.ocr}")
    print(f" Parser Model  : {args.parser.upper()}")
    print("=" * 68)

    # Initialize Adapter
    if args.ocr == "rapidocr":
        try:
            from pyreceipt.adapters.paddle_ocr import RapidOCRAdapter
            ocr_adapter = RapidOCRAdapter()
            ocr_name = "RapidOCR (ONNX)"
        except ImportError:
            print("Error: rapidocr-onnxruntime is not installed.")
            sys.exit(1)
    elif args.ocr == "easyocr":
        try:
            from pyreceipt.adapters.easy_ocr import EasyOCRAdapter
            ocr_lang = ["en"] if args.lang == "en" else [args.lang, "en"]
            ocr_adapter = EasyOCRAdapter(lang_list=ocr_lang)
            ocr_name = "EasyOCR (PyTorch CRAFT)"
        except ImportError:
            print("Error: easyocr is not installed.")
            sys.exit(1)
    else:
        from pyreceipt.adapters.tesseract_ocr import TesseractOCRAdapter
        tess_lang = "eng" if args.lang == "en" else args.lang
        ocr_adapter = TesseractOCRAdapter(lang=tess_lang)
        ocr_name = "Tesseract OCR"

    # Telemetry and Execution
    tracemalloc.start()
    start_overall = time.perf_counter()

    if args.parser == "layoutlm":
        try:
            from pyreceipt.adapters.layoutlm_parser import LayoutLMReceiptParser
            llm_parser = LayoutLMReceiptParser()
            tracemalloc.reset_peak()
            start_parse = time.perf_counter()
            receipt = llm_parser.parse_image(image_path)
            parse_time = time.perf_counter() - start_parse
            _, parse_peak = tracemalloc.get_traced_memory()
            parse_peak_mb = parse_peak / (1024 * 1024)
            ocr_time = 0.0
            ocr_peak_mb = 0.0
            raw_text = ""
        except Exception as e:
            print(f"Error initializing LayoutLM: {e}")
            sys.exit(1)
    else:
        # Step 1: OCR Text/Box Extraction
        tracemalloc.reset_peak()
        start_ocr = time.perf_counter()
        boxes = ocr_adapter.extract_boxes(image_path)
        raw_text = "\n".join(b["text"] for b in boxes)
        ocr_time = time.perf_counter() - start_ocr
        _, ocr_peak = tracemalloc.get_traced_memory()
        ocr_peak_mb = ocr_peak / (1024 * 1024)

        # Step 2: Parsing
        tracemalloc.reset_peak()
        start_parse = time.perf_counter()
        if args.parser == "spatial2d":
            spatial_parser = Spatial2DBoxParser(lang_code=args.lang)
            receipt = spatial_parser.parse(boxes)
        else:
            regex_parser = ReceiptParser(lang_code=args.lang)
            receipt = regex_parser.parse(raw_text)

        parse_time = time.perf_counter() - start_parse
        _, parse_peak = tracemalloc.get_traced_memory()
        parse_peak_mb = parse_peak / (1024 * 1024)

    overall_time = time.perf_counter() - start_overall
    _, overall_peak = tracemalloc.get_traced_memory()
    overall_peak_mb = overall_peak / (1024 * 1024)
    tracemalloc.stop()

    # Display Extracted Entities
    print(f"\n [EXTRACTED RECEIPT ENTITIES ({args.parser.upper()} Output)]")
    print(f"  Merchant / Company : {receipt.company if receipt.company else '[Not Found]'}")
    print(f"  Transaction Date   : {receipt.date if receipt.date else '[Not Found]'}")
    print(f"  Total Amount       : {receipt.total:.2f}")
    print(f"  Expense Category   : {receipt.category.name}")

    # Ground Truth Verification if available
    gt = find_ground_truth(image_path)
    if gt:
        print("-" * 68)
        print(" [ACCURACY & CAPTURE VERIFICATION (Ground Truth Available)]")
        regex_matched = abs(receipt.total - gt["total"]) < 0.01

        print(f"  Expected Total     : {gt['total']:.2f}")
        print(f"  Parser Extraction  : {'[CORRECT] Matched Ground Truth' if regex_matched else '[MISMATCH] Wrong Total Extracted'}")

    # Performance Breakdown
    print("\n [PERFORMANCE TELEMETRY BREAKDOWN]")
    print(f" {'Pipeline Stage':<26} | {'Time (s)':<14} | {'Peak RAM (MB)':<15}")
    print(" " + "-" * 26 + "-+-" + "-" * 14 + "-+-" + "-" * 15)
    if args.parser != "layoutlm":
        print(f" {ocr_name + ' Stage':<26} | {ocr_time:12.4f} s | {ocr_peak_mb:13.2f} MB")
    print(f" {args.parser.upper() + ' Parser Stage':<26} | {parse_time:12.4f} s | {parse_peak_mb:13.2f} MB")
    print(" " + "-" * 26 + "-+-" + "-" * 14 + "-+-" + "-" * 15)
    print(f" {'OVERALL PIPELINE':<26} | {overall_time:12.4f} s | {overall_peak_mb:13.2f} MB")
    print("=" * 68)


if __name__ == "__main__":
    main()

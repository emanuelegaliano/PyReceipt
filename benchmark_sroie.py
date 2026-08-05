#!/usr/bin/env python3
"""PyReceipt Benchmark Suite for SROIE2019 Dataset.

Evaluates accuracy and hardware performance pre-regex (ground-truth OCR text)
and post-regex (end-to-end Tesseract OCR + Regex parser on images).
"""

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
import tracemalloc
from typing import Dict, List, Any, Optional

from pyreceipt.adapters.tesseract_ocr import TesseractOCRAdapter
from pyreceipt.core.parser import RegexReceiptParser


def parse_ground_truth(entity_file: Path) -> Dict[str, Any]:
    """Parse ground truth JSON entity file.

    Args:
        entity_file: Path to entity file in entities/ directory.

    Returns:
        Dictionary with gt_company, gt_date, gt_total fields.
    """
    with open(entity_file, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)

    gt_total = 0.0
    try:
        raw_tot = data.get("total", "0.0").replace(",", "").strip()
        gt_total = float(raw_tot)
    except ValueError:
        pass

    return {
        "company": data.get("company", "").strip(),
        "date": data.get("date", "").strip(),
        "total": gt_total,
    }


def parse_box_file(box_file: Path) -> str:
    """Extract text lines from bounding box OCR file.

    Box format: x1,y1,x2,y2,x3,y3,x4,y4,text_content

    Args:
        box_file: Path to box file in box/ directory.

    Returns:
        Joined raw text string.
    """
    lines: List[str] = []
    with open(box_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(",", 8)
            if len(parts) >= 9:
                lines.append(parts[8])
            elif parts:
                lines.append(line.strip())
    return "\n".join(lines)


def is_total_match(parsed_total: float, gt_total: float) -> bool:
    """Check if parsed total matches ground truth within 0.01 tolerance."""
    if gt_total <= 0:
        return True
    return abs(parsed_total - gt_total) < 0.01


def is_date_match(parsed_date: str, gt_date: str) -> bool:
    """Check if parsed date matches ground truth date."""
    if not gt_date:
        return True
    p_norm = re.sub(r"[^\d]", "", parsed_date)
    g_norm = re.sub(r"[^\d]", "", gt_date)
    return p_norm == g_norm if (p_norm and g_norm) else parsed_date == gt_date


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PyReceipt SROIE2019 Benchmark Suite - Pre-Regex & Post-Regex Evaluation"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="datasets/SROIE2019/test",
        help="Path to SROIE2019 split directory (e.g. datasets/SROIE2019/test)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=30,
        help="Maximum number of sample receipts to evaluate (default: 30)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Parser language configuration code (default: 'en')",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    entities_dir = dataset_path / "entities"
    box_dir = dataset_path / "box"
    img_dir = dataset_path / "img"

    if not entities_dir.is_dir() or not box_dir.is_dir() or not img_dir.is_dir():
        print(f"Error: Dataset path '{dataset_path}' must contain entities/, box/, and img/ subdirectories.")
        sys.exit(1)

    entity_files = sorted(list(entities_dir.glob("*.txt")))
    if args.max_samples and args.max_samples > 0:
        entity_files = entity_files[: args.max_samples]

    total_samples = len(entity_files)
    if total_samples == 0:
        print(f"Error: No entity files found in '{entities_dir}'.")
        sys.exit(1)

    print("=" * 65)
    print("        PyReceipt SROIE2019 Benchmark Suite")
    print("=" * 65)
    print(f" Dataset Path    : {dataset_path}")
    print(f" Target Samples  : {total_samples}")
    print(f" Parser Lang     : {args.lang}")
    print("=" * 65)

    # Initialize components
    tess_lang = "eng" if args.lang == "en" else args.lang
    ocr_adapter = TesseractOCRAdapter(lang=tess_lang)
    receipt_parser = RegexReceiptParser(lang_code=args.lang)

    # Telemetry tracking variables
    pre_total_correct = 0
    pre_date_correct = 0

    post_total_correct = 0
    post_date_correct = 0

    ocr_times: List[float] = []
    ocr_peaks: List[float] = []

    regex_times: List[float] = []
    regex_peaks: List[float] = []

    for idx, ent_file in enumerate(entity_files, start=1):
        sample_id = ent_file.stem
        gt = parse_ground_truth(ent_file)

        box_file = box_dir / f"{sample_id}.txt"
        img_file = img_dir / f"{sample_id}.jpg"

        # -------------------------------------------------------------
        # Mode 1: Pre-Regex Evaluation (Text Box Ground Truth)
        # -------------------------------------------------------------
        if box_file.exists():
            box_text = parse_box_file(box_file)
            pre_receipt = receipt_parser.parse(box_text)

            if is_total_match(pre_receipt.total, gt["total"]):
                pre_total_correct += 1
            if is_date_match(pre_receipt.date, gt["date"]):
                pre_date_correct += 1

        # -------------------------------------------------------------
        # Mode 2: Post-Regex Evaluation (End-to-End Image OCR + Regex)
        # -------------------------------------------------------------
        if img_file.exists():
            tracemalloc.start()
            tracemalloc.reset_peak()
            t0 = time.perf_counter()
            raw_ocr_text = ocr_adapter.extract_text(str(img_file))
            t1 = time.perf_counter()
            _, peak_ocr = tracemalloc.get_traced_memory()
            ocr_times.append(t1 - t0)
            ocr_peaks.append(peak_ocr / (1024 * 1024))

            tracemalloc.reset_peak()
            t2 = time.perf_counter()
            post_receipt = receipt_parser.parse(raw_ocr_text)
            t3 = time.perf_counter()
            _, peak_regex = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            regex_times.append(t3 - t2)
            regex_peaks.append(peak_regex / (1024 * 1024))

            if is_total_match(post_receipt.total, gt["total"]):
                post_total_correct += 1
            if is_date_match(post_receipt.date, gt["date"]):
                post_date_correct += 1

        print(f" Progress: [{idx:02d}/{total_samples:02d}] Processed {sample_id}", end="\r")

    print("\n" + "=" * 65)
    print("                 BENCHMARK RESULTS REPORT")
    print("=" * 65)

    pre_tot_acc = (pre_total_correct / total_samples) * 100
    pre_date_acc = (pre_date_correct / total_samples) * 100

    post_tot_acc = (post_total_correct / total_samples) * 100
    post_date_acc = (post_date_correct / total_samples) * 100

    avg_ocr_time = sum(ocr_times) / len(ocr_times) if ocr_times else 0.0
    avg_ocr_peak = sum(ocr_peaks) / len(ocr_peaks) if ocr_peaks else 0.0

    avg_regex_time = sum(regex_times) / len(regex_times) if regex_times else 0.0
    avg_regex_peak = sum(regex_peaks) / len(regex_peaks) if regex_peaks else 0.0

    print("\n [1. ACCURACY EVALUATION]")
    print(f" {'Evaluation Stage':<25} | {'Total Acc (%)':<15} | {'Date Acc (%)':<15}")
    print(" " + "-" * 25 + "-+-" + "-" * 15 + "-+-" + "-" * 15)
    print(f" Pre-Regex (Box OCR Text)  | {pre_tot_acc:14.1f}% | {pre_date_acc:14.1f}%")
    print(f" Post-Regex (End-to-End)   | {post_tot_acc:14.1f}% | {post_date_acc:14.1f}%")
    print("=" * 65)

    print("\n [2. HARDWARE PERFORMANCE TELEMETRY]")
    print(f" {'Stage':<25} | {'Avg Time (s)':<15} | {'Avg Peak RAM (MB)':<17}")
    print(" " + "-" * 25 + "-+-" + "-" * 15 + "-+-" + "-" * 17)
    print(f" Tesseract OCR Stage       | {avg_ocr_time:13.4f} s | {avg_ocr_peak:15.2f} MB")
    print(f" Regex Parser Stage        | {avg_regex_time:13.4f} s | {avg_regex_peak:15.2f} MB")
    print(" " + "-" * 25 + "-+-" + "-" * 15 + "-+-" + "-" * 17)
    total_avg_time = avg_ocr_time + avg_regex_time
    total_avg_peak = max(avg_ocr_peak, avg_regex_peak)
    print(f" OVERALL PIPELINE          | {total_avg_time:13.4f} s | {total_avg_peak:15.2f} MB")
    print("=" * 65)


if __name__ == "__main__":
    main()

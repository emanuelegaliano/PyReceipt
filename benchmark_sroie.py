#!/usr/bin/env python3
"""PyReceipt Benchmark Suite for SROIE2019 Dataset.

Evaluates and isolates:
1. OCR Raw Capture Capability: Did the OCR model transcribe the numbers/dates from the image?
2. Pre-Regex Baseline: Parser accuracy on human-annotated ground-truth OCR boxes.
3. End-to-End Final Accuracy: Final pipeline output (OCR + Selected Parser).
4. Failure Root-Cause Attribution: Isolates OCR transcription errors vs. Parser misattribution errors.
"""

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
import tracemalloc
from typing import Dict, List, Any, Optional, Tuple

from pyreceipt.core.parser import RegexReceiptParser
from pyreceipt.adapters.spatial_2d_parser import Spatial2DBoxParser


def parse_ground_truth(entity_file: Path) -> Dict[str, Any]:
    """Parse ground truth JSON entity file."""
    with open(entity_file, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)

    gt_total = 0.0
    try:
        raw_tot = str(data.get("total", "0.0")).replace(",", "").replace("$", "").replace("RM", "").strip()
        gt_total = float(raw_tot)
    except ValueError:
        pass

    return {
        "company": data.get("company", "").strip(),
        "date": data.get("date", "").strip(),
        "total": gt_total,
    }


def parse_box_file(box_file: Path) -> List[Dict[str, Any]]:
    """Extract structured bounding boxes from human-annotated box file."""
    boxes: List[Dict[str, Any]] = []
    with open(box_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(",", 8)
            if len(parts) >= 9:
                try:
                    xs = [int(parts[0]), int(parts[2]), int(parts[4]), int(parts[6])]
                    ys = [int(parts[1]), int(parts[3]), int(parts[5]), int(parts[7])]
                    boxes.append({
                        "text": parts[8],
                        "box": [min(xs), min(ys), max(xs), max(ys)],
                        "conf": 1.0,
                    })
                except Exception:
                    boxes.append({"text": parts[8], "box": [0, 0, 100, 20], "conf": 1.0})
    return boxes


def normalize_ocr_numbers(text: str) -> str:
    """Normalize common OCR spacing artifacts in numbers (e.g., '21. 60' -> '21.60')."""
    return re.sub(r"(\d+)\s*[\.,]\s*(\d{2})\b", r"\1.\2", text)


def is_total_in_raw_text(raw_text: str, gt_total: float) -> bool:
    """Check if the ground truth total exists anywhere in the raw OCR output."""
    if gt_total <= 0:
        return True
    norm_text = normalize_ocr_numbers(raw_text)
    str_2dec = f"{gt_total:.2f}"
    str_g = f"{gt_total:g}"
    str_comma = str_2dec.replace(".", ",")
    return (str_2dec in norm_text) or (str_g in norm_text) or (str_comma in norm_text)


def is_date_in_raw_text(raw_text: str, gt_date: str) -> bool:
    """Check if the ground truth date components exist in the raw OCR output."""
    if not gt_date:
        return True
    g_digits = re.sub(r"[^\d]", "", gt_date)
    r_digits = re.sub(r"[^\d]", "", raw_text)
    if g_digits and g_digits in r_digits:
        return True
    parts = re.split(r"[/.-]", gt_date.strip())
    if len(parts) >= 3 and all(p in raw_text for p in parts if len(p) >= 2):
        return True
    return False


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


def load_env() -> Dict[str, str]:
    """Load key-value environment variables from .env if present."""
    env_vars: Dict[str, str] = {}
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.exists():
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


def resolve_sroie_dir(base_dir: str) -> Path:
    """Resolve the SROIE split directory containing entities, box, and img."""
    p = Path(base_dir)
    if (p / "entities").is_dir() and (p / "img").is_dir():
        return p
    for candidate in ["test", "train", "task1train", "task1test", "0325updated.task1train(626p)"]:
        sub = p / candidate
        if sub.is_dir() and (sub / "entities").is_dir():
            return sub
    if p.is_dir():
        for sub in p.iterdir():
            if sub.is_dir() and (sub / "entities").is_dir():
                return sub
    return p


def create_ocr_adapter(ocr_key: str, lang: str = "en") -> Tuple[Optional[Any], str, Optional[str]]:
    """Instantiate OCR adapter dynamically with error handling."""
    if ocr_key == "tesseract":
        try:
            from pyreceipt.adapters.tesseract_ocr import TesseractOCRAdapter
            tess_lang = "eng" if lang == "en" else lang
            return TesseractOCRAdapter(lang=tess_lang), "Tesseract OCR", None
        except Exception as e:
            return None, "Tesseract OCR", str(e)

    elif ocr_key == "rapidocr":
        try:
            from pyreceipt.adapters.paddle_ocr import RapidOCRAdapter
            return RapidOCRAdapter(), "RapidOCR (ONNX)", None
        except Exception as e:
            return None, "RapidOCR (ONNX)", str(e)

    elif ocr_key == "easyocr":
        try:
            from pyreceipt.adapters.easy_ocr import EasyOCRAdapter
            ocr_lang = ["en"] if lang == "en" else [lang, "en"]
            return EasyOCRAdapter(lang_list=ocr_lang), "EasyOCR (CRAFT)", None
        except Exception as e:
            return None, "EasyOCR (CRAFT)", str(e)

    return None, ocr_key, f"Unknown OCR engine '{ocr_key}'"


def evaluate_engine(
    ocr_name: str,
    ocr_adapter: Any,
    parser_type: str,
    entity_files: List[Path],
    box_dir: Path,
    img_dir: Path,
    lang: str = "en",
) -> Dict[str, Any]:
    """Run full benchmark evaluation for an OCR engine and Parser configuration."""
    total_samples = len(entity_files)

    # Initialize selected parser
    if parser_type == "spatial2d":
        parser_instance = Spatial2DBoxParser()
        parser_display = "Spatial 2D Box Parser"
    elif parser_type == "layoutlm":
        from pyreceipt.adapters.layoutlm_parser import LayoutLMReceiptParser
        parser_instance = LayoutLMReceiptParser()
        parser_display = "LayoutLM Document AI"
    else:
        parser_instance = RegexReceiptParser(lang_code=lang)
        parser_display = "Regex Parser (1D Baseline)"

    # Counters
    ocr_total_captured = 0
    ocr_date_captured = 0
    pre_total_correct = 0
    pre_date_correct = 0
    post_total_correct = 0
    post_date_correct = 0

    fail_ocr_missed = 0
    fail_parser_misattributed = 0

    ocr_times: List[float] = []
    ocr_peaks: List[float] = []
    parse_times: List[float] = []
    parse_peaks: List[float] = []

    print(f"\n--- Running Benchmark: {ocr_name} + {parser_display} ({total_samples} samples) ---")

    for idx, ent_file in enumerate(entity_files, start=1):
        sample_id = ent_file.stem
        gt = parse_ground_truth(ent_file)

        box_file = box_dir / f"{sample_id}.txt"
        img_file = img_dir / f"{sample_id}.jpg"
        if not img_file.exists():
            img_file = img_dir / f"{sample_id}.png"

        # Baseline: Pre-Evaluation on Human Box Ground Truth
        if box_file.exists():
            gt_boxes = parse_box_file(box_file)
            if parser_type == "spatial2d":
                pre_receipt = parser_instance.parse(gt_boxes)
            else:
                gt_text = "\n".join(b["text"] for b in gt_boxes)
                pre_receipt = parser_instance.parse(gt_text)

            if is_total_match(pre_receipt.total, gt["total"]):
                pre_total_correct += 1
            if is_date_match(pre_receipt.date, gt["date"]):
                pre_date_correct += 1

        # Real Execution: End-to-End Image OCR + Parsing
        if img_file.exists():
            # Step 1: OCR
            tracemalloc.start()
            tracemalloc.reset_peak()
            t0 = time.perf_counter()

            if parser_type == "layoutlm":
                ocr_time = 0.0
                peak_ocr = 0.0
                raw_ocr_text = ""
                extracted_boxes = []
            else:
                extracted_boxes = ocr_adapter.extract_boxes(str(img_file))
                raw_ocr_text = "\n".join(b["text"] for b in extracted_boxes)
                t1 = time.perf_counter()
                _, peak_ocr = tracemalloc.get_traced_memory()
                ocr_times.append(t1 - t0)
                ocr_peaks.append(peak_ocr / (1024 * 1024))

            # Check OCR raw capture rate
            total_in_ocr = is_total_in_raw_text(raw_ocr_text, gt["total"]) if raw_ocr_text else True
            date_in_ocr = is_date_in_raw_text(raw_ocr_text, gt["date"]) if raw_ocr_text else True

            if total_in_ocr:
                ocr_total_captured += 1
            if date_in_ocr:
                ocr_date_captured += 1

            # Step 2: Parser Execution
            tracemalloc.reset_peak()
            t2 = time.perf_counter()

            if parser_type == "spatial2d":
                post_receipt = parser_instance.parse(extracted_boxes)
            elif parser_type == "layoutlm":
                post_receipt = parser_instance.parse_image(str(img_file))
            else:
                post_receipt = parser_instance.parse(raw_ocr_text)

            t3 = time.perf_counter()
            _, peak_parse = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            parse_times.append(t3 - t2)
            parse_peaks.append(peak_parse / (1024 * 1024))

            # Check final match
            matched_total = is_total_match(post_receipt.total, gt["total"])
            matched_date = is_date_match(post_receipt.date, gt["date"])

            if matched_total:
                post_total_correct += 1
            else:
                if total_in_ocr:
                    fail_parser_misattributed += 1
                else:
                    fail_ocr_missed += 1

            if matched_date:
                post_date_correct += 1

        print(f" Progress: [{idx:02d}/{total_samples:02d}] Sample {sample_id}", end="\r")

    print(f"\n Completed {ocr_name} with {parser_display}.")

    return {
        "engine_name": ocr_name,
        "parser_name": parser_display,
        "total_samples": total_samples,
        "ocr_total_cap": (ocr_total_captured / total_samples) * 100 if total_samples else 0.0,
        "ocr_date_cap": (ocr_date_captured / total_samples) * 100 if total_samples else 0.0,
        "pre_total_acc": (pre_total_correct / total_samples) * 100 if total_samples else 0.0,
        "pre_date_acc": (pre_date_correct / total_samples) * 100 if total_samples else 0.0,
        "post_total_acc": (post_total_correct / total_samples) * 100 if total_samples else 0.0,
        "post_date_acc": (post_date_correct / total_samples) * 100 if total_samples else 0.0,
        "fail_ocr_missed": fail_ocr_missed,
        "fail_parser_misattributed": fail_parser_misattributed,
        "fail_parser_pct": (fail_parser_misattributed / total_samples) * 100 if total_samples else 0.0,
        "avg_ocr_time": sum(ocr_times) / len(ocr_times) if ocr_times else 0.0,
        "avg_ocr_peak": sum(ocr_peaks) / len(ocr_peaks) if ocr_peaks else 0.0,
        "avg_parse_time": sum(parse_times) / len(parse_times) if parse_times else 0.0,
        "avg_parse_peak": sum(parse_peaks) / len(parse_peaks) if parse_peaks else 0.0,
    }


def print_single_report(res: Dict[str, Any]) -> None:
    """Print detailed performance and root-cause benchmark report."""
    print("\n" + "=" * 70)
    print(f"       BENCHMARK REPORT: {res['engine_name']} + {res['parser_name']}")
    print("=" * 70)

    print("\n [1. LAYERED ACCURACY BREAKDOWN]")
    print(f" {'Evaluation Layer':<28} | {'Total Acc (%)':<16} | {'Date Acc (%)':<16}")
    print(" " + "-" * 28 + "-+-" + "-" * 16 + "-+-" + "-" * 16)
    print(f" 1. OCR Raw Capture Rate     | {res['ocr_total_cap']:15.1f}% | {res['ocr_date_cap']:15.1f}%")
    print(f" 2. Pre-Eval (Box Truth)     | {res['pre_total_acc']:15.1f}% | {res['pre_date_acc']:15.1f}%")
    print(f" 3. Final Pipeline Accuracy  | {res['post_total_acc']:15.1f}% | {res['post_date_acc']:15.1f}%")
    print("=" * 70)

    total_failures = res["fail_ocr_missed"] + res["fail_parser_misattributed"]
    print("\n [2. ROOT-CAUSE FAILURE ANALYSIS (Total Price Field)]")
    print(f"  Total Evaluated Samples    : {res['total_samples']}")
    print(f"  Correct Total Extractions  : {res['total_samples'] - total_failures} ({res['post_total_acc']:.1f}%)")
    print(f"  Total Extraction Failures  : {total_failures} ({(total_failures/res['total_samples'])*100:.1f}%)")
    print("  " + "-" * 66)
    print(f"  * Parser Misattributions   : {res['fail_parser_misattributed']:2d} ({res['fail_parser_pct']:.1f}%)  [OCR captured number, Parser chose wrong field]")
    ocr_miss_pct = (res["fail_ocr_missed"] / res["total_samples"]) * 100 if res["total_samples"] else 0.0
    print(f"  * OCR Transcription Errors : {res['fail_ocr_missed']:2d} ({ocr_miss_pct:.1f}%)  [OCR model missed/garbled the characters]")
    print("=" * 70)

    print("\n [3. HARDWARE & PERFORMANCE TELEMETRY]")
    print(f" {'Pipeline Stage':<28} | {'Avg Time (s)':<16} | {'Avg Peak RAM (MB)':<17}")
    print(" " + "-" * 28 + "-+-" + "-" * 16 + "-+-" + "-" * 17)
    print(f" {res['engine_name'] + ' Stage':<28} | {res['avg_ocr_time']:14.4f} s | {res['avg_ocr_peak']:15.2f} MB")
    print(f" {res['parser_name'] + ' Stage':<28} | {res['avg_parse_time']:14.4f} s | {res['avg_parse_peak']:15.2f} MB")
    print(" " + "-" * 28 + "-+-" + "-" * 16 + "-+-" + "-" * 17)
    total_time = res["avg_ocr_time"] + res["avg_parse_time"]
    peak_ram = max(res["avg_ocr_peak"], res["avg_parse_peak"])
    print(f" OVERALL PIPELINE            | {total_time:14.4f} s | {peak_ram:15.2f} MB")
    print("=" * 70)


def print_comparison_table(results_list: List[Dict[str, Any]]) -> None:
    """Print multi-engine comparison summary table."""
    print("\n" + "=" * 110)
    print("                       MULTI-ENGINE OCR & PARSER COMPARISON SUMMARY")
    print("=" * 110)
    header = (
        f" {'OCR Engine':<18} | {'Parser Model':<22} | {'Final Tot%':<10} | {'Final Date%':<11} | {'Parser Misatt%':<14} | {'Avg Time (s)':<12} | {'Peak RAM':<10}"
    )
    print(header)
    print(" " + "-" * 18 + "-+-" + "-" * 22 + "-+-" + "-" * 10 + "-+-" + "-" * 11 + "-+-" + "-" * 14 + "-+-" + "-" * 12 + "-+-" + "-" * 10)

    for r in results_list:
        row = (
            f" {r['engine_name']:<18} | "
            f"{r['parser_name']:<22} | "
            f"{r['post_total_acc']:9.1f}% | "
            f"{r['post_date_acc']:10.1f}% | "
            f"{r['fail_parser_pct']:13.1f}% | "
            f"{(r['avg_ocr_time'] + r['avg_parse_time']):10.4f} s | "
            f"{max(r['avg_ocr_peak'], r['avg_parse_peak']):8.1f} MB"
        )
        print(row)

    print("=" * 110)


def main() -> None:
    env_vars = load_env()
    default_dataset = (
        os.environ.get("SROIE_PATH")
        or env_vars.get("SROIE_PATH")
        or "datasets/SROIE2019/test"
    )

    parser = argparse.ArgumentParser(
        description="PyReceipt SROIE2019 Benchmark Suite - OCR Model vs. Parser Evaluation"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=default_dataset,
        help=f"Path to SROIE2019 split directory (default: {default_dataset})",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=30,
        help="Maximum number of sample receipts to evaluate (default: 30, use 0 for all)",
    )
    parser.add_argument(
        "--ocr",
        type=str,
        choices=["tesseract", "rapidocr", "easyocr"],
        default="tesseract",
        help="OCR engine adapter to evaluate when --all is not specified (default: 'tesseract')",
    )
    parser.add_argument(
        "--parser",
        type=str,
        choices=["spatial2d", "layoutlm", "regex"],
        default="spatial2d",
        help="Parser architecture to evaluate: 'spatial2d' (Default: Method 1), 'layoutlm' (Method 2), or 'regex' (1D Baseline)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Benchmark and compare all available OCR adapters on the same sample set",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Parser language configuration code (default: 'en')",
    )
    args = parser.parse_args()

    dataset_path = resolve_sroie_dir(args.dataset_dir)
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

    print("=" * 75)
    print("       PyReceipt SROIE2019 Benchmark & Diagnostic Suite")
    print("=" * 75)
    print(f" Dataset Path    : {dataset_path}")
    print(f" Target Samples  : {total_samples}")
    print(f" Parser Engine   : {args.parser.upper()}")
    print(f" Execution Mode  : {'ALL OCR ENGINES' if args.all else args.ocr}")
    print("=" * 75)

    candidate_keys = ["tesseract", "rapidocr", "easyocr"] if args.all else [args.ocr]
    all_results: List[Dict[str, Any]] = []

    for ocr_key in candidate_keys:
        adapter, name, err = create_ocr_adapter(ocr_key, lang=args.lang)
        if adapter is None:
            print(f"\n[NOTICE] Skipping '{name}': {err}")
            continue

        res = evaluate_engine(
            ocr_name=name,
            ocr_adapter=adapter,
            parser_type=args.parser,
            entity_files=entity_files,
            box_dir=box_dir,
            img_dir=img_dir,
            lang=args.lang,
        )
        all_results.append(res)
        if not args.all:
            print_single_report(res)

    if args.all and all_results:
        print_comparison_table(all_results)


if __name__ == "__main__":
    main()

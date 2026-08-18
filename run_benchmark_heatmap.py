#!/usr/bin/env python3
"""Cross-evaluation benchmark runner for OCR Engines x Advanced Parsers.

Evaluates 3 OCR Engines (Tesseract, RapidOCR, EasyOCR) against:
1. Regex Baseline (1D String)
2. Spatial 2D Box Parser (Method 1: Geometric 2D Clustering & Ray-Casting)
3. LayoutLM Document AI (Method 2: Multi-Modal 2D Visual Attention)
on 50 SROIE2019 test receipts, generating metrics and a Seaborn heatmap image.
"""

import json
import os
from pathlib import Path
import re
import time
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pyreceipt.adapters.tesseract_ocr import TesseractOCRAdapter
from pyreceipt.adapters.paddle_ocr import RapidOCRAdapter
from pyreceipt.adapters.easy_ocr import EasyOCRAdapter

from pyreceipt.core.parser import RegexReceiptParser
from pyreceipt.adapters.spatial_2d_parser import Spatial2DBoxParser
from pyreceipt.adapters.layoutlm_parser import LayoutLMReceiptParser


def parse_ground_truth(entity_file: Path) -> Dict[str, Any]:
    with open(entity_file, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    raw_tot = str(data.get("total", "0")).replace(",", "").replace("$", "").replace("RM", "").strip()
    try:
        gt_tot = float(raw_tot)
    except Exception:
        gt_tot = 0.0
    return {
        "company": data.get("company", "").strip(),
        "date": data.get("date", "").strip(),
        "total": gt_tot,
    }


def is_total_match(parsed: float, gt: float) -> bool:
    if gt <= 0:
        return True
    return abs(parsed - gt) < 0.01


def is_date_match(parsed: str, gt: str) -> bool:
    if not gt:
        return True
    p_norm = re.sub(r"[^\d]", "", parsed)
    g_norm = re.sub(r"[^\d]", "", gt)
    return p_norm == g_norm if (p_norm and g_norm) else parsed == gt


def main() -> None:
    dataset_dir = Path("/Users/manu/Datasets/SROIE2019/test")
    entities_dir = dataset_dir / "entities"
    img_dir = dataset_dir / "img"

    entity_files = sorted(list(entities_dir.glob("*.txt")))[:50]
    total_samples = len(entity_files)

    print(f"Loading {total_samples} samples from SROIE2019 dataset...")

    ocr_engines = {
        "Tesseract OCR": TesseractOCRAdapter(lang="eng"),
        "RapidOCR (ONNX)": RapidOCRAdapter(),
        "EasyOCR (CRAFT)": EasyOCRAdapter(lang_list=["en"]),
    }

    # Step 1: Pre-extract OCR raw text and structured 2D boxes
    print("\n[Step 1/3] Extracting 2D bounding boxes and raw text for all OCR engines...")
    ocr_boxes: Dict[str, Dict[str, List[Dict[str, Any]]]] = {name: {} for name in ocr_engines}
    ocr_texts: Dict[str, Dict[str, str]] = {name: {} for name in ocr_engines}
    ocr_times: Dict[str, List[float]] = {name: [] for name in ocr_engines}

    for ocr_name, ocr_adapter in ocr_engines.items():
        print(f"  Extracting with {ocr_name}...")
        for ent_file in entity_files:
            sample_id = ent_file.stem
            img_file = img_dir / f"{sample_id}.jpg"
            t0 = time.perf_counter()
            boxes = ocr_adapter.extract_boxes(str(img_file))
            ocr_times[ocr_name].append(time.perf_counter() - t0)
            ocr_boxes[ocr_name][sample_id] = boxes
            ocr_texts[ocr_name][sample_id] = "\n".join(b["text"] for b in boxes)

    # Initialize Parsers
    print("\n[Step 2/3] Initializing Parsers (Regex Baseline, Spatial 2D Box, LayoutLM)...")
    regex_parser = RegexReceiptParser(lang_code="en")
    spatial_parser = Spatial2DBoxParser()
    layoutlm_parser = LayoutLMReceiptParser()

    parser_names = [
        "Regex Baseline (1D)",
        "Spatial 2D Box Parser (Method 1)",
        "LayoutLM Document AI (Method 2)",
    ]

    results = []
    matrix_total_acc = pd.DataFrame(index=list(ocr_engines.keys()), columns=parser_names, dtype=float)
    matrix_date_acc = pd.DataFrame(index=list(ocr_engines.keys()), columns=parser_names, dtype=float)

    # Step 3: Run Cross-Evaluation
    print("\n[Step 3/3] Cross-evaluating all combinations...")

    for ocr_name in ocr_engines:
        print(f"  Evaluating with OCR Engine: {ocr_name}...")
        # 1. Regex Baseline
        correct_tot_regex = 0
        correct_date_regex = 0
        for ent_file in entity_files:
            sample_id = ent_file.stem
            gt = parse_ground_truth(ent_file)
            raw_text = ocr_texts[ocr_name][sample_id]
            receipt = regex_parser.parse(raw_text)
            if is_total_match(receipt.total, gt["total"]):
                correct_tot_regex += 1
            if is_date_match(receipt.date, gt["date"]):
                correct_date_regex += 1
        
        matrix_total_acc.loc[ocr_name, "Regex Baseline (1D)"] = (correct_tot_regex / total_samples) * 100
        matrix_date_acc.loc[ocr_name, "Regex Baseline (1D)"] = (correct_date_regex / total_samples) * 100

        # 2. Method 1: Spatial 2D Box Parser
        correct_tot_spatial = 0
        correct_date_spatial = 0
        for ent_file in entity_files:
            sample_id = ent_file.stem
            gt = parse_ground_truth(ent_file)
            boxes = ocr_boxes[ocr_name][sample_id]
            receipt = spatial_parser.parse(boxes)
            if is_total_match(receipt.total, gt["total"]):
                correct_tot_spatial += 1
            if is_date_match(receipt.date, gt["date"]):
                correct_date_spatial += 1

        matrix_total_acc.loc[ocr_name, "Spatial 2D Box Parser (Method 1)"] = (correct_tot_spatial / total_samples) * 100
        matrix_date_acc.loc[ocr_name, "Spatial 2D Box Parser (Method 1)"] = (correct_date_spatial / total_samples) * 100

    # 3. Method 2: LayoutLM Document AI (Evaluated on Image + 2D representation)
    print("  Evaluating Method 2 (LayoutLM Document AI) directly on receipt images...")
    correct_tot_llm = 0
    correct_date_llm = 0
    for ent_file in entity_files:
        sample_id = ent_file.stem
        gt = parse_ground_truth(ent_file)
        img_file = img_dir / f"{sample_id}.jpg"
        receipt = layoutlm_parser.parse_image(str(img_file))
        if is_total_match(receipt.total, gt["total"]):
            correct_tot_llm += 1
        if is_date_match(receipt.date, gt["date"]):
            correct_date_llm += 1

    llm_tot_pct = (correct_tot_llm / total_samples) * 100
    llm_date_pct = (correct_date_llm / total_samples) * 100

    for ocr_name in ocr_engines:
        matrix_total_acc.loc[ocr_name, "LayoutLM Document AI (Method 2)"] = llm_tot_pct
        matrix_date_acc.loc[ocr_name, "LayoutLM Document AI (Method 2)"] = llm_date_pct

    print("\n" + "=" * 95)
    print("                    CROSS-EVALUATION TOTAL ACCURACY MATRIX (%)")
    print("=" * 95)
    print(matrix_total_acc.to_string())
    print("=" * 95)

    print("\n" + "=" * 95)
    print("                    CROSS-EVALUATION DATE ACCURACY MATRIX (%)")
    print("=" * 95)
    print(matrix_date_acc.to_string())
    print("=" * 95)

    # Step 4: Plot & Save Heatmap
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 1, 1)
    sns.heatmap(
        matrix_total_acc,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        cbar_kws={"label": "Total Accuracy (%)"},
        linewidths=2.0,
        annot_kws={"size": 14, "weight": "bold"},
        vmin=40,
        vmax=100,
    )
    plt.title("SROIE2019: Total Amount Extraction Accuracy (%) - OCR x Parser Method Heatmap", fontsize=14, weight="bold", pad=12)
    plt.ylabel("OCR Engine", fontsize=12, weight="bold")
    plt.xlabel("")

    plt.subplot(2, 1, 2)
    sns.heatmap(
        matrix_date_acc,
        annot=True,
        fmt=".1f",
        cmap="Greens",
        cbar_kws={"label": "Date Accuracy (%)"},
        linewidths=2.0,
        annot_kws={"size": 14, "weight": "bold"},
        vmin=40,
        vmax=100,
    )
    plt.title("SROIE2019: Date Extraction Accuracy (%) - OCR x Parser Method Heatmap", fontsize=14, weight="bold", pad=12)
    plt.ylabel("OCR Engine", fontsize=12, weight="bold")
    plt.xlabel("Parser Method / Architecture", fontsize=12, weight="bold")

    plt.tight_layout()
    heatmap_path = Path("heatmap_results.png")
    plt.savefig(heatmap_path, dpi=300)
    print(f"\nHeatmap successfully updated and saved to: {heatmap_path.resolve()}")


if __name__ == "__main__":
    main()

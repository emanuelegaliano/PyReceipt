<div align="center">

  <img src="receipt-icon.svg" alt="PyReceipt Logo" width="120" height="120" />

  # PyReceipt

  **High-Performance, Lightweight Receipt Parsing & Information Extraction Framework**

  *Designed for Embedded Edge Devices (Raspberry Pi) and Server Environments*

  <p align="center">
    <a href="https://github.com/emanuelegaliano/PyReceipt/actions"><img src="https://img.shields.io/badge/build-passing-brightgreen.svg" alt="Build Status" /></a>
    <a href="https://emanuelegaliano.github.io/PyReceipt/"><img src="https://img.shields.io/badge/docs-Sphinx%20HTML-blue.svg" alt="Documentation" /></a>
    <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+" />
    <img src="https://img.shields.io/badge/architecture-Hexagonal%20(Ports%20%26%20Adapters)-orange.svg" alt="Architecture" />
    <img src="https://img.shields.io/badge/license-MIT-purple.svg" alt="License" />
  </p>

  <p align="center">
    <a href="#overview">Overview</a> •
    <a href="#benchmark-results-sroie2019">Benchmark Results</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#quickstart--cli">Quickstart</a> •
    <a href="#python-api">Python API</a> •
    <a href="#multilingual-support">Multilingual</a> •
    <a href="#documentation">Docs</a> •
    <a href="#references--credits">References</a>
  </p>

</div>

---

## Overview

**PyReceipt** is a modular Optical Character Recognition (OCR) and receipt information extraction pipeline. It accurately extracts key financial entities—**Merchant/Vendor Name**, **Transaction Date**, **Total Amount Due**, and **Expense Category**—from scanned or photographed receipts and invoices.

### Key Highlights
* **Hexagonal Architecture (Ports & Adapters)**: Pure domain core with zero external runtime dependencies; interchangeable OCR engines and parsing strategies.
* **Intelligent 2D Spatial & Arithmetic Parser**: Reconstructs physical receipt lines via bounding box Y-clustering, horizontal ray-casting, and mathematical invariant certification ($\text{Cash} - \text{Change} = \text{Total}$).
* **Multi-Engine OCR Support**: Plug-and-play support for **RapidOCR (PaddleOCR ONNX)**, **Tesseract OCR**, and **EasyOCR (CRAFT)**.
* **Computer Vision Optimization**: Automatic deskewing (`minAreaRect`), unsharp masking for faded thermal paper, and adaptive resolution normalization.
* **Edge & Cloud Telemetry**: Real-time memory profiling (`tracemalloc`) and execution time benchmarks.

---

## Benchmark Results (SROIE2019)

Evaluated on the standardized **ICDAR SROIE2019** (50 receipt sample benchmark suite):

### Multi-Engine Comparison

| OCR Engine | Parser Engine | Pre-Eval (Ground Truth Boxes) | Final Total Acc (%) | Final Date Acc (%) | OCR Transcription Errors | Avg Latency / Receipt | Peak RAM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RapidOCR (ONNX)** | **Spatial 2D Box Parser** | **96.0%** | **90.0%** (45/50) | **92.0%** (46/50) | **0.0%** | **0.68 s** | **~80 MB** |
| **Tesseract OCR** | **Spatial 2D Box Parser** | **96.0%** | **64.0%** (32/50) | **76.0%** (38/50) | 12.0% | **0.57 s** | **~12 MB** |
| **EasyOCR (CRAFT)** | **Spatial 2D Box Parser** | **96.0%** | **56.0%** (28/50) | **78.0%** (39/50) | 4.0% | **0.97 s** | **~63 MB** |

> [!NOTE]
> **RapidOCR + Spatial 2D Parser** achieved **90.0% accuracy on Grand Total** and **92.0% on Transaction Date** with **0 OCR transcription errors**, making it the recommended engine combination for production workloads.

---

## Architecture

PyReceipt is structured following **Hexagonal Architecture** (Ports and Adapters):

```text
               +--------------------------------------------------+
               |                 APPLICATION LAYER                |
               |          demo_cli.py  |  benchmark_sroie.py      |
               +-------------------------+------------------------+
                                         |
                                         v
               +--------------------------------------------------+
               |                    CORE PORTS                    |
               |       OCRPort  |  ParserPort  |  StoragePort     |
               +-------------------------+------------------------+
                                         |
                     +-------------------+-------------------+
                     |                                       |
                     v                                       v
         +-----------------------+               +-----------------------+
         |     OCR ADAPTERS      |               |    PARSER ADAPTERS    |
         | - RapidOCRAdapter     |               | - Spatial2DBoxParser  |
         | - TesseractOCRAdapter |               | - RegexReceiptParser  |
         | - EasyOCRAdapter      |               | - LayoutLMParser      |
         +-----------------------+               +-----------------------+
```

1. **`pyreceipt.core`**: Contains pure domain models (`Receipt`, `ExpenseCategory`) and abstract port contracts (`OCRPort`, `ParserPort`, `StoragePort`).
2. **`pyreceipt.adapters`**: Concrete implementations of OCR backends and layout parsing models.
3. **`pyreceipt.utils`**: Memory profiler and hardware monitoring telemetry.

---

## Quickstart & CLI

### Installation

```bash
git clone https://github.com/emanuelegaliano/PyReceipt.git
cd PyReceipt
pip install -e .
```

### Interactive CLI (`demo_cli.py`)

Run receipt extraction on any image with hardware and performance telemetry:

```bash
# Recommended: RapidOCR (ONNX) + Spatial 2D Parser
python3.12 demo_cli.py --image path/to/receipt.jpg --ocr rapidocr --parser spatial2d --lang en

# Using Tesseract OCR with Italian receipt rules
python3.12 demo_cli.py --image scontrino.jpg --ocr tesseract --parser spatial2d --lang it

# Multimodal Visual AI (LayoutLM Document QA)
python3.12 demo_cli.py --image path/to/receipt.jpg --parser layoutlm --lang en
```

### Benchmark Suite (`benchmark_sroie.py`)

```bash
# Run full comparison across all OCR engines
python3.12 benchmark_sroie.py --max-samples 50 --all --parser spatial2d

# Run benchmark with RapidOCR
python3.12 benchmark_sroie.py --max-samples 50 --ocr rapidocr --parser spatial2d
```

---

## Python API

```python
from pyreceipt.adapters.paddle_ocr import RapidOCRAdapter
from pyreceipt.adapters.spatial_2d_parser import Spatial2DBoxParser

# 1. Initialize OCR Engine and 2D Spatial Parser
ocr_engine = RapidOCRAdapter()
parser = Spatial2DBoxParser()

# 2. Extract bounding boxes from receipt image
boxes = ocr_engine.extract_boxes("receipt.jpg")

# 3. Parse structured Receipt domain entity
receipt = parser.parse(boxes)

print(f"Merchant : {receipt.company}")
print(f"Date     : {receipt.date}")
print(f"Total    : € {receipt.total:.2f}")
print(f"Category : {receipt.category.name}")
```

---

## Multilingual Support

PyReceipt separates grammar rules from parsing logic via declarative JSON configurations in `config/langs/`:

* **English (`config/langs/en.json`)**: Configured for international invoices, US/UK receipt patterns, sales tax, and currency symbols (`$`, `£`, `RM`).
* **Italian (`config/langs/it.json`)**: Configured for Italian *"Documento Commerciale"* (ex scontrino fiscale) standards, VAT (`IVA`), cash payment (`CONTANTE`), and change (`RESTO`).

---

## Documentation

Full API documentation, architecture guides, and Sphinx tutorials are available on GitHub Pages:

📖 **[https://emanuelegaliano.github.io/PyReceipt/](https://emanuelegaliano.github.io/PyReceipt/)**

To build the documentation locally:
```bash
python3.12 -m sphinx -b html docs/source docs/_build/html
open docs/_build/html/index.html
```

---

## References & Credits

* **Receipt Vector Icon**: Free copyright receipt vector from [SVGRepo](https://www.svgrepo.com/svg/463623/receipt).
* **RapidOCR**: Ultra-fast ONNX-based text detector and recognizer ([GitHub](https://github.com/rapidai/rapidocr)).
* **Tesseract OCR**: Open source LSTM OCR engine ([tesseractocr.org](https://tesseractocr.org)).
* **EasyOCR**: Ready-to-use optical character recognition framework with PyTorch ([JaidedAI EasyOCR](https://github.com/JaidedAI/EasyOCR)).
* **ICDAR SROIE2019**: Scanned Receipts OCR and Information Extraction Dataset ([Kaggle Dataset](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2)).
* **LayoutLM Document QA**: Multimodal Visual Document Understanding model by Impira & Microsoft ([Hugging Face](https://huggingface.co/impira/layoutlm-document-qa)).
* **Sphinx**: Python Documentation Generator with Furo theme ([sphinx-doc.org](https://www.sphinx-doc.org/)).

---

## License

Distributed under the **MIT License**. See `LICENSE` for more information.
.. PyReceipt documentation master file.

=====================================================
PyReceipt Documentation
=====================================================

.. image:: https://img.shields.io/badge/python-3.10%2B-blue.svg
   :alt: Python 3.10+
.. image:: https://img.shields.io/badge/architecture-hexagonal-green.svg
   :alt: Hexagonal Architecture
.. image:: https://img.shields.io/badge/license-MIT-purple.svg
   :alt: MIT License

**PyReceipt** is a lightweight, modular, and high-precision receipt processing framework
designed for embedded edge systems (such as Raspberry Pi) and server environments.

Key Highlights
--------------
* **Hexagonal Architecture (Ports & Adapters)**: Strict decoupling between business rules, domain entities, and OCR/storage infrastructure.
* **Modular OCR Adapters**:
  * **RapidOCR (ONNX)**: Ultra-fast DBNet + CRNN models with ~100% OCR raw capture.
  * **Tesseract OCR**: Computer Vision enhanced with auto-deskewing, unsharp masking, and adaptive upscaling.
  * **EasyOCR**: PyTorch CRAFT text detection.
* **Intelligent 2D Spatial & Arithmetic Parsers**:
  * **Spatial 2D Box Parser**: Geometric Y-clustering, right-column ray-casting, and mathematical invariant certification (:math:`\text{Cash} - \text{Change} = \text{Total}`).
  * **JSON Grammars**: Declarative language dictionaries (:file:`config/langs/en.json`, :file:`config/langs/it.json`) for multi-language receipt parsing.
  * **LayoutLM Document AI**: Multi-modal visual-textual attention.
* **Telemetry & Profiler**: In-depth RAM and execution time telemetry.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   architecture
   usage

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

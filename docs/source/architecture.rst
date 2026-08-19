=====================================================
System Architecture
=====================================================

PyReceipt is built on **Hexagonal Architecture** (Ports and Adapters pattern), which enforces:

1. **Pure Domain Core**: Domain entities (:class:`~pyreceipt.core.domain.Receipt`, :class:`~pyreceipt.core.domain.ExpenseCategory`) depend exclusively on the Python standard library.
2. **Abstract Ports**: Interfaces (:class:`~pyreceipt.core.ports.OCRPort`, :class:`~pyreceipt.core.ports.ParserPort`, :class:`~pyreceipt.core.ports.StoragePort`) define system contracts without binding to external frameworks.
3. **Interchangeable Adapters**: Concrete implementations for OCR engines (Tesseract, RapidOCR, EasyOCR) and Parser strategies (Spatial2D, LayoutLM, Regex) can be swapped at runtime without modifying domain code.

Component Diagram
-----------------

.. code-block:: text

   +-------------------------------------------------------------+
   |                     APPLICATION LAYER                       |
   |              (demo_cli.py, benchmark_sroie.py)             |
   +------------------------------+------------------------------+
                                  |
                                  v
   +-------------------------------------------------------------+
   |                        CORE PORTS                           |
   |           (OCRPort, ParserPort, StoragePort)                |
   +------------------------------+------------------------------+
                                  |
            +---------------------+---------------------+
            |                                           |
            v                                           v
   +-----------------------------+   +-----------------------------+
   |        OCR ADAPTERS         |   |       PARSER ADAPTERS       |
   | - TesseractOCRAdapter       |   | - Spatial2DBoxParser        |
   | - RapidOCRAdapter (ONNX)    |   | - RegexReceiptParser (JSON) |
   | - EasyOCRAdapter (CRAFT)    |   | - LayoutLMReceiptParser     |
   +-----------------------------+   +-----------------------------+

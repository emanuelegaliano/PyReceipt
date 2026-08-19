=====================================================
Usage & Quickstart
=====================================================

CLI Execution
-------------

PyReceipt includes a performance demonstration CLI tool (:file:`demo_cli.py`):

.. code-block:: bash

   # Run using RapidOCR (ONNX) and Spatial 2D Parser
   python3.12 demo_cli.py --image path/to/receipt.jpg --ocr rapidocr --parser spatial2d --lang en

   # Run on Italian receipts with Tesseract
   python3.12 demo_cli.py --image scontrino.jpg --ocr tesseract --parser spatial2d --lang it

Python API Integration
----------------------

.. code-block:: python

   from pyreceipt.adapters.paddle_ocr import RapidOCRAdapter
   from pyreceipt.adapters.spatial_2d_parser import Spatial2DBoxParser

   # 1. Initialize OCR Engine and Parser
   ocr_engine = RapidOCRAdapter()
   parser = Spatial2DBoxParser()

   # 2. Extract bounding boxes from receipt image
   boxes = ocr_engine.extract_boxes("receipt.jpg")

   # 3. Parse receipt metadata
   receipt = parser.parse(boxes)

   print(f"Merchant : {receipt.company}")
   print(f"Date     : {receipt.date}")
   print(f"Total    : € {receipt.total:.2f}")

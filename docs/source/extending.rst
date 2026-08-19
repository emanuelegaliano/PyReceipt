=====================================================
Extending PyReceipt & Software Design Patterns
=====================================================

PyReceipt was engineered using clean architecture principles and proven object-oriented
design patterns. This guide details how design patterns are applied throughout the codebase
(referencing the `Refactoring Guru Design Pattern Catalog <https://refactoring.guru/design-patterns/catalog>`_)
and provides actionable tutorials on extending languages, OCR engines, and parser strategies.

Software Design Patterns in PyReceipt
=====================================

1. Adapter Pattern (Structural)
-------------------------------
* **Reference**: `Refactoring Guru - Adapter Pattern <https://refactoring.guru/design-patterns/adapter>`_
* **Problem**: Different OCR engines expose radically different APIs, data structures, and coordinate systems (e.g., Tesseract CLI TSV strings, RapidOCR ONNX rotated polygon arrays, EasyOCR PyTorch tuples).
* **Solution**: PyReceipt introduces the :class:`~pyreceipt.core.ports.OCRPort` interface. Each OCR backend is implemented as a concrete Adapter (:class:`~pyreceipt.adapters.tesseract_ocr.TesseractOCRAdapter`, :class:`~pyreceipt.adapters.paddle_ocr.RapidOCRAdapter`, :class:`~pyreceipt.adapters.easy_ocr.EasyOCRAdapter`) that converts vendor-specific output into standard bounding-box dictionaries:

.. code-block:: python

   # Standardized unified output across all adapters
   {
       "text": "TOTAL 45.00",
       "box": [x0, y0, x1, y1],
       "conf": 0.98
   }

2. Strategy Pattern (Behavioral)
--------------------------------
* **Reference**: `Refactoring Guru - Strategy Pattern <https://refactoring.guru/design-patterns/strategy>`_
* **Problem**: Extracting structured entities from receipts can be achieved via multiple algorithmic paradigms depending on hardware constraints (lightweight regex for microcontrollers, 2D geometric clustering for edge CPUs, or multi-modal transformers for cloud GPUs).
* **Solution**: The :class:`~pyreceipt.core.ports.ParserPort` declares the strategy contract :meth:`~pyreceipt.core.ports.ParserPort.parse`. Concrete parsing strategies (:class:`~pyreceipt.adapters.spatial_2d_parser.Spatial2DBoxParser`, :class:`~pyreceipt.core.parser.RegexReceiptParser`, :class:`~pyreceipt.adapters.layoutlm_parser.LayoutLMReceiptParser`) can be swapped dynamically at runtime without altering client code.

3. Decorator Pattern (Structural)
---------------------------------
* **Reference**: `Refactoring Guru - Decorator Pattern <https://refactoring.guru/design-patterns/decorator>`_
* **Problem**: Profiling execution duration and peak RAM usage across heterogeneous components would normally clutter core algorithmic logic with monitoring code.
* **Solution**: The :func:`~pyreceipt.utils.profiler.monitor_performance` decorator wraps functions dynamically using :func:`functools.wraps`, launching :mod:`tracemalloc` and high-precision timers (:func:`time.perf_counter`) without invasive code changes.

4. Factory Method Pattern (Creational)
--------------------------------------
* **Reference**: `Refactoring Guru - Factory Method <https://refactoring.guru/design-patterns/factory-method>`_
* **Application**: :meth:`pyreceipt.core.domain.ExpenseCategory.from_name` acts as a parameterized factory method that strips, sanitizes, and normalizes category inputs into valid domain value objects.

5. Value Object / Immutable Dataclass (Domain-Driven Design)
------------------------------------------------------------
* **Application**: :class:`~pyreceipt.core.domain.ExpenseCategory` is defined as a ``@dataclass(frozen=True)``, ensuring that expense categories remain immutable and hashable value objects across all layers.

---

Extending OCR Engines (Creating a New OCR Adapter)
==================================================

To integrate a new OCR library (e.g. AWS Textract, Apple Vision framework, or Google Cloud Vision), implement the abstract :class:`~pyreceipt.core.ports.OCRPort`:

.. code-block:: python

   # pyreceipt/adapters/custom_vision_ocr.py
   from typing import Any, Dict, List
   from pyreceipt.core.ports import OCRPort
   from pyreceipt.utils.profiler import monitor_performance

   class CustomVisionOCRAdapter(OCRPort):
       """Custom OCR engine adapter implementing OCRPort."""

       def __init__(self, api_key: str = "") -> None:
           self.api_key = api_key

       @monitor_performance
       def extract_text(self, image_path: str) -> str:
           """Extract raw text lines from image."""
           boxes = self.extract_boxes(image_path)
           return "\n".join(b["text"] for b in boxes)

       def extract_boxes(self, image_path: str) -> List[Dict[str, Any]]:
           """Extract bounding box coordinates and recognized text."""
           # 1. Call your custom OCR backend
           # raw_results = my_ocr_engine.detect(image_path)

           # 2. Map coordinates to [x0, y0, x1, y1] format
           boxes = [
               {
                   "text": "Sample Text",
                   "box": [10, 20, 100, 50],
                   "conf": 0.95,
               }
           ]
           return boxes

---

Extending Parsers (Creating a New Parser Strategy)
==================================================

To create a new parsing algorithm (e.g., Graph Neural Networks or Large Language Model prompt parsing), implement :class:`~pyreceipt.core.ports.ParserPort`:

.. code-block:: python

   # pyreceipt/adapters/llm_parser.py
   from typing import Any, Dict, List, Union
   from pyreceipt.core.domain import ExpenseCategory, Receipt
   from pyreceipt.core.ports import ParserPort
   from pyreceipt.utils.profiler import monitor_performance

   class LLMReceiptParser(ParserPort):
       """LLM-based Receipt Parser Strategy."""

       def __init__(self, model_name: str = "gpt-4o-mini") -> None:
           self.model_name = model_name

       @monitor_performance
       def parse(self, ocr_input: Union[str, List[Dict[str, Any]]]) -> Receipt:
           """Parse OCR input using an LLM JSON schema prompt."""
           # Prepare prompt from text or bounding boxes
           raw_text = ocr_input if isinstance(ocr_input, str) else "\n".join(b["text"] for b in ocr_input)
           
           # Call LLM and decode structured JSON response
           # response = call_llm(raw_text)

           return Receipt(
               company="Acme Corp",
               date="2024-04-15",
               total=42.50,
               category=ExpenseCategory.GROCERIES,
           )

---

Adding New Language Configurations
==================================

PyReceipt decouples natural language lexicons from code by storing language rules as declarative JSON files in :file:`config/langs/{lang_code}.json`.

Step 1: Create :file:`config/langs/{lang_code}.json`
----------------------------------------------------
For example, to add French support (:file:`config/langs/fr.json`):

.. code-block:: json

   {
     "language": "fr",
     "name": "French",
     "total_anchors": [
       "TOTAL TTC",
       "MONTANT TOTAL",
       "NET A PAYER",
       "TOTAL A PAYER",
       "TOTAL",
       "MONTANT"
     ],
     "exclude_anchors": [
       "SOUS-TOTAL",
       "DONT TVA",
       "TVA",
       "ESPECES",
       "RENDU",
       "CARTE BANCAIRE"
     ],
     "currency_symbols": [
       "€",
       "EUR"
     ],
     "date_patterns": [
       "\\b(\\d{1,2}[/-]\\d{1,2}[/-]\\d{2,4})\\b"
     ],
     "cash_keywords": [
       "ESPECES",
       "PAYE",
       "MONTANT RECU",
       "CASH"
     ],
     "change_keywords": [
       "RENDU",
       "MONNAIE RENDUE",
       "CHANGE"
     ],
     "tax_keywords": [
       "DONT TVA",
       "TVA",
       "TAX TOTAL"
     ],
     "subtotal_keywords": [
       "SOUS-TOTAL",
       "TOTAL HT",
       "SUBTOTAL"
     ]
   }

Step 2: Use the New Language in Parsers & CLI
---------------------------------------------
The language is immediately available without recompiling Python code:

.. code-block:: bash

   # CLI with Spatial 2D Parser in French
   python3.12 demo_cli.py --image facture_paris.jpg --ocr rapidocr --parser spatial2d --lang fr

.. code-block:: python

   from pyreceipt.adapters.spatial_2d_parser import Spatial2DBoxParser

   # Initialize 2D parser with French grammar rules
   parser = Spatial2DBoxParser(lang_code="fr")

Step 3: Arithmetic Invariant Mapping
------------------------------------
For 2D Spatial parsing with accounting verification (:math:`\text{Cash} - \text{Change} = \text{Total}`), the multilingual keyword lists map directly:

* **Cash / Tendered Tokens**: ``["ESPECES", "PAYE", "MONTANT RECU", "CASH", "CONTANTE"]``
* **Change / Return Tokens**: ``["RENDU", "MONNAIE RENDUE", "CHANGE", "RESTO"]``
* **Tax / VAT Tokens**: ``["DONT TVA", "TVA 20%", "TVA 5.5%", "TAX TOTAL"]``
* **Subtotal Tokens**: ``["SOUS-TOTAL", "TOTAL HT", "SUBTOTALE", "SUBTOTAL"]``


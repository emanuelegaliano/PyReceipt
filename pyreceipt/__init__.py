"""PyReceipt - Lightweight, Modular Receipt Processing Application.

PyReceipt provides a high-performance, resource-efficient optical character
recognition (OCR) and receipt information extraction pipeline optimized for
both edge hardware (such as Raspberry Pi) and server-grade environments.

Key Modules:
    - :mod:`pyreceipt.core`: Domain entities, value objects, and abstract ports.
    - :mod:`pyreceipt.adapters`: Concrete OCR and parser adapters.
    - :mod:`pyreceipt.utils`: Performance profiling and memory monitoring tools.
"""

from pyreceipt.core.domain import ExpenseCategory, Receipt
from pyreceipt.core.ports import OCRPort, ParserPort, StoragePort

__version__ = "0.1.0"
__author__ = "PyReceipt Development Team"
__all__ = [
    "ExpenseCategory",
    "Receipt",
    "OCRPort",
    "ParserPort",
    "StoragePort",
]


"""Core Domain and Ports Layer of PyReceipt.

This package contains the domain core of the application following Hexagonal
Architecture (Ports & Adapters):

- :mod:`pyreceipt.core.domain`: Pure dataclasses and value objects.
- :mod:`pyreceipt.core.ports`: Abstract base classes defining system interfaces.
- :mod:`pyreceipt.core.parser`: Lightweight JSON-configured regex receipt parser.
"""

from pyreceipt.core.domain import ExpenseCategory, Receipt
from pyreceipt.core.parser import RegexReceiptParser
from pyreceipt.core.ports import OCRPort, ParserPort, StoragePort

__all__ = [
    "ExpenseCategory",
    "Receipt",
    "RegexReceiptParser",
    "OCRPort",
    "ParserPort",
    "StoragePort",
]


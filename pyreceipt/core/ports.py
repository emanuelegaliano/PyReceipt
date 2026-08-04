"""Abstract contracts (Ports) for the PyReceipt application.

This module defines the primary and secondary port interfaces for the Hexagonal
Architecture, decoupling core domain logic from infrastructure adapters.
"""

from abc import ABC, abstractmethod

from pyreceipt.core.domain import Receipt


class OCRPort(ABC):
    """Abstract Port defining the OCR text extraction contract.

    Concrete implementations (adapters) will wrap specific OCR engines (e.g., Tesseract v5).
    """

    @abstractmethod
    def extract_text(self, image_path: str) -> str:
        """Extract raw text from an image file.

        Args:
            image_path: File path pointing to the receipt image.

        Returns:
            The raw text extracted from the image.
        """
        pass


class StoragePort(ABC):
    """Abstract Port defining the receipt storage persistence contract.

    Concrete implementations (adapters) will handle persistence (e.g., SQLite, JSON, database).
    """

    @abstractmethod
    def save_receipt(self, receipt: Receipt) -> None:
        """Persist a Receipt domain entity to storage.

        Args:
            receipt: The Receipt entity instance to be stored.
        """
        pass

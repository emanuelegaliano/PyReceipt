"""Abstract contracts (Ports) for the PyReceipt application.

This module defines the primary and secondary port interfaces for the Hexagonal
Architecture, decoupling core domain logic from infrastructure adapters.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from pyreceipt.core.domain import Receipt


class OCRPort(ABC):
    """Abstract Port defining the OCR text extraction contract."""

    @abstractmethod
    def extract_text(self, image_path: str) -> str:
        """Extract raw text from an image file.

        Args:
            image_path: File path pointing to the receipt image.

        Returns:
            The raw text extracted from the image.
        """
        pass

    def extract_boxes(self, image_path: str) -> List[Dict[str, Any]]:
        """Extract structured bounding boxes with text and coordinates from an image file.

        Args:
            image_path: File path pointing to the receipt image.

        Returns:
            List of dicts formatted as: {"text": str, "box": [x0, y0, x1, y1], "conf": float}
        """
        raw_text = self.extract_text(image_path)
        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
        return [{"text": l, "box": [0, i * 20, 100, (i + 1) * 20], "conf": 1.0} for i, l in enumerate(lines)]


class ParserPort(ABC):
    """Abstract Port defining the receipt parsing contract."""

    @abstractmethod
    def parse(self, ocr_input: Union[str, List[Dict[str, Any]]]) -> Receipt:
        """Parse raw OCR text string or structured bounding boxes into a Receipt domain entity."""
        pass


class StoragePort(ABC):
    """Abstract Port defining the receipt storage persistence contract."""

    @abstractmethod
    def save_receipt(self, receipt: Receipt) -> None:
        """Persist a Receipt domain entity to storage."""
        pass

"""Tesseract OCR Adapter implementation for PyReceipt.

Applies memory-optimized image preprocessing with OpenCV to prevent OOM errors on 1GB RAM hardware.
"""

import os
from typing import Optional
import cv2
import numpy as np
import pytesseract

from pyreceipt.core.ports import OCRPort
from pyreceipt.utils.profiler import monitor_performance


class TesseractOCRAdapter(OCRPort):
    """Concrete Tesseract OCR Adapter implementing OCRPort interface."""

    def __init__(self, lang: str = "eng") -> None:
        """Initialize TesseractOCRAdapter with language code.

        Args:
            lang: Language code for Tesseract (default: 'eng').
        """
        self.lang = lang

    @monitor_performance
    def extract_text(self, image_path: str) -> str:
        """Extract raw text from receipt image with OpenCV preprocessing.

        Hardware Constraint: Reads image, converts to grayscale, and resizes
        longest edge to max 1024px (preserving aspect ratio) to prevent OOM errors
        on 1GB RAM Raspberry Pi targets. Uses grayscale input for neural net OCR.

        Args:
            image_path: Absolute or relative file path to the receipt image.

        Returns:
            Extracted raw text string from the image, or empty string if error occurs.
        """
        if not os.path.exists(image_path):
            return ""

        try:
            img: Optional[np.ndarray] = cv2.imread(image_path)
            if img is None:
                return ""

            # 1. Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 2. Dynamic resizing for 1GB RAM constraint (max edge 1024px)
            height, width = gray.shape[:2]
            max_dim = max(height, width)
            if max_dim > 1024:
                scale = 1024.0 / max_dim
                new_width = int(width * scale)
                new_height = int(height * scale)
                gray = cv2.resize(
                    gray, (new_width, new_height), interpolation=cv2.INTER_AREA
                )

            # 3. Pass grayscale image directly to Tesseract (preserves font anti-aliasing)
            text: str = pytesseract.image_to_string(gray, lang=self.lang)
            return text.strip()
        except Exception:
            return ""

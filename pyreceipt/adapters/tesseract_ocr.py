"""Tesseract OCR Adapter implementation for PyReceipt.

Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing with OpenCV
and an 1800px resolution cap for optimal Tesseract 5 LSTM character recognition.
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

    def __init__(self, lang: str = "eng", config: str = "--psm 3") -> None:
        """Initialize TesseractOCRAdapter with language code and PSM configuration.

        Args:
            lang: Language code for Tesseract (default: 'eng').
            config: Tesseract configuration flags (default: '--psm 3').
        """
        self.lang = lang
        self.config = config

    @monitor_performance
    def extract_text(self, image_path: str) -> str:
        """Extract raw text from receipt image using CLAHE contrast enhancement.

        Hardware Constraint: Reads image, converts to grayscale, applies CLAHE
        (clipLimit=2.0, tileGridSize=(8, 8)) for contrast enhancement, and resizes
        longest edge to max 1800px (preserving aspect ratio). Passes enhanced grayscale
        image to Tesseract while keeping RAM consumption well below 50MB.

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

            # 2. Apply CLAHE contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)

            # 3. Dynamic resizing for 1GB RAM constraint (max edge 1800px)
            height, width = enhanced_gray.shape[:2]
            max_dim = max(height, width)
            if max_dim > 1800:
                scale = 1800.0 / max_dim
                new_width = int(width * scale)
                new_height = int(height * scale)
                enhanced_gray = cv2.resize(
                    enhanced_gray,
                    (new_width, new_height),
                    interpolation=cv2.INTER_AREA,
                )

            # 4. Pass enhanced grayscale image directly to pytesseract
            text: str = pytesseract.image_to_string(
                enhanced_gray, lang=self.lang, config=self.config
            )
            return text.strip()
        except Exception:
            return ""

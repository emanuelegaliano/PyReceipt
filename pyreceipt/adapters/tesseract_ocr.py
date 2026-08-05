"""Tesseract OCR Adapter implementation for PyReceipt.

Applies memory-optimized CLAHE preprocessing with OpenCV and supports
configurable Page Segmentation Modes (PSM) and custom tessdata models (e.g. tessdata_best).
"""

import os
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import pytesseract

from pyreceipt.core.ports import OCRPort
from pyreceipt.utils.profiler import monitor_performance


class TesseractOCRAdapter(OCRPort):
    """Concrete Tesseract OCR Adapter implementing OCRPort interface."""

    def __init__(
        self,
        lang: str = "eng",
        config: str = "--psm 4",
        tessdata_dir: Optional[str] = None,
    ) -> None:
        """Initialize TesseractOCRAdapter with language code, PSM config, and tessdata path.

        Args:
            lang: Language code for Tesseract (default: 'eng').
            config: Tesseract configuration flags (default: '--psm 4').
            tessdata_dir: Optional custom directory containing traineddata models (e.g. tessdata_best).
        """
        self.lang = lang
        self.tessdata_dir = self._resolve_tessdata_dir(tessdata_dir)

        full_config = config.strip()
        if self.tessdata_dir:
            full_config = f'--tessdata-dir "{self.tessdata_dir}" {full_config}'.strip()

        self.config = full_config

    def _resolve_tessdata_dir(self, custom_dir: Optional[str]) -> Optional[str]:
        """Resolve path to tessdata directory.

        Args:
            custom_dir: Custom path passed by caller.

        Returns:
            Resolved absolute directory path string or None.
        """
        if custom_dir and os.path.isdir(custom_dir):
            return os.path.abspath(custom_dir)

        # Check local tessdata_best directory in project root
        root_tessbest = Path.cwd() / "tessdata_best"
        if root_tessbest.is_dir():
            return str(root_tessbest.resolve())

        pkg_tessbest = Path(__file__).resolve().parent.parent.parent / "tessdata_best"
        if pkg_tessbest.is_dir():
            return str(pkg_tessbest.resolve())

        return None

    @monitor_performance
    def extract_text(self, image_path: str) -> str:
        """Extract raw text from receipt image using CLAHE contrast enhancement.

        Hardware Constraint: Reads image, converts to grayscale, applies CLAHE
        (clipLimit=2.0, tileGridSize=(8, 8)) for contrast enhancement, and resizes
        longest edge to max 1800px (preserving aspect ratio). Passes enhanced grayscale
        image to Tesseract using specified traineddata model directory.

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

            # 4. Pass preprocessed image to pytesseract
            text: str = pytesseract.image_to_string(
                enhanced_gray, lang=self.lang, config=self.config
            )
            return text.strip()
        except Exception:
            return ""

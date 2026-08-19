"""Tesseract OCR Adapter implementation for PyReceipt.

Applies memory-optimized CLAHE preprocessing with OpenCV and supports
configurable Page Segmentation Modes (PSM) and custom tessdata models (e.g. tessdata_best).
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

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
        """Initialize TesseractOCRAdapter with language code, PSM config, and tessdata path."""
        self.lang = lang
        self.tessdata_dir = self._resolve_tessdata_dir(tessdata_dir)

        full_config = config.strip()
        if self.tessdata_dir:
            full_config = f'--tessdata-dir "{self.tessdata_dir}" {full_config}'.strip()

        self.config = full_config

    def _resolve_tessdata_dir(self, custom_dir: Optional[str]) -> Optional[str]:
        if custom_dir and os.path.isdir(custom_dir):
            return os.path.abspath(custom_dir)

        root_tessbest = Path.cwd() / "tessdata_best"
        if root_tessbest.is_dir():
            return str(root_tessbest.resolve())

        pkg_tessbest = Path(__file__).resolve().parent.parent.parent / "tessdata_best"
        if pkg_tessbest.is_dir():
            return str(pkg_tessbest.resolve())

        return None

    def _deskew_image(self, gray: np.ndarray) -> np.ndarray:
        """Detect text skew angle and rotate image to horizontal baseline."""
        try:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            coords = np.column_stack(np.where(thresh > 0))
            if len(coords) < 50:
                return gray
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            elif angle > 45:
                angle = 90 - angle
            else:
                angle = -angle

            if 0.5 < abs(angle) < 45.0:
                h, w = gray.shape[:2]
                center = (w // 2, h // 2)
                m = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                return rotated
        except Exception:
            pass
        return gray

    def _unsharp_mask(self, gray: np.ndarray) -> np.ndarray:
        """Sharpen faint thermal print using unsharp masking."""
        try:
            gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
            unsharp = cv2.addWeighted(gray, 1.4, gaussian, -0.4, 0)
            return unsharp
        except Exception:
            return gray

    def _preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        if not os.path.exists(image_path):
            return None
        img: Optional[np.ndarray] = cv2.imread(image_path)
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Deskew image to level text rows
        gray = self._deskew_image(gray)

        # 2. Adaptive resolution scaling (upscale low-res, downscale huge images)
        height, width = gray.shape[:2]
        max_dim = max(height, width)
        if max_dim < 1400:
            scale = 1600.0 / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        elif max_dim > 2400:
            scale = 2000.0 / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # 3. Unsharp masking to accentuate thermal characters
        gray = self._unsharp_mask(gray)

        # 4. Contrast Limited Adaptive Histogram Equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)

        return enhanced_gray

    @monitor_performance
    def extract_text(self, image_path: str) -> str:
        """Extract raw text from receipt image."""
        try:
            enhanced_gray = self._preprocess_image(image_path)
            if enhanced_gray is None:
                return ""
            text: str = pytesseract.image_to_string(
                enhanced_gray, lang=self.lang, config=self.config
            )
            return text.strip()
        except Exception:
            return ""

    def extract_boxes(self, image_path: str) -> List[Dict[str, Any]]:
        """Extract word-level bounding boxes with coordinates."""
        try:
            enhanced_gray = self._preprocess_image(image_path)
            if enhanced_gray is None:
                return []
            data = pytesseract.image_to_data(
                enhanced_gray,
                lang=self.lang,
                config=self.config,
                output_type=Output.DICT,
            )
            boxes: List[Dict[str, Any]] = []
            n_boxes = len(data["text"])
            for i in range(n_boxes):
                text = str(data["text"][i]).strip()
                if not text:
                    continue
                x = int(data["left"][i])
                y = int(data["top"][i])
                w = int(data["width"][i])
                h = int(data["height"][i])
                conf = float(data["conf"][i]) if str(data["conf"][i]) != "-1" else 0.0
                boxes.append({
                    "text": text,
                    "box": [x, y, x + w, y + h],
                    "conf": conf,
                })
            return boxes
        except Exception:
            return []


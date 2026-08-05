"""RapidOCR (PaddleOCR ONNX) Adapter implementation for PyReceipt.

Leverages rapidocr_onnxruntime for state-of-the-art document layout analysis
and text detection using ONNX models loaded into memory once.
"""

import os
from typing import Optional, List, Tuple, Any
from rapidocr_onnxruntime import RapidOCR

from pyreceipt.core.ports import OCRPort
from pyreceipt.utils.profiler import monitor_performance


class RapidOCRAdapter(OCRPort):
    """Concrete RapidOCR Adapter implementing OCRPort interface."""

    def __init__(self) -> None:
        """Initialize RapidOCRAdapter and load ONNX models into memory once."""
        self.engine = RapidOCR()

    @monitor_performance
    def extract_text(self, image_path: str) -> str:
        """Extract raw text from receipt image using RapidOCR engine.

        Relies on RapidOCR's internal image normalization pipeline.
        Iterates over detection tuples (bbox, text, confidence) and joins text lines.

        Args:
            image_path: Absolute or relative file path to the receipt image.

        Returns:
            Extracted raw text string from the image, or empty string if error occurs.
        """
        if not os.path.exists(image_path):
            return ""

        try:
            result, _ = self.engine(image_path)
            if not result:
                return ""

            extracted_lines: List[str] = []
            for item in result:
                # item structure: [bbox, text, confidence]
                if item and len(item) >= 2:
                    text_content = str(item[1]).strip()
                    if text_content:
                        extracted_lines.append(text_content)

            return "\n".join(extracted_lines)
        except Exception:
            return ""

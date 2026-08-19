"""RapidOCR (PaddleOCR ONNX) Adapter implementation for PyReceipt.

Leverages ultra-fast ONNX runtime models (DBNet text detector + CRNN text recognizer)
for highly accurate and fast text/box extraction across orientations and fonts.
"""

import os
from typing import Any, Dict, List, Optional
from rapidocr_onnxruntime import RapidOCR

from pyreceipt.core.ports import OCRPort
from pyreceipt.utils.profiler import monitor_performance


class RapidOCRAdapter(OCRPort):
    """Concrete RapidOCR Adapter implementing OCRPort interface.

    Uses PaddleOCR's lightweight DBNet + CRNN models running on ONNX Runtime.

    Attributes:
        engine (RapidOCR): The initialized RapidOCR ONNX inference engine.
    """

    def __init__(self) -> None:
        """Initialize RapidOCRAdapter and load ONNX models into memory once."""
        self.engine = RapidOCR()

    @monitor_performance
    def extract_text(self, image_path: str) -> str:
        """Extract raw text from receipt image using RapidOCR engine.

        Args:
            image_path: Absolute or relative file path to the receipt image.

        Returns:
            Newline-separated multiline text string extracted from the image.
        """
        boxes = self.extract_boxes(image_path)
        return "\n".join(b["text"] for b in boxes)

    def extract_boxes(self, image_path: str) -> List[Dict[str, Any]]:
        """Extract bounding boxes with text, polygon coordinates, and confidence.

        Args:
            image_path: Absolute or relative file path to the receipt image.

        Returns:
            A list of dictionary objects with structure::

                {
                    "text": str,
                    "box": [x0, y0, x1, y1],
                    "conf": float
                }

            Returns an empty list if the image cannot be read or processed.
        """
        if not os.path.exists(image_path):
            return []

        try:
            result, _ = self.engine(image_path)
            if not result:
                return []

            boxes: List[Dict[str, Any]] = []
            for item in result:
                if item and len(item) >= 2:
                    text_content = str(item[1]).strip()
                    if text_content:
                        poly = item[0]
                        xs = [int(p[0]) for p in poly]
                        ys = [int(p[1]) for p in poly]
                        conf = float(item[2]) if len(item) > 2 else 1.0
                        boxes.append({
                            "text": text_content,
                            "box": [min(xs), min(ys), max(xs), max(ys)],
                            "conf": conf,
                        })

            # Sort top to bottom (Y-axis), then left to right (X-axis)
            boxes.sort(key=lambda b: (b["box"][1], b["box"][0]))
            return boxes
        except Exception:
            return []


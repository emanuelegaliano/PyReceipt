"""EasyOCR Adapter implementation for PyReceipt.

Wraps the EasyOCR deep learning framework (CRAFT text detection + PyTorch recognition)
to extract text lines and bounding boxes from receipt images.
"""

import os
from typing import Any, Dict, List, Optional

from pyreceipt.core.ports import OCRPort
from pyreceipt.utils.profiler import monitor_performance


class EasyOCRAdapter(OCRPort):
    """Concrete EasyOCR Adapter implementing OCRPort interface.

    Uses PyTorch CRAFT text detection and ResNet-based sequence recognition.

    Attributes:
        langs (List[str]): List of ISO language codes supported by this instance.
        reader (easyocr.Reader): The initialized EasyOCR Reader engine.
    """

    def __init__(
        self,
        lang_list: Optional[List[str]] = None,
        gpu: bool = True,
    ) -> None:
        """Initialize EasyOCRAdapter and load reader model into memory once.

        Args:
            lang_list: Optional list of ISO language codes (e.g. `['en', 'it']`).
                Defaults to `['en']`.
            gpu: Whether to utilize GPU acceleration (CUDA / MPS) if available.
        """
        import easyocr

        self.langs = lang_list if lang_list is not None else ["en"]
        self.reader = easyocr.Reader(self.langs, gpu=gpu)

    @monitor_performance
    def extract_text(self, image_path: str) -> str:
        """Extract raw text lines from receipt image using EasyOCR engine.

        Args:
            image_path: Absolute or relative file path to the receipt image.

        Returns:
            Newline-separated text string extracted from the image.
        """
        boxes = self.extract_boxes(image_path)
        return "\n".join(b["text"] for b in boxes)

    def extract_boxes(self, image_path: str) -> List[Dict[str, Any]]:
        """Extract bounding boxes with text and coordinates using EasyOCR engine.

        Args:
            image_path: Absolute or relative file path to the receipt image.

        Returns:
            A list of dictionary objects with structure::

                {
                    "text": str,
                    "box": [x0, y0, x1, y1],
                    "conf": float
                }
        """
        if not os.path.exists(image_path):
            return []

        try:
            results = self.reader.readtext(image_path, detail=1, paragraph=False)
            if not results:
                return []

            boxes: List[Dict[str, Any]] = []
            for item in results:
                if len(item) >= 2:
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


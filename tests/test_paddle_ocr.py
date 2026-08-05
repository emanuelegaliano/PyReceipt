"""Unit tests for the RapidOCR (PaddleOCR ONNX) Adapter."""

import unittest
import numpy as np
import cv2

from pyreceipt.adapters.paddle_ocr import RapidOCRAdapter
from pyreceipt.core.ports import OCRPort


class TestRapidOCRAdapter(unittest.TestCase):
    """Test suite for RapidOCRAdapter."""

    def test_rapid_ocr_adapter_inherits_port(self) -> None:
        """Verify that RapidOCRAdapter implements OCRPort interface."""
        adapter = RapidOCRAdapter()
        self.assertIsInstance(adapter, OCRPort)

    def test_rapid_ocr_adapter_handles_nonexistent_file(self) -> None:
        """Verify that non-existent image paths return empty string gracefully."""
        adapter = RapidOCRAdapter()
        result = adapter.extract_text("non_existent_image_12345.jpg")
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()

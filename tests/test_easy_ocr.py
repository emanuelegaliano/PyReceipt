"""Unit tests for the EasyOCR Adapter."""

import unittest
from unittest.mock import MagicMock, patch

try:
    import easyocr
    from pyreceipt.adapters.easy_ocr import EasyOCRAdapter
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

from pyreceipt.core.ports import OCRPort


@unittest.skipUnless(EASYOCR_AVAILABLE, "easyocr package not installed")
class TestEasyOCRAdapter(unittest.TestCase):
    """Test suite for EasyOCRAdapter."""

    @patch("easyocr.Reader")
    def test_easy_ocr_adapter_inherits_port(self, mock_reader_cls) -> None:
        """Verify that EasyOCRAdapter implements OCRPort interface."""
        adapter = EasyOCRAdapter(gpu=False)
        self.assertIsInstance(adapter, OCRPort)

    @patch("easyocr.Reader")
    def test_easy_ocr_adapter_handles_nonexistent_file(self, mock_reader_cls) -> None:
        """Verify that non-existent image paths return empty string gracefully."""
        adapter = EasyOCRAdapter(gpu=False)
        result = adapter.extract_text("non_existent_image_12345.jpg")
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()

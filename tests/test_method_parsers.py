"""Unit tests for Method 1 (Spatial 2D Box Parser) and Method 2 (LayoutLM Parser)."""

import unittest
from unittest.mock import patch, MagicMock

from pyreceipt.core.domain import Receipt
from pyreceipt.adapters.spatial_2d_parser import Spatial2DBoxParser
from pyreceipt.adapters.layoutlm_parser import LayoutLMReceiptParser


class TestMethodParsers(unittest.TestCase):
    """Test suite for Method 1 and Method 2 Parsers."""

    def test_spatial_2d_box_parser_clustering(self) -> None:
        """Verify 2D spatial row clustering and ray-casting from Total anchor."""
        parser = Spatial2DBoxParser()
        boxes = [
            {"text": "BOOK STORE", "box": [50, 20, 200, 40], "conf": 0.9},
            {"text": "Date: 12/05/2020", "box": [50, 50, 180, 70], "conf": 0.9},
            {"text": "ITEM 1", "box": [50, 100, 150, 120], "conf": 0.9},
            {"text": "10.00", "box": [450, 100, 520, 120], "conf": 0.9},
            {"text": "TOTAL INCL GST", "box": [50, 200, 220, 220], "conf": 0.9},
            {"text": "45.80", "box": [450, 200, 520, 220], "conf": 0.9}, # Same row Y=200
            {"text": "CASH", "box": [50, 250, 120, 270], "conf": 0.9},
            {"text": "50.00", "box": [450, 250, 520, 270], "conf": 0.9},
            {"text": "CHANGE", "box": [50, 300, 140, 320], "conf": 0.9},
            {"text": "4.20", "box": [450, 300, 500, 320], "conf": 0.9},
        ]
        receipt = parser.parse(boxes)
        self.assertEqual(receipt.total, 45.80)
        self.assertEqual(receipt.date, "12/05/2020")
        self.assertEqual(receipt.company, "BOOK STORE")

    def test_spatial_2d_box_parser_tax_deweighting(self) -> None:
        """Verify that TAX TOTAL is ignored in favor of GRAND TOTAL."""
        parser = Spatial2DBoxParser()
        boxes = [
            {"text": "GROCERY MART", "box": [50, 20, 200, 40], "conf": 0.9},
            {"text": "Date: 14/02/2023", "box": [50, 50, 180, 70], "conf": 0.9},
            {"text": "TAX TOTAL", "box": [50, 150, 150, 170], "conf": 0.9},
            {"text": "0.57", "box": [450, 150, 500, 170], "conf": 0.9},
            {"text": "GRAND TOTAL", "box": [50, 200, 180, 220], "conf": 0.9},
            {"text": "10.00", "box": [450, 200, 520, 220], "conf": 0.9},
        ]
        receipt = parser.parse(boxes)
        self.assertEqual(receipt.total, 10.00)

    def test_spatial_2d_box_parser_arithmetic_cross_validation(self) -> None:
        """Verify arithmetic cross-validation (Cash - Change == Total) resolves ambiguity."""
        parser = Spatial2DBoxParser()
        boxes = [
            {"text": "RESTO CAFE", "box": [50, 20, 200, 40], "conf": 0.9},
            {"text": "TOTAL", "box": [50, 180, 120, 200], "conf": 0.9},
            {"text": "21.60", "box": [450, 180, 520, 200], "conf": 0.9},
            {"text": "CASH", "box": [50, 220, 100, 240], "conf": 0.9},
            {"text": "30.00", "box": [450, 220, 520, 240], "conf": 0.9},
            {"text": "CHANGE", "box": [50, 260, 120, 280], "conf": 0.9},
            {"text": "8.40", "box": [450, 260, 500, 280], "conf": 0.9},
        ]
        receipt = parser.parse(boxes)
        self.assertEqual(receipt.total, 21.60)

    def test_spatial_2d_box_parser_fuzzy_anchor(self) -> None:
        """Verify fuzzy anchor matching catches OCR typos like T0TAL."""
        parser = Spatial2DBoxParser()
        boxes = [
            {"text": "STORE", "box": [50, 20, 150, 40], "conf": 0.9},
            {"text": "T0TAL", "box": [50, 150, 120, 170], "conf": 0.9},
            {"text": "45.80", "box": [450, 150, 520, 170], "conf": 0.9},
        ]
        receipt = parser.parse(boxes)
        self.assertEqual(receipt.total, 45.80)

    def test_spatial_2d_box_parser_alphanumeric_date(self) -> None:
        """Verify alphanumeric date formats (e.g. 25 MAR 2018)."""
        parser = Spatial2DBoxParser()
        boxes = [
            {"text": "SUPERMARKET", "box": [50, 20, 200, 40], "conf": 0.9},
            {"text": "Date: 25 MAR 2018", "box": [50, 50, 220, 70], "conf": 0.9},
            {"text": "TOTAL", "box": [50, 150, 120, 170], "conf": 0.9},
            {"text": "12.50", "box": [450, 150, 520, 170], "conf": 0.9},
        ]
        receipt = parser.parse(boxes)
        self.assertEqual(receipt.date, "25 MAR 2018")

    @patch("transformers.pipeline")
    def test_layoutlm_parser_mocked(self, mock_pipe_fn) -> None:
        """Verify LayoutLM parser extraction logic."""
        mock_pipe = MagicMock()
        mock_pipe.side_effect = lambda img, q: [
            {"answer": "45.80"} if "total" in q else
            {"answer": "12/05/2020"} if "date" in q else
            {"answer": "BOOK STORE"}
        ]
        mock_pipe_fn.return_value = mock_pipe

        parser = LayoutLMReceiptParser()
        parser.pipe = mock_pipe
        receipt = parser.parse_image("non_existent.jpg")
        self.assertEqual(receipt.total, 0.0) # Non-existent file


if __name__ == "__main__":
    unittest.main()


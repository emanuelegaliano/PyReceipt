"""Unit tests for RegexReceiptParser using Python standard library unittest."""

import unittest

from pyreceipt.core.domain import ExpenseCategory, Receipt
from pyreceipt.core.parser import RegexReceiptParser


class TestRegexReceiptParser(unittest.TestCase):
    """Test suite for RegexReceiptParser with language-specific JSON configurations."""

    def test_parse_italian_receipt(self) -> None:
        """Verify parsing of Italian receipt text using 'it' configuration."""
        raw_text = """
        *** SUPERMERCATO MILANO ***
        Via Roma 45
        Data: 15/04/2024
        Pane 2.50
        Latte 1.50
        SUBTOTAL 4.00
        TOTALE: 15.50
        Grazie e Arrivederci
        """
        parser = RegexReceiptParser(lang_code="it")
        receipt = parser.parse(raw_text)

        self.assertIsInstance(receipt, Receipt)
        self.assertEqual(receipt.company, "SUPERMERCATO MILANO")
        self.assertEqual(receipt.date, "15/04/2024")
        self.assertEqual(receipt.total, 15.50)
        self.assertEqual(receipt.category, ExpenseCategory.OTHER)

    def test_parse_english_receipt(self) -> None:
        """Verify parsing of English receipt text using 'en' configuration."""
        raw_text = """
        # STARBUCKS COFFEE
        123 Main Street
        Date: 05/26/2016
        1 Coffee 3.00
        2 Lunch 45.90
        SUB TOTAL: 51.90
        TOTAL: $56.58
        THANK YOU!
        """
        parser = RegexReceiptParser(lang_code="en")
        receipt = parser.parse(raw_text)

        self.assertIsInstance(receipt, Receipt)
        self.assertEqual(receipt.company, "STARBUCKS COFFEE")
        self.assertEqual(receipt.date, "05/26/2016")
        self.assertEqual(receipt.total, 56.58)
        self.assertEqual(receipt.category, ExpenseCategory.OTHER)

    def test_unsupported_language_raises_file_not_found(self) -> None:
        """Verify FileNotFoundError is raised when initialized with an unsupported lang code."""
        with self.assertRaises(FileNotFoundError):
            RegexReceiptParser(lang_code="fr_UNSUPPORTED")

    def test_company_fallback_first_alphanumeric_line(self) -> None:
        """Verify company extraction skips non-alphanumeric header lines."""
        raw_text = """
        *** --- ***
        === ===
        ACME STORE INC.
        Date: 01/01/2025
        TOTAL 10.00
        """
        parser = RegexReceiptParser(lang_code="en")
        receipt = parser.parse(raw_text)

        self.assertEqual(receipt.company, "ACME STORE INC.")

    def test_parser_empty_text(self) -> None:
        """Verify behavior on empty raw text."""
        parser = RegexReceiptParser(lang_code="it")
        receipt = parser.parse("")

        self.assertEqual(receipt.company, "UNKNOWN")
        self.assertEqual(receipt.date, "")
        self.assertEqual(receipt.total, 0.0)
        self.assertEqual(receipt.category, ExpenseCategory.OTHER)

    def test_parse_ocr_spaces_and_commas(self) -> None:
        """Verify parsing of money strings with OCR spaces and comma separators."""
        raw_text = """
        ITALIAN CAFE
        25.12.2023
        Espresso 1.20
        Panino 4.80
        TOTALE : 15 , 50
        """
        parser = RegexReceiptParser(lang_code="it")
        receipt = parser.parse(raw_text)

        self.assertEqual(receipt.date, "25.12.2023")
        self.assertEqual(receipt.total, 15.50)

    def test_greedy_total_fallback_max_float(self) -> None:
        """Verify greedy fallback returns largest float when total keyword is missing."""
        raw_text = """
        RESTAURANT BARKLEY
        2024-11-15
        Item 1 12.00
        Item 2 45.00
        Item 3 99.95
        Random Code 00123
        """
        parser = RegexReceiptParser(lang_code="en")
        receipt = parser.parse(raw_text)

        self.assertEqual(receipt.company, "RESTAURANT BARKLEY")
        self.assertEqual(receipt.date, "2024-11-15")
        self.assertEqual(receipt.total, 99.95)


if __name__ == "__main__":
    unittest.main()

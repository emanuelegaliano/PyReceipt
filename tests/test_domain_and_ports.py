"""Unit tests for PyReceipt Phase 1: Domain and Ports."""

from datetime import datetime
import os
import sys
import unittest

# Add root project directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyreceipt.core.domain import ExpenseCategory, Receipt
from pyreceipt.core.ports import OCRPort, StoragePort


class TestExpenseCategory(unittest.TestCase):
    """Test suite for ExpenseCategory Value Object."""

    def test_default_categories(self) -> None:
        """Test default pre-defined categories."""
        self.assertEqual(ExpenseCategory.GROCERIES.name, "GROCERIES")
        self.assertEqual(ExpenseCategory.RESTAURANT.name, "RESTAURANT")
        self.assertEqual(ExpenseCategory.TRANSPORT.name, "TRANSPORT")
        self.assertEqual(ExpenseCategory.OTHER.name, "OTHER")

    def test_dynamic_category_creation(self) -> None:
        """Test dynamic creation and normalization of custom categories."""
        custom_cat = ExpenseCategory.from_name("  electronics & Tech  ")
        self.assertEqual(custom_cat.name, "ELECTRONICS & TECH")

        # Test value object equality
        cat_a = ExpenseCategory("GROCERIES")
        self.assertEqual(cat_a, ExpenseCategory.GROCERIES)

    def test_empty_category_name_raises(self) -> None:
        """Test that empty category names raise ValueError."""
        with self.assertRaises(ValueError):
            ExpenseCategory.from_name("   ")


class TestReceipt(unittest.TestCase):
    """Test suite for Receipt domain model."""

    def test_receipt_instantiation(self) -> None:
        """Test creating a valid Receipt instance."""
        category = ExpenseCategory.from_name("Hardware")
        now = datetime.now()
        receipt = Receipt(
            company="SuperMarket Ltd",
            date=now,
            total=45.99,
            category=category,
        )
        self.assertEqual(receipt.company, "SuperMarket Ltd")
        self.assertEqual(receipt.date, now)
        self.assertEqual(receipt.total, 45.99)
        self.assertEqual(receipt.category.name, "HARDWARE")


class TestPortsAbstractContracts(unittest.TestCase):
    """Test suite for Abstract Base Classes (Ports)."""

    def test_ocr_port_cannot_be_instantiated_directly(self) -> None:
        """Verify OCRPort cannot be instantiated without implementing extract_text."""
        with self.assertRaises(TypeError):
            OCRPort()  # type: ignore

    def test_storage_port_cannot_be_instantiated_directly(self) -> None:
        """Verify StoragePort cannot be instantiated without implementing save_receipt."""
        with self.assertRaises(TypeError):
            StoragePort()  # type: ignore

    def test_concrete_ocr_port_instantiation(self) -> None:
        """Verify concrete subclass of OCRPort works correctly."""

        class DummyOCRAdapter(OCRPort):
            def extract_text(self, image_path: str) -> str:
                return "Dummy receipt text"

        adapter = DummyOCRAdapter()
        self.assertEqual(adapter.extract_text("path/to/img.jpg"), "Dummy receipt text")

    def test_concrete_storage_port_instantiation(self) -> None:
        """Verify concrete subclass of StoragePort works correctly."""

        saved_receipts = []

        class DummyStorageAdapter(StoragePort):
            def save_receipt(self, receipt: Receipt) -> None:
                saved_receipts.append(receipt)

        adapter = DummyStorageAdapter()
        receipt = Receipt(
            company="Cafe",
            date="2026-08-04",
            total=3.50,
            category=ExpenseCategory.RESTAURANT,
        )
        adapter.save_receipt(receipt)
        self.assertEqual(len(saved_receipts), 1)
        self.assertEqual(saved_receipts[0], receipt)


if __name__ == "__main__":
    unittest.main()

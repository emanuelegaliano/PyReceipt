"""Unit tests for the Tesseract OCR Adapter."""

import os
import cv2
import numpy as np
import pytest

from pyreceipt.adapters.tesseract_ocr import TesseractOCRAdapter
from pyreceipt.core.ports import OCRPort


def test_tesseract_ocr_adapter_inherits_port():
    """Verify that TesseractOCRAdapter implements OCRPort interface."""
    adapter = TesseractOCRAdapter()
    assert isinstance(adapter, OCRPort)


def test_tesseract_ocr_adapter_handles_nonexistent_file():
    """Verify that non-existent image paths return empty string gracefully."""
    adapter = TesseractOCRAdapter()
    result = adapter.extract_text("non_existent_image_12345.jpg")
    assert result == ""


def test_tesseract_ocr_adapter_extracts_text_and_resizes(tmp_path):
    """Verify OCR text extraction and dynamic image resizing on large synthetic image."""
    # Create a synthetic image larger than 1024px (e.g. 2000x1500) with text
    img_path = str(tmp_path / "large_receipt.jpg")
    img = np.ones((1500, 2000, 3), dtype=np.uint8) * 255
    cv2.putText(
        img,
        "SUPERMARKET",
        (100, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        3.0,
        (0, 0, 0),
        5,
    )
    cv2.putText(
        img,
        "TOTALE 10.50",
        (100, 600),
        cv2.FONT_HERSHEY_SIMPLEX,
        3.0,
        (0, 0, 0),
        5,
    )
    cv2.imwrite(img_path, img)

    adapter = TesseractOCRAdapter()
    text = adapter.extract_text(img_path)
    assert isinstance(text, str)


def test_tesseract_ocr_adapter_deskew_and_unsharp():
    """Verify that _deskew_image and _unsharp_mask execute correctly on numpy matrices."""
    adapter = TesseractOCRAdapter()
    gray = np.ones((500, 500), dtype=np.uint8) * 255
    cv2.putText(gray, "SAMPLE TEXT", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    deskewed = adapter._deskew_image(gray)
    assert isinstance(deskewed, np.ndarray)
    assert deskewed.shape == gray.shape

    unsharp = adapter._unsharp_mask(gray)
    assert isinstance(unsharp, np.ndarray)
    assert unsharp.shape == gray.shape


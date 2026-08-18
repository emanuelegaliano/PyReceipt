"""Adapters package of PyReceipt for concrete OCR and Parser implementations."""

from pyreceipt.adapters.tesseract_ocr import TesseractOCRAdapter
from pyreceipt.adapters.spatial_2d_parser import Spatial2DBoxParser

__all__ = [
    "TesseractOCRAdapter",
    "Spatial2DBoxParser",
]

try:
    from pyreceipt.adapters.paddle_ocr import RapidOCRAdapter

    __all__.append("RapidOCRAdapter")
except ImportError:
    pass

try:
    from pyreceipt.adapters.easy_ocr import EasyOCRAdapter

    __all__.append("EasyOCRAdapter")
except ImportError:
    pass

try:
    from pyreceipt.adapters.layoutlm_parser import LayoutLMReceiptParser

    __all__.append("LayoutLMReceiptParser")
except ImportError:
    pass

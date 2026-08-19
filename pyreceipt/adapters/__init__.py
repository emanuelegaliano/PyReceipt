"""Infrastructure Adapters Layer of PyReceipt.

Contains concrete implementations of the abstract ports defined in :mod:`pyreceipt.core.ports`:
    - **OCR Adapters**:
        - :class:`pyreceipt.adapters.tesseract_ocr.TesseractOCRAdapter` (OpenCV CV + Tesseract LSTM)
        - :class:`pyreceipt.adapters.paddle_ocr.RapidOCRAdapter` (PaddleOCR DBNet + CRNN ONNX)
        - :class:`pyreceipt.adapters.easy_ocr.EasyOCRAdapter` (PyTorch CRAFT + ResNet)
    - **Parser Adapters**:
        - :class:`pyreceipt.adapters.spatial_2d_parser.Spatial2DBoxParser` (Geometric 2D clustering & arithmetic verification)
        - :class:`pyreceipt.adapters.layoutlm_parser.LayoutLMReceiptParser` (Visual Document AI QA)
"""

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


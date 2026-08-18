"""Spatial 2D Bounding Box Receipt Parser (Method 1: Geometric 2D Clustering & Ray-Casting).

Reconstructs physical 2D rows from OCR bounding box coordinates (Y-overlap),
performs horizontal ray-casting to right-align price columns, and filters out
settlement/cash lines.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union

from pyreceipt.core.domain import ExpenseCategory, Receipt
from pyreceipt.core.ports import ParserPort
from pyreceipt.utils.profiler import monitor_performance


class Spatial2DBoxParser(ParserPort):
    """Geometric 2D Bounding Box Receipt Parser."""

    TOTAL_ANCHORS = [
        "TOTAL", "TOTAL INCL GST", "TOTAL AMOUNT", "TOTAL RM", "TOTAL (RM)",
        "NETT TOTAL", "NET TOTAL", "AMOUNT DUE", "AMOUNT PAYABLE", "GRAND TOTAL",
        "JUMLAH", "TOTAL PAYABLE", "BALANCE DUE", "TOTAL ROUNDED"
    ]

    NEGATIVE_ANCHORS = [
        "CASH", "TUNAI", "CHANGE", "BAKI", "ROUNDING", "SUBTOTAL", "SUB-TOTAL",
        "SUB TOTAL", "TAX INVOICE", "GST REG", "TEL", "FAX", "CARD NO", "APPROVAL",
        "DISCOUNT", "VOUCHER", "POINTS", "MEMBER"
    ]

    DATE_PATTERNS = [
        re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"),
        re.compile(r"\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"),
        re.compile(r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b", re.IGNORECASE),
        re.compile(r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})\b", re.IGNORECASE),
        re.compile(r"\b(\d{1,2}[.]\d{1,2}[.]\d{2,4})\b"),
    ]

    PRICE_PATTERN = re.compile(r"\b([0-9]{1,5}(?:\s*[\.,]\s*[0-9]{2}))\b")

    def __init__(self, y_tolerance_ratio: float = 0.45) -> None:
        self.y_tol = y_tolerance_ratio

    def _clean_price(self, price_str: str) -> float:
        clean = re.sub(r"[^\d.,]", "", price_str).replace(" ", "").replace(",", ".")
        try:
            return float(clean)
        except ValueError:
            return 0.0

    def _cluster_boxes_into_rows(self, boxes: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group word boxes into horizontal 2D physical lines based on vertical overlap."""
        if not boxes:
            return []

        # Sort all boxes primarily by top-Y, secondarily by left-X
        sorted_boxes = sorted(boxes, key=lambda b: (b["box"][1], b["box"][0]))
        rows: List[List[Dict[str, Any]]] = []

        for b in sorted_boxes:
            b_y_mid = (b["box"][1] + b["box"][3]) / 2.0
            b_h = max(b["box"][3] - b["box"][1], 10)

            # Check if this box fits in an existing row
            matched_row = None
            for row in rows:
                row_y_mid = sum((item["box"][1] + item["box"][3]) / 2.0 for item in row) / len(row)
                row_h = sum(item["box"][3] - item["box"][1] for item in row) / len(row)
                tol = max(b_h, row_h) * self.y_tol

                if abs(b_y_mid - row_y_mid) <= tol:
                    matched_row = row
                    break

            if matched_row is not None:
                matched_row.append(b)
            else:
                rows.append([b])

        # Sort elements within each row from Left to Right (X-coordinate)
        for row in rows:
            row.sort(key=lambda item: item["box"][0])

        # Sort all rows from Top to Bottom (Y-coordinate)
        rows.sort(key=lambda row: row[0]["box"][1])
        return rows

    @monitor_performance
    def parse(self, ocr_input: Union[str, List[Dict[str, Any]]]) -> Receipt:
        """Parse structured bounding boxes or raw text using 2D geometric alignment."""
        if isinstance(ocr_input, str):
            # Synthetic boxes from raw text lines if boxes not provided
            lines = [l.strip() for l in ocr_input.splitlines() if l.strip()]
            boxes = [{"text": l, "box": [0, i * 20, 100, (i + 1) * 20], "conf": 1.0} for i, l in enumerate(lines)]
        else:
            boxes = ocr_input

        if not boxes:
            return Receipt(company="UNKNOWN", date="", total=0.0, category=ExpenseCategory.OTHER)

        # 1. 2D Row Clustering
        rows = self._cluster_boxes_into_rows(boxes)
        total_rows = len(rows)

        # 2. Extract Company Name (from first valid top row)
        company = "UNKNOWN"
        for row in rows[: min(5, total_rows)]:
            row_text = " ".join(item["text"] for item in row).strip()
            if len(row_text) >= 3 and not re.search(r"(tax invoice|receipt|welcome|bill|tel|fax|date)", row_text, re.IGNORECASE):
                if not re.match(r"^[\d\W]+$", row_text):
                    company = row_text
                    break

        # 3. Extract Date
        date = ""
        full_text = " ".join(item["text"] for row in rows for item in row)
        for pat in self.DATE_PATTERNS:
            m = pat.search(full_text)
            if m:
                date = m.group(1).strip()
                break

        # 4. Geometric 2D Ray-Casting for Final Total
        candidates: List[Tuple[float, float, str]] = [] # (score, amount, row_text)

        for r_idx, row in enumerate(rows):
            row_text = " ".join(item["text"] for item in row)
            row_upper = row_text.upper()
            rel_y = r_idx / max(total_rows, 1)

            # Check for Total anchors
            is_total_anchor = any(anc in row_upper for anc in self.TOTAL_ANCHORS)
            has_negative = any(neg in row_upper for neg in self.NEGATIVE_ANCHORS)

            if is_total_anchor and not has_negative:
                # Find prices in this same physical 2D row (Right-most item)
                prices_in_row = []
                for item in reversed(row):
                    found = self.PRICE_PATTERN.findall(item["text"])
                    if found:
                        for p in found:
                            val = self._clean_price(p)
                            if 0 < val < 50000:
                                prices_in_row.append(val)

                if prices_in_row:
                    score = 100.0 + (rel_y * 20.0)
                    if "ROUNDED" in row_upper:
                        score += 30.0
                    if "INCL" in row_upper:
                        score += 25.0
                    candidates.append((score, prices_in_row[0], row_text))
                else:
                    # Look ahead 1 row below (Y+1) if number is right under label
                    if r_idx + 1 < total_rows:
                        next_row = rows[r_idx + 1]
                        next_text = " ".join(item["text"] for item in next_row)
                        for item in reversed(next_row):
                            found = self.PRICE_PATTERN.findall(item["text"])
                            for p in found:
                                val = self._clean_price(p)
                                if 0 < val < 50000:
                                    candidates.append((85.0 + rel_y * 10, val, row_text + " " + next_text))

        total = 0.0
        if candidates:
            candidates.sort(key=lambda c: c[0], reverse=True)
            total = candidates[0][1]

        # Secondary fallback: non-cash price from bottom 50%
        if total <= 0:
            for row in reversed(rows[int(total_rows * 0.4):]):
                row_text = " ".join(item["text"] for item in row).upper()
                if any(neg in row_text for neg in ["CASH", "CHANGE", "TUNAI", "BAKI", "SUBTOTAL"]):
                    continue
                found = self.PRICE_PATTERN.findall(row_text)
                if found:
                    total = self._clean_price(found[-1])
                    break

        return Receipt(company=company, date=date, total=total, category=ExpenseCategory.OTHER)

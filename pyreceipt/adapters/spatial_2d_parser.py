"""Spatial 2D Bounding Box Receipt Parser (Method 1: Geometric 2D Clustering & Ray-Casting).

Reconstructs physical 2D rows from OCR bounding box coordinates (Y-overlap),
performs horizontal ray-casting to right-align price columns, filters out
tax/settlement/cash lines, and performs arithmetic cross-validation.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union

from pyreceipt.core.domain import ExpenseCategory, Receipt
from pyreceipt.core.ports import ParserPort
from pyreceipt.utils.profiler import monitor_performance


class Spatial2DBoxParser(ParserPort):
    """Geometric 2D Bounding Box Receipt Parser with Arithmetic Verification."""

    TOTAL_KEYWORD_REGEX = re.compile(
        r"(?:[TO0][O0][TAFLI1][A4][LI1]|JUMLAH|GRAND\s*TOTAL|NETT?\s*TOTAL|AMOUNT\s*DUE|AMOUNT\s*PAYABLE|BALANCE\s*DUE|TOTAL\s*ROUNDED|TOTAL\s*INCL|TOTAL\s*RM|TOTAL\s*AMOUNT|TOTAL\s*SALES|TOTAL\s*PAYABLE|TOTAL\s*BILL|RINGKASAN|TOTAL\s*PAID|TOTAL\s*PRICE)",
        re.IGNORECASE,
    )

    TAX_EXCLUSION_KEYWORDS = [
        "TAX TOTAL", "GST TOTAL", "TOTAL TAX", "TOTAL GST", "TAX AMOUNT", "GST AMOUNT",
        "GST (6%)", "SR @", "ZR @", "SR@", "ZR@", "GST SUMMARY", "TAX (RM)", "TAX INVOICE",
        "GST REG", "AMOUNT EXCL", "TOTAL EXCL", "TOTAL DISCOUNT", "TOTAL SAVINGS", "TOTAL SAVING",
        "TOTAL ITEM", "TOTAL ITEMS", "TOTAL QTY", "TOTAL PIECES", "TOTAL PCS", "TOTAL UNIT",
        "SUBTOTAL", "SUB-TOTAL", "SUB TOTAL", "SERVICE TAX", "TAXABLE AMOUNT", "EXCLUDING GST",
        "EXCL GST", "AMOUNT (EXCL", "GSTANALYSIS", "GST ANALYSIS", "TAX/AMT", "TAXABLE"
    ]

    PAYMENT_NEGATIVE_KEYWORDS = [
        "CASH", "TUNAI", "CHANGE", "BAKI", "KEMBALI", "ROUNDING", "ROUNDED ADJ",
        "ROUNDING ADJ", "VISA", "MASTER", "CREDIT CARD", "DEBIT CARD", "CARD NO",
        "APPROVAL", "TEL", "FAX", "MEMBER", "POINTS", "VOUCHER", "DISCOUNT"
    ]

    CASH_KEYWORDS = ["CASH", "TUNAI", "CASH TENDERED", "TENDERED", "BAYAR", "CASH RECEIVED", "PAID", "TENDER"]
    CHANGE_KEYWORDS = ["CHANGE", "BAKI", "KEMBALI", "BALANCE RETURN", "CHANGE DUE"]
    SUBTOTAL_KEYWORDS = ["SUBTOTAL", "SUB-TOTAL", "SUB TOTAL", "AMOUNT EXCL", "TAXABLE"]
    TAX_KEYWORDS = ["GST", "TAX", "SST", "6%", "TAX AMOUNT", "GST AMOUNT", "SR @", "TAX/AMT"]

    DATE_PATTERNS = [
        # Match DD/MM/YYYY or DD-MM-YYYY or DD/MM/YY even when directly followed by time/letters
        re.compile(r"(\b\d{1,2}[/-]\d{1,2}[/-](?:20\d{2}|19\d{2}|\d{2}))"),
        # Match YYYY/MM/DD or YYYY-MM-DD
        re.compile(r"(\b(?:20\d{2}|19\d{2})[/-]\d{1,2}[/-]\d{1,2})"),
        # Match DD Mon YYYY or DD-Mon-YY or DDMonYYYY (e.g., 07Mar2018, 22 MAR 18, 25-Apr-2017)
        re.compile(r"(\b\d{1,2}\s*[-/]?\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*[-/]?\s*(?:20\d{2}|19\d{2}|\d{2}))", re.IGNORECASE),
        # Match Mon DD, YYYY
        re.compile(r"(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+(?:20\d{2}|19\d{2}|\d{2}))", re.IGNORECASE),
        # Match DD.MM.YYYY or DD.MM.YY
        re.compile(r"(\b\d{1,2}[.]\d{1,2}[.](?:20\d{2}|19\d{2}|\d{2}))"),
        # Match YYYY.MM.DD
        re.compile(r"(\b(?:20\d{2}|19\d{2})[.]\d{1,2}[.]\d{1,2})"),
    ]

    # Robust price pattern matching attached currency prefixes (e.g. RM110.00, $45.80, 10-00)
    PRICE_PATTERN = re.compile(r"(?:^|[^\d])([0-9]{1,5}\s*[\.,:']\s*[0-9]{2})(?=[^\d]|$)")
    PRICE_HYPHEN_PATTERN = re.compile(r"(?:^|[^\d])([0-9]{1,5}\s*[-–]\s*[0-9]{2})(?=[^\d]|$)")

    def __init__(self, y_tolerance_ratio: float = 0.45) -> None:
        self.y_tol = y_tolerance_ratio

    def _clean_price(self, price_str: str) -> float:
        clean = price_str.replace(" ", "").replace(":", ".").replace("'", ".").replace(",", ".").replace("-", ".").replace("–", ".")
        clean = re.sub(r"[^\d.]", "", clean)
        try:
            return float(clean)
        except ValueError:
            return 0.0

    def _cluster_boxes_into_rows(self, boxes: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group word boxes into horizontal 2D physical lines based on vertical overlap."""
        if not boxes:
            return []

        sorted_boxes = sorted(boxes, key=lambda b: (b["box"][1], b["box"][0]))
        rows: List[List[Dict[str, Any]]] = []

        for b in sorted_boxes:
            b_y_mid = (b["box"][1] + b["box"][3]) / 2.0
            b_h = max(b["box"][3] - b["box"][1], 10)

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

        for row in rows:
            row.sort(key=lambda item: item["box"][0])

        rows.sort(key=lambda row: row[0]["box"][1])
        return rows

    def _extract_prices_from_row(self, row: List[Dict[str, Any]]) -> List[float]:
        """Extract valid prices from right to left in a row."""
        prices: List[float] = []
        for item in reversed(row):
            text = item["text"]
            found = self.PRICE_PATTERN.findall(text) or self.PRICE_HYPHEN_PATTERN.findall(text)
            if found:
                for p in reversed(found):
                    val = self._clean_price(p)
                    if 0 < val < 50000:
                        prices.append(val)
        return prices

    def _is_tax_row(self, row_upper: str) -> bool:
        """Determine if a row represents tax/GST information rather than grand total."""
        for tax_kw in self.TAX_EXCLUSION_KEYWORDS:
            if tax_kw in row_upper and not any(incl in row_upper for incl in ["INCL", "INCLUSIVE"]):
                return True
        return False

    def _extract_context_amounts(self, rows: List[List[Dict[str, Any]]]) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Extract contextual amounts: cash, change, subtotal, and tax."""
        cash_amounts: List[float] = []
        change_amounts: List[float] = []
        subtotal_amounts: List[float] = []
        tax_amounts: List[float] = []

        for row in rows:
            row_text = " ".join(item["text"] for item in row).upper()
            prices = self._extract_prices_from_row(row)
            if not prices:
                continue
            val = prices[0]

            if any(kw in row_text for kw in self.CHANGE_KEYWORDS):
                change_amounts.append(val)
            elif any(kw in row_text for kw in self.CASH_KEYWORDS) and not any(kw in row_text for kw in ["TOTAL", "SUBTOTAL"]):
                cash_amounts.append(val)
            elif any(kw in row_text for kw in self.SUBTOTAL_KEYWORDS):
                subtotal_amounts.append(val)
            elif any(kw in row_text for kw in self.TAX_KEYWORDS) and not any(kw in row_text for kw in ["INCL", "INCLUSIVE"]):
                tax_amounts.append(val)

        return cash_amounts, change_amounts, subtotal_amounts, tax_amounts

    @monitor_performance
    def parse(self, ocr_input: Union[str, List[Dict[str, Any]]]) -> Receipt:
        """Parse structured bounding boxes or raw text using 2D geometric alignment and arithmetic verification."""
        if isinstance(ocr_input, str):
            lines = [l.strip() for l in ocr_input.splitlines() if l.strip()]
            boxes = [{"text": l, "box": [0, i * 20, 100, (i + 1) * 20], "conf": 1.0} for i, l in enumerate(lines)]
        else:
            boxes = ocr_input

        if not boxes:
            return Receipt(company="UNKNOWN", date="", total=0.0, category=ExpenseCategory.OTHER)

        rows = self._cluster_boxes_into_rows(boxes)
        total_rows = len(rows)

        # 1. Extract Company Name (from top header lines)
        company = "UNKNOWN"
        for row in rows[: min(5, total_rows)]:
            row_text = " ".join(item["text"] for item in row).strip()
            if len(row_text) >= 3 and not re.search(r"(tax invoice|receipt|welcome|bill|tel|fax|date|table)", row_text, re.IGNORECASE):
                if not re.match(r"^[\d\W]+$", row_text):
                    company = row_text
                    break

        # 2. Extract Date (with priority on Date anchor proximity)
        date = ""
        for row in rows:
            row_text = " ".join(item["text"] for item in row)
            if re.search(r"(date|tarikh|time|inv\s*date|dated|dt)", row_text, re.IGNORECASE):
                for pat in self.DATE_PATTERNS:
                    m = pat.search(row_text)
                    if m:
                        date = m.group(1).strip()
                        break
            if date:
                break

        if not date:
            full_text = " ".join(item["text"] for row in rows for item in row)
            for pat in self.DATE_PATTERNS:
                m = pat.search(full_text)
                if m:
                    date = m.group(1).strip()
                    break

        # 3. Contextual Amounts for Arithmetic Validation
        cash_list, change_list, subtotal_list, tax_list = self._extract_context_amounts(rows)

        # 4. Geometric Ray-Casting & Scoring for Total
        candidates: List[Tuple[float, float, str]] = []  # (score, amount, row_text)

        for r_idx, row in enumerate(rows):
            row_text = " ".join(item["text"] for item in row)
            row_upper = row_text.upper()
            rel_y = r_idx / max(total_rows, 1)

            is_tax = self._is_tax_row(row_upper)
            is_total_anchor = bool(self.TOTAL_KEYWORD_REGEX.search(row_upper))
            has_payment_negative = any(neg in row_upper for neg in self.PAYMENT_NEGATIVE_KEYWORDS)

            if is_total_anchor and not is_tax:
                prices_in_row = self._extract_prices_from_row(row)

                if prices_in_row:
                    score = 100.0 + (rel_y * 30.0)
                    if any(term in row_upper for term in ["GRAND TOTAL", "NETT TOTAL", "NET TOTAL"]):
                        score += 50.0
                    if any(term in row_upper for term in ["ROUNDED", "TOTAL ROUNDED"]):
                        score += 35.0
                    if any(term in row_upper for term in ["INCL", "INCLUSIVE"]):
                        score += 25.0
                    if any(term in row_upper for term in ["AMOUNT DUE", "AMOUNT PAYABLE", "TOTAL PAYABLE"]):
                        score += 40.0
                    if has_payment_negative:
                        score -= 20.0

                    candidates.append((score, prices_in_row[0], row_text))
                else:
                    for step in [1, 2]:
                        if r_idx + step < total_rows:
                            next_row = rows[r_idx + step]
                            next_text = " ".join(item["text"] for item in next_row)
                            next_prices = self._extract_prices_from_row(next_row)
                            if next_prices:
                                candidates.append((85.0 + (rel_y * 10.0) - (step * 5.0), next_prices[0], row_text + " " + next_text))
                                break

        # 5. Arithmetic Certified Candidates Injection
        if cash_list and change_list:
            for cash in cash_list:
                for change in change_list:
                    diff = round(cash - change, 2)
                    if 0 < diff < 50000:
                        candidates.append((250.0, diff, "ARITHMETIC_CERTIFIED_CASH_MINUS_CHANGE"))

        # 6. Arithmetic Cross-Validation Boost
        validated_candidates: List[Tuple[float, float, str]] = []
        for score, val, r_text in candidates:
            adj_score = score

            # Cash - Change == Total Check
            for cash in cash_list:
                for change in change_list:
                    if abs((cash - change) - val) < 0.02:
                        adj_score += 150.0

            # Subtotal + Tax == Total Check
            for sub in subtotal_list:
                for tax in tax_list:
                    if abs((sub + tax) - val) < 0.02:
                        adj_score += 100.0

            # Penalty if candidate is explicitly equal to change amount
            if change_list and any(abs(val - chg) < 0.01 for chg in change_list) and not any(abs(val - c) < 0.01 for c in cash_list):
                adj_score -= 80.0

            validated_candidates.append((adj_score, val, r_text))

        total = 0.0
        if validated_candidates:
            validated_candidates.sort(key=lambda c: c[0], reverse=True)
            total = validated_candidates[0][1]

        # 7. Final Fallback: Largest non-cash, non-tax price in bottom 50%
        if total <= 0:
            for row in reversed(rows[int(total_rows * 0.35):]):
                row_text = " ".join(item["text"] for item in row).upper()
                if any(neg in row_text for neg in ["CASH", "TUNAI", "CHANGE", "BAKI", "SUBTOTAL", "TAX", "GST", "DISCOUNT"]):
                    continue
                found = self._extract_prices_from_row(row)
                if found:
                    total = found[0]
                    break

        return Receipt(company=company, date=date, total=total, category=ExpenseCategory.OTHER)


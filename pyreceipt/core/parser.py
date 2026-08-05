"""Regex-based Receipt Parser driven by external JSON language configurations.

Decouples language-specific extraction heuristics (anchors, exclusions, currency symbols)
from parser logic using Python's standard library.
"""

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

from pyreceipt.core.domain import ExpenseCategory, Receipt
from pyreceipt.utils.profiler import monitor_performance


class RegexReceiptParser:
    """JSON-driven anchor-based Regex parser for extracting structured receipt entities."""

    def __init__(
        self,
        lang_code: str = "it",
        config_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize RegexReceiptParser by loading language config JSON.

        Args:
            lang_code: Two-letter ISO language code (e.g., 'it', 'en').
            config_dir: Optional path to directory containing {lang_code}.json files.

        Raises:
            FileNotFoundError: If the specified language JSON configuration file does not exist.
        """
        self.lang_code = lang_code.lower()
        self.config = self._load_config(config_dir)
        self._compile_patterns()

    def _load_config(
        self, config_dir: Optional[Union[str, Path]]
    ) -> Dict[str, Any]:
        """Locate and load JSON language configuration file using pathlib and json.

        Args:
            config_dir: Optional custom directory path.

        Returns:
            Parsed JSON configuration dictionary.

        Raises:
            FileNotFoundError: If JSON file cannot be found.
        """
        possible_paths: List[Path] = []

        if config_dir:
            possible_paths.append(Path(config_dir) / f"{self.lang_code}.json")

        # Package relative path: pyreceipt/config/langs/{lang_code}.json
        pkg_root = Path(__file__).resolve().parent.parent
        possible_paths.append(pkg_root / "config" / "langs" / f"{self.lang_code}.json")

        # Project root path: ./config/langs/{lang_code}.json
        possible_paths.append(Path.cwd() / "config" / "langs" / f"{self.lang_code}.json")

        json_path: Optional[Path] = None
        for path in possible_paths:
            if path.is_file():
                json_path = path
                break

        if not json_path:
            raise FileNotFoundError(
                f"Language configuration file for '{self.lang_code}' not found."
            )

        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _compile_patterns(self) -> None:
        """Compile regex patterns dynamically from JSON configuration anchors."""
        total_anchors: List[str] = self.config.get("total_anchors", ["TOTAL", "TOTALE"])
        exclude_anchors: List[str] = self.config.get("exclude_anchors", ["SUBTOTAL"])
        currency_symbols: List[str] = self.config.get("currency_symbols", ["€", "$", "EUR", "USD"])
        date_patterns: List[str] = self.config.get(
            "date_patterns", [r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"]
        )

        # Build exclude pattern
        if exclude_anchors:
            escaped_excludes = [re.escape(a) for a in exclude_anchors]
            self.exclude_pattern = re.compile(
                r"(?i)\b(?:" + "|".join(escaped_excludes) + r")\b"
            )
        else:
            self.exclude_pattern = None

        # Build total anchor pattern with negative lookbehind for SUB
        escaped_anchors = [re.escape(a) for a in total_anchors]
        self.total_anchor_pattern = re.compile(
            r"(?i)(?<!SUB\s)(?<!SUB)(?<!SUB-)\b(?:" + "|".join(escaped_anchors) + r")\b"
        )

        # Build price pattern handling currency symbols ($ € EUR etc.) and decimals (, .)
        escaped_currencies = [re.escape(c) for c in currency_symbols]
        curr_group = r"(?:" + "|".join(escaped_currencies) + r")?"
        self.price_pattern = re.compile(
            curr_group + r"\s*([0-9]+[.,][0-9]{2})\b"
        )

        # Compile date patterns
        self.date_regexes = [re.compile(p) for p in date_patterns]

    @monitor_performance
    def parse(self, raw_text: str) -> Receipt:
        """Parse raw OCR text string into a structured Receipt domain entity.

        Args:
            raw_text: Text extracted from receipt image.

        Returns:
            Populated Receipt dataclass object.
        """
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

        company = self._extract_company(lines)
        date_val = self._extract_date(raw_text)
        total_val = self._extract_total(lines, raw_text)

        return Receipt(
            company=company,
            date=date_val,
            total=total_val,
            category=ExpenseCategory.OTHER,
        )

    def _extract_company(self, lines: List[str]) -> str:
        """Extract company name by taking first line with alphanumeric content.

        Args:
            lines: List of non-empty text lines.

        Returns:
            Cleaned company name string or 'UNKNOWN'.
        """
        for line in lines:
            if re.search(r"\w", line):
                cleaned = line.strip("*#=- ")
                return cleaned if cleaned else line
        return "UNKNOWN"

    def _extract_date(self, text: str) -> str:
        """Extract transaction date from text using compiled date patterns.

        Args:
            text: Full raw text string.

        Returns:
            Extracted date string or empty string.
        """
        for pattern in self.date_regexes:
            match = pattern.search(text)
            if match:
                return match.group(1)
        return ""

    def _extract_total(self, lines: List[str], full_text: str) -> float:
        """Extract total monetary amount using anchor-based regex and bottom-up scanning.

        Args:
            lines: Non-empty lines of raw OCR text.
            full_text: Complete raw OCR text string.

        Returns:
            Extracted float total amount or 0.0 if not found.
        """
        if not lines:
            return 0.0

        # Pass 1: Bottom-up line scan for anchor matching without excluded keywords
        for line in reversed(lines):
            if self.exclude_pattern and self.exclude_pattern.search(line):
                continue

            if self.total_anchor_pattern.search(line):
                price_match = self.price_pattern.search(line)
                if price_match:
                    try:
                        raw_num = price_match.group(1).replace(",", ".")
                        val = float(raw_num)
                        if val > 0:
                            return val
                    except ValueError:
                        continue

        # Pass 2: Anchor line with price on next line
        for i in reversed(range(len(lines))):
            line = lines[i]
            if self.exclude_pattern and self.exclude_pattern.search(line):
                continue

            if self.total_anchor_pattern.search(line):
                targets = [line]
                if i + 1 < len(lines):
                    targets.append(lines[i + 1])

                for target in targets:
                    price_match = self.price_pattern.search(target)
                    if price_match:
                        try:
                            raw_num = price_match.group(1).replace(",", ".")
                            val = float(raw_num)
                            if val > 0:
                                return val
                        except ValueError:
                            continue

        # Pass 3: Fallback full text match
        try:
            full_match = self.total_anchor_pattern.search(full_text)
            if full_match:
                price_match = self.price_pattern.search(full_text[full_match.start():])
                if price_match:
                    raw_num = price_match.group(1).replace(",", ".")
                    return float(raw_num)
        except Exception:
            pass

        return 0.0


# Backward-compatibility alias
ReceiptParser = RegexReceiptParser

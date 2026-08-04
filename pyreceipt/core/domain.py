"""Domain models for the PyReceipt application.

This module defines the core domain entities and value objects, strictly using
the Python Standard Library to maintain zero runtime dependencies on low-resource targets.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Union


@dataclass(frozen=True)
class ExpenseCategory:
    """Value Object representing an expense category.

    Supports pre-defined default categories while allowing dynamic creation
    of new custom categories at runtime during transaction entry.
    Category names are automatically stripped of whitespace and capitalized.

    Attributes:
        name: The normalized string name of the expense category.
    """

    name: str

    GROCERIES: ClassVar["ExpenseCategory"]
    RESTAURANT: ClassVar["ExpenseCategory"]
    TRANSPORT: ClassVar["ExpenseCategory"]
    OTHER: ClassVar["ExpenseCategory"]

    def __post_init__(self) -> None:
        """Normalize the category name upon initialization.

        Raises:
            ValueError: If the normalized category name is empty.
        """
        normalized = self.name.strip().upper()
        if not normalized:
            raise ValueError("Category name cannot be empty.")
        object.__setattr__(self, "name", normalized)

    @classmethod
    def from_name(cls, name: str) -> "ExpenseCategory":
        """Factory method to create an ExpenseCategory from a raw name string.

        Args:
            name: Raw category name provided during transaction entry.

        Returns:
            A normalized ExpenseCategory instance.
        """
        return cls(name=name)


# Pre-defined default category constants
ExpenseCategory.GROCERIES = ExpenseCategory("GROCERIES")
ExpenseCategory.RESTAURANT = ExpenseCategory("RESTAURANT")
ExpenseCategory.TRANSPORT = ExpenseCategory("TRANSPORT")
ExpenseCategory.OTHER = ExpenseCategory("OTHER")


@dataclass
class Receipt:
    """Core domain entity representing a receipt transaction.

    Attributes:
        company: Name of the vendor or merchant.
        date: Transaction date, represented either as a string or datetime object.
        total: Total monetary value of the purchase.
        category: ExpenseCategory value object assigned to this receipt.
    """

    company: str
    date: Union[str, datetime]
    total: float
    category: ExpenseCategory

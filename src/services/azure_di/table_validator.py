"""
Table validator for numerical continuity checks.

Provides numerical validation logic to verify if tables across pages
are continuous based on balance progression and column alignment.
"""
import logging
import re
from typing import List, Dict, Any

from src.core.config import settings

logger = logging.getLogger(__name__)


class TableValidator:
    """Validates numerical continuity for table merging."""

    def validate_numerical_continuity(
        self,
        previous_row: List[str],
        current_row: List[str],
        tolerance: float = None
    ) -> bool:
        """
        Check if two rows are numerically continuous (validate running balance).

        For bank statements, checks if the balance continues correctly between rows.

        Args:
            previous_row: Last row from previous table
            current_row: First row from current table
            tolerance: Acceptable difference for floating point comparison
                      (default: from settings)

        Returns:
            True if rows appear to be continuous, False otherwise
        """
        if tolerance is None:
            tolerance = settings.AZURE_DI_BALANCE_TOLERANCE

        prev_nums = self._extract_numeric_columns(previous_row)
        curr_nums = self._extract_numeric_columns(current_row)

        # Need at least some numbers in both rows
        if not prev_nums['has_numbers'] or not curr_nums['has_numbers']:
            logger.info("Numerical continuity: No numbers found in rows")
            return False

        # Primary check: Balance continuity
        prev_balance = prev_nums['balance']
        curr_balance = curr_nums['balance']

        if prev_balance is not None and curr_balance is not None:
            # Simple continuity check: current balance should be "close" to previous
            # In a continuous statement, balance changes by transactions
            # We check if the change is reasonable (not a huge jump)

            balance_diff = abs(curr_balance - prev_balance)

            # If balance is exactly the same, definitely continuous
            if balance_diff <= tolerance:
                logger.info(f"Numerical continuity: Same balance ({curr_balance:.2f})")
                return True

            # If balance changed, check if it's a reasonable transaction amount
            # Heuristic: If change is less than 50% of previous balance, likely continuous
            if prev_balance != 0:
                percent_change = balance_diff / abs(prev_balance)
                if percent_change < 0.5:  # Less than 50% change
                    logger.info(
                        f"Numerical continuity: Balance change is reasonable "
                        f"({prev_balance:.2f} → {curr_balance:.2f}, {percent_change*100:.1f}% change)"
                    )
                    return True
                else:
                    logger.info(
                        f"Numerical continuity: Balance change too large "
                        f"({prev_balance:.2f} → {curr_balance:.2f}, {percent_change*100:.1f}% change)"
                    )
                    return False

            # If previous balance was 0, accept any reasonable current balance
            if abs(curr_balance) < 1000000:  # Sanity check: balance < 1M
                logger.info(f"Numerical continuity: Starting from zero balance")
                return True

        # Fallback: Check if any numbers match positions (column alignment)
        prev_positions = set(idx for idx, _ in prev_nums['positions'])
        curr_positions = set(idx for idx, _ in curr_nums['positions'])

        if prev_positions and curr_positions:
            overlap = len(prev_positions & curr_positions)
            total = max(len(prev_positions), len(curr_positions))
            if overlap / total >= 0.5:  # At least 50% of columns have numbers in same positions
                logger.info(f"Numerical continuity: Column positions match ({overlap}/{total})")
                return True

        logger.info("Numerical continuity: Validation check failed (no match criteria met)")
        return False

    def _extract_numeric_columns(self, row: List[str]) -> Dict[str, Any]:
        """
        Extract numeric values from a row and identify balance/debit/credit columns.

        Args:
            row: List of cell values as strings

        Returns:
            Dict with keys:
            - 'amounts': List of all numeric values found
            - 'balance': Last numeric value (usually the balance column)
            - 'positions': List of (index, value) tuples for all numbers
            - 'has_numbers': Boolean indicating if any numbers were found
        """
        amounts = []
        positions = []

        for idx, cell in enumerate(row):
            if not cell:
                continue

            # Clean the cell value
            cell_clean = str(cell).strip()

            # Match numbers (including decimals, commas, and negatives)
            # Supports formats: 1,234.56 or 1234.56 or -1234.56
            number_pattern = r'-?\d+(?:,\d{3})*(?:\.\d+)?'
            matches = re.findall(number_pattern, cell_clean)

            for match in matches:
                try:
                    # Remove commas and convert to float
                    value = float(match.replace(',', ''))
                    amounts.append(value)
                    positions.append((idx, value))
                except ValueError:
                    continue

        result = {
            'amounts': amounts,
            'positions': positions,
            'balance': amounts[-1] if amounts else None,  # Last number is usually balance
            'has_numbers': len(amounts) > 0
        }

        return result

import re
import logging
from typing import Optional, Tuple # <--- ADD THIS LINE

# Use root logger
log = logging.getLogger(__name__)

def parse_salary(salary_text: Optional[str], target_currency: str = "USD") -> Tuple[Optional[int], Optional[int]]: # <-- Use Tuple here too
    """
    Very basic salary text parser. Tries to extract min/max annual salary.
    Assumes annual salary. Converts roughly based on common symbols.
    THIS IS A SIMPLISTIC IMPLEMENTATION AND MAY NEED SIGNIFICANT IMPROVEMENT.

    Args:
        salary_text: String containing salary information (e.g., "$80k - $100k", "£90,000 per year", "Up to 120000 EUR").
        target_currency: Target currency (currently ignored, needs proper conversion logic).

    Returns:
        A tuple (min_salary, max_salary), both Optional[int].
    """
    if not salary_text or not isinstance(salary_text, str) or not salary_text.strip():
        return None, None

    min_salary, max_salary = None, None
    # Pre-process text: lowercase, remove commas/currency symbols/common phrases
    text = salary_text.lower().replace(',', '').replace('per year', '').replace('annually', '')
    text = re.sub(r'[£$€]', '', text).strip() # Remove common currency symbols

    # Handle "k" for thousands AFTER removing symbols/commas
    text = re.sub(r'(\d+)\s*k', lambda m: str(int(m.group(1)) * 1000), text)

    # Look for explicit ranges (e.g., "80000 - 100000", "80000 to 100000")
    # Be more flexible with separators and spacing
    range_match = re.search(r'(\d+)\s*[-–to]+\s*(\d+)', text)
    if range_match:
        try:
            s1 = int(range_match.group(1))
            s2 = int(range_match.group(2))
            min_salary = min(s1, s2)
            max_salary = max(s1, s2)
            log.debug(f"Parsed range: {min_salary}-{max_salary} from '{salary_text}'")
            return min_salary, max_salary
        except ValueError:
            log.warning(f"Found range pattern but failed to convert numbers in: '{salary_text}'")


    # Look for "up to" or "max" variants
    up_to_match = re.search(r'(?:up to|max(?:imum)?|less than|under)\s*(\d+)', text)
    if up_to_match:
        try:
            max_salary = int(up_to_match.group(1))
            # Min salary is unknown
            log.debug(f"Parsed max salary: {max_salary} from '{salary_text}'")
            return None, max_salary
        except ValueError:
             log.warning(f"Found 'up to' pattern but failed to convert number in: '{salary_text}'")

    # Look for "minimum" or "starting at" or "from" variants
    min_match = re.search(r'(?:min(?:imum)?|starting at|from|over|above)\s*(\d+)', text)
    if min_match:
        try:
            min_salary = int(min_match.group(1))
            # Max salary is unknown
            log.debug(f"Parsed min salary: {min_salary} from '{salary_text}'")
            return min_salary, None
        except ValueError:
             log.warning(f"Found 'min' pattern but failed to convert number in: '{salary_text}'")

    # Look for a single plausible number if no ranges/keywords found
    # Avoid matching things like years (e.g., 2023)
    single_match = re.findall(r'\d+', text)
    plausible_salaries = [int(n) for n in single_match if 5000 < int(n) < 1000000] # Heuristic range for annual salary

    if len(plausible_salaries) == 1:
        salary_val = plausible_salaries[0]
        # Treat single value as both min and max for filtering purposes
        log.debug(f"Parsed single salary value: {salary_val} from '{salary_text}'")
        return salary_val, salary_val
    elif len(plausible_salaries) > 1:
         # Multiple numbers without clear range words - ambiguous. Could take min/max?
         log.warning(f"Ambiguous salary - multiple numbers found in '{salary_text}': {plausible_salaries}. Taking min/max.")
         return min(plausible_salaries), max(plausible_salaries)


    log.debug(f"Could not parse salary info from text: '{salary_text}'")
    return None, None


def normalize_string(text: Optional[str]) -> str:
    """Converts text to lowercase and strips whitespace."""
    if text is None:
        return ""
    # Check if it's already a string before lowercasing
    if isinstance(text, (str, bytes)):
         return str(text).lower().strip()
    else:
         # Handle potential non-string types gracefully (e.g., numbers)
         return str(text).strip()
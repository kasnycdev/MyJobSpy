import re
import logging

def parse_salary(salary_text: Optional[str], target_currency: str = "USD") -> tuple[Optional[int], Optional[int]]:
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
    if not salary_text:
        return None, None

    min_salary, max_salary = None, None
    text = salary_text.lower().replace(',', '').replace('per year', '').replace('annually', '').strip()

    # Handle "k" for thousands
    text = re.sub(r'(\d+)k', lambda m: str(int(m.group(1)) * 1000), text)

    # Look for ranges (e.g., "80000 - 100000", "80000 to 100000")
    range_match = re.search(r'(\d+)\s*[-–to]+\s*(\d+)', text)
    if range_match:
        min_salary = int(range_match.group(1))
        max_salary = int(range_match.group(2))
        if min_salary > max_salary: # Swap if order is wrong
            min_salary, max_salary = max_salary, min_salary
        return min_salary, max_salary

    # Look for "up to" or "max"
    up_to_match = re.search(r'(?:up to|max(?:imum)?)\s*(\d+)', text)
    if up_to_match:
        max_salary = int(up_to_match.group(1))
        # We don't know the minimum in this case
        return None, max_salary

     # Look for "minimum" or "starting at"
    min_match = re.search(r'(?:min(?:imum)?|starting at)\s*(\d+)', text)
    if min_match:
        min_salary = int(min_match.group(1))
         # We don't know the maximum
        return min_salary, None

    # Look for a single number
    single_match = re.findall(r'\d+', text)
    if len(single_match) == 1:
        # Could be min, max, or exact - treat as both for broadest match possibility
        salary_val = int(single_match[0])
        # Heuristic: if it's < 1000 assume it might be hourly/monthly - ignore for annual for now
        # This is very rough!
        if salary_val > 5000: # Arbitrary threshold for likely annual salary
             return salary_val, salary_val
        else:
             logging.debug(f"Ignoring potentially non-annual salary value: {salary_val} in '{salary_text}'")
             return None, None
    elif len(single_match) > 1:
         # Multiple numbers without clear range words - could be complex, take highest/lowest?
         nums = sorted([int(n) for n in single_match if int(n) > 5000]) # Simple filter
         if len(nums) >= 2:
             return nums[0], nums[-1]
         elif len(nums) == 1:
             return nums[0], nums[0]

    logging.debug(f"Could not parse salary range from text: '{salary_text}'")
    return None, None


def normalize_string(text: Optional[str]) -> str:
    """Converts text to lowercase and strips whitespace."""
    if text is None:
        return ""
    return text.lower().strip()
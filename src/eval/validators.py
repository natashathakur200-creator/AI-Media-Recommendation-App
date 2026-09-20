import json
from typing import Any, Dict, List, Tuple


# Maximum allowed output sizes
DEFAULT_MAX_SUMMARY_CHARS = 1200
DEFAULT_MAX_ACTION_ITEMS = 10


# Basic safety blocklist
# These phrases should never appear in a normal
# structured business-analysis response.
BLOCKLIST = [
    "system prompt",
    "developer message",
    "ignore previous instructions",
    "reveal the system prompt",
    "reveal the developer message",
]


def parse_json_strict(text: str) -> Tuple[bool, Dict[str, Any], str]:
    """
    Parse model output as strict JSON.

    Returns:
        (success, parsed_data, error_message)
    """

    try:
        data = json.loads(text)

        if not isinstance(data, dict):
            return (
                False,
                {},
                "Output JSON is not an object."
            )

        return True, data, ""

    except Exception as error:
        return (
            False,
            {},
            f"Invalid JSON: {error}"
        )


def validate_required_keys(
    data: Dict[str, Any],
    required: List[str]
) -> List[str]:
    """
    Check that all required keys exist.
    """

    missing = [
        key
        for key in required
        if key not in data
    ]

    return missing


def validate_urgency(
    data: Dict[str, Any],
    allowed: List[str]
) -> str:
    """
    Validate the urgency enum.
    """

    urgency = data.get("urgency")

    if urgency is None:
        return "Missing urgency."

    if not isinstance(urgency, str):
        return "Urgency must be a string."

    if urgency.lower() not in {
        value.lower()
        for value in allowed
    }:
        return f"Urgency '{urgency}' not allowed."

    return ""


def validate_summary(
    data: Dict[str, Any],
    max_chars: int
) -> str:
    """
    Validate summary type, length and emptiness.
    """

    summary = data.get("summary")

    if not isinstance(summary, str):
        return "summary must be a string."

    if len(summary) > max_chars:
        return (
            f"summary too long: "
            f"{len(summary)} > {max_chars} chars."
        )

    if not summary.strip():
        return "summary is empty."

    return ""


def validate_budget(data: Dict[str, Any]) -> str:
    """
    Validate the budget field.

    The model may return a number or a string because
    business budgets can be written in formats such as
    'INR 50,000 per month'.
    """

    budget = data.get("budget")

    if budget is None:
        return ""

    if not isinstance(budget, (str, int, float)):
        return "budget must be a string, number, or null."

    return ""


def validate_string_or_null(
    data: Dict[str, Any],
    field_name: str
) -> str:
    """
    Validate fields that should contain either a string
    or null.
    """

    value = data.get(field_name)

    if value is not None and not isinstance(value, str):
        return f"{field_name} must be a string or null."

    return ""


def safety_check(text: str) -> str:
    """
    Basic safety check on raw model output.

    This catches common prompt-leakage and instruction
    override phrases before the output is accepted.
    """

    normalized_text = (text or "").lower()

    for phrase in BLOCKLIST:

        if phrase in normalized_text:
            return (
                f"Blocked phrase detected: '{phrase}'"
            )

    return ""


def evaluate_output(
    raw_model_text: str,
    required_keys: List[str],
    urgency_allowed: List[str],
    max_summary_chars: int = DEFAULT_MAX_SUMMARY_CHARS,
    max_action_items: int = DEFAULT_MAX_ACTION_ITEMS,
) -> Dict[str, Any]:
    """
    Complete validation pipeline.

    Order:

    1. Safety check
    2. Strict JSON parsing
    3. Required key validation
    4. Urgency validation
    5. Summary validation
    6. Field-type validation

    Returns a structured evaluation result.
    """

    # --------------------------------------------------
    # 1. Safety check BEFORE parsing
    # --------------------------------------------------

    safety_error = safety_check(
        raw_model_text
    )

    if safety_error:
        return {
            "pass": False,
            "errors": [safety_error]
        }

    # --------------------------------------------------
    # 2. Strict JSON parsing
    # --------------------------------------------------

    ok, data, parse_error = parse_json_strict(
        raw_model_text
    )

    if not ok:
        return {
            "pass": False,
            "errors": [parse_error]
        }

    # --------------------------------------------------
    # 3. Required keys
    # --------------------------------------------------

    errors: List[str] = []

    missing_keys = validate_required_keys(
        data,
        required_keys
    )

    if missing_keys:
        errors.append(
            f"Missing keys: {missing_keys}"
        )

    # --------------------------------------------------
    # 4. Urgency
    # --------------------------------------------------

    urgency_error = validate_urgency(
        data,
        urgency_allowed
    )

    if urgency_error:
        errors.append(urgency_error)

    # --------------------------------------------------
    # 5. Summary
    # --------------------------------------------------

    summary_error = validate_summary(
        data,
        max_summary_chars
    )

    if summary_error:
        errors.append(summary_error)

    # --------------------------------------------------
    # 6. Field types
    # --------------------------------------------------

    string_fields = [
        "industry",
        "business_name",
        "location",
        "target_audience",
        "objective",
    ]

    for field_name in string_fields:

        field_error = validate_string_or_null(
            data,
            field_name
        )

        if field_error:
            errors.append(field_error)

    # --------------------------------------------------
    # Budget
    # --------------------------------------------------

    budget_error = validate_budget(data)

    if budget_error:
        errors.append(budget_error)

    # --------------------------------------------------
    # Final result
    # --------------------------------------------------

    return {
        "pass": len(errors) == 0,
        "errors": errors,
        "parsed": data
    }
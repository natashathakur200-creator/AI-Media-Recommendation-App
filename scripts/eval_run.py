import os
import sys
import json
import time
import re
from pathlib import Path
from typing import Dict, Any, List


# ============================================================
# Add project root to Python path
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Imports
# ============================================================

from src.eval.validators import evaluate_output
from src.workflow_steps import step2_extract_structured


# ============================================================
# File paths
# ============================================================

TEST_CASES_PATH = (
    Path(PROJECT_ROOT)
    / "tests"
    / "test_cases.json"
)

REPORT_PATH = (
    Path(PROJECT_ROOT)
    / "reports"
    / "eval_report.json"
)


# ============================================================
# Quota helpers
# ============================================================

def is_quota_error(error_message: str) -> bool:
    """
    Detect Gemini quota/rate-limit errors.
    """

    if not error_message:
        return False

    text = error_message.lower()

    indicators = [
        "429",
        "resource_exhausted",
        "quota exceeded",
        "free_tier_requests",
        "ratelimit",
        "rate limit",
    ]

    return any(
        indicator in text
        for indicator in indicators
    )


def get_retry_seconds(
    error_message: str,
    default_seconds: int = 65
) -> int:
    """
    Extract retry time from Gemini error message.

    Example:
        retry in 31s
        retry in 57 seconds
    """

    if not error_message:
        return default_seconds

    match = re.search(
        r"retry in\s+(\d+)\s*(?:s|seconds)?",
        error_message.lower()
    )

    if match:
        seconds = int(match.group(1))

        # Add a small safety buffer.
        return seconds + 5

    return default_seconds


# ============================================================
# Call system under test
# ============================================================

def call_system_under_test(
    user_input: str
) -> Dict[str, Any]:
    """
    Call the Day 11 system under test.

    Returns:
        raw_output
        system_error
        quota_error
    """

    try:

        result = step2_extract_structured(
            user_input
        )

        # ----------------------------------------------------
        # Handle dictionary result
        # ----------------------------------------------------

        if isinstance(result, dict):

            # Normal successful result
            if "data" in result:

                raw_output = json.dumps(
                    result["data"],
                    ensure_ascii=False
                )

                return {
                    "raw_output": raw_output,
                    "system_error": "",
                    "quota_error": False
                }

            # Some implementations may return output directly
            if "output" in result:

                raw_output = result["output"]

                if isinstance(
                    raw_output,
                    dict
                ):
                    raw_output = json.dumps(
                        raw_output,
                        ensure_ascii=False
                    )

                return {
                    "raw_output": str(raw_output),
                    "system_error": "",
                    "quota_error": False
                }

            # ------------------------------------------------
            # Error returned by workflow
            # ------------------------------------------------

            error_message = str(
                result.get(
                    "error",
                    ""
                )
            )

            if is_quota_error(
                error_message
            ):

                return {
                    "raw_output": "",
                    "system_error": error_message,
                    "quota_error": True
                }

            return {
                "raw_output": "",
                "system_error": error_message,
                "quota_error": False
            }

        # ----------------------------------------------------
        # Direct string result
        # ----------------------------------------------------

        if isinstance(result, str):

            return {
                "raw_output": result,
                "system_error": "",
                "quota_error": False
            }

        # ----------------------------------------------------
        # Unexpected result
        # ----------------------------------------------------

        return {
            "raw_output": "",
            "system_error": (
                "Unexpected system result type: "
                + str(type(result))
            ),
            "quota_error": False
        }

    except Exception as error:

        error_message = str(error)

        if is_quota_error(
            error_message
        ):

            return {
                "raw_output": "",
                "system_error": error_message,
                "quota_error": True
            }

        return {
            "raw_output": "",
            "system_error": error_message,
            "quota_error": False
        }


# ============================================================
# Quota recovery
# ============================================================

def run_test_with_quota_recovery(
    user_input: str,
    max_quota_retries: int = 3
) -> Dict[str, Any]:
    """
    Run one test and automatically recover from Gemini
    quota/rate-limit errors.

    IMPORTANT:
    Quota-skipped tests are NOT considered passed.
    """

    retry_count = 0

    while True:

        result = call_system_under_test(
            user_input
        )

        # ----------------------------------------------------
        # Successful call
        # ----------------------------------------------------

        if not result["quota_error"]:

            return result

        # ----------------------------------------------------
        # Maximum retry count reached
        # ----------------------------------------------------

        if retry_count >= max_quota_retries:

            return result

        retry_count += 1

        wait_seconds = get_retry_seconds(
            result["system_error"]
        )

        print(
            f"  Gemini quota reached. "
            f"Waiting {wait_seconds} seconds "
            f"before retry "
            f"({retry_count}/{max_quota_retries})..."
        )

        time.sleep(
            wait_seconds
        )


# ============================================================
# Load test cases
# ============================================================

def load_test_cases() -> List[Dict[str, Any]]:

    if not TEST_CASES_PATH.exists():

        raise FileNotFoundError(
            f"Test cases file not found: "
            f"{TEST_CASES_PATH}"
        )

    with open(
        TEST_CASES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    if not isinstance(
        data,
        list
    ):

        raise ValueError(
            "test_cases.json must contain a JSON list."
        )

    return data


# ============================================================
# Main evaluator
# ============================================================

def main():

    print()
    print("=" * 60)
    print("DAY 11 - AI MEDIA PLANNING EVALUATION")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # Load cases
    # --------------------------------------------------------

    cases = load_test_cases()

    print(
        f"Loaded {len(cases)} test cases."
    )

    print()

    # --------------------------------------------------------
    # Create reports folder
    # --------------------------------------------------------

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Report structure
    # --------------------------------------------------------

    report = {
        "total": len(cases),
        "passed": 0,
        "failed": 0,
        "skipped_quota": 0,
        "results": []
    }

    # --------------------------------------------------------
    # Run tests
    # --------------------------------------------------------

    for index, case in enumerate(
        cases
    ):

        case_id = case.get(
            "id",
            f"test_{index + 1}"
        )

        user_input = case.get(
            "input",
            ""
        )

        print(
            f"Running: {case_id}"
        )

        # ----------------------------------------------------
        # Small spacing between tests.
        #
        # This helps avoid immediately triggering a
        # rate-limit after a successful request.
        # ----------------------------------------------------

        if index > 0:

            print(
                "  Waiting 15 seconds "
                "before next test..."
            )

            time.sleep(15)

        # ----------------------------------------------------
        # Run model with quota recovery
        # ----------------------------------------------------

        system_result = (
            run_test_with_quota_recovery(
                user_input=user_input
            )
        )

        raw_output = system_result[
            "raw_output"
        ]

        system_error = system_result[
            "system_error"
        ]

        quota_error = system_result[
            "quota_error"
        ]

        # ----------------------------------------------------
        # Quota still unavailable
        # ----------------------------------------------------

        if quota_error:

            # Defensive initialization.
            # This prevents KeyError even if the report
            # structure changes in the future.

            report["skipped_quota"] = (
                report.get(
                    "skipped_quota",
                    0
                ) + 1
            )

            report["results"].append({

                "id": case_id,

                "status": "QUOTA_SKIPPED",

                "pass": None,

                "errors": [
                    system_error
                ]

            })

            print(
                "  QUOTA_SKIPPED"
            )

            print()

            continue

        # ----------------------------------------------------
        # System-level error
        # ----------------------------------------------------

        if system_error:

            report["failed"] += 1

            report["results"].append({

                "id": case_id,

                "status": "FAILED",

                "pass": False,

                "errors": [
                    system_error
                ]

            })

            print(
                "  FAILED - System error"
            )

            print()

            continue

        # ----------------------------------------------------
        # Validate AI output
        # ----------------------------------------------------

        validation = evaluate_output(

            raw_model_text=raw_output,

            required_keys=[
                "industry",
                "business_name",
                "location",
                "target_audience",
                "objective",
                "budget",
                "urgency",
                "summary"
            ],

            urgency_allowed=[
                "low",
                "medium",
                "high"
            ],

            max_summary_chars=1200,

            max_action_items=10
        )

        # ----------------------------------------------------
        # PASS
        # ----------------------------------------------------

        if validation.get(
            "pass",
            False
        ):

            report["passed"] += 1

            report["results"].append({

                "id": case_id,

                "status": "PASSED",

                "pass": True,

                "errors": [],

                "output": validation.get(
                    "parsed",
                    {}
                )

            })

            print(
                "  PASSED"
            )

        # ----------------------------------------------------
        # FAIL
        # ----------------------------------------------------

        else:

            errors = validation.get(
                "errors",
                [
                    "Unknown validation error."
                ]
            )

            report["failed"] += 1

            report["results"].append({

                "id": case_id,

                "status": "FAILED",

                "pass": False,

                "errors": errors,

                "output": validation.get(
                    "parsed",
                    {}
                )

            })

            print(
                "  FAILED"
            )

            for error in errors:

                print(
                    f"    - {error}"
                )

        print()

    # ========================================================
    # Summary
    # ========================================================

    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    print(
        f"Total tests:     {report['total']}"
    )

    print(
        f"Passed:          {report['passed']}"
    )

    print(
        f"Failed:          {report['failed']}"
    )

    print(
        f"Quota skipped:   {report['skipped_quota']}"
    )

    # --------------------------------------------------------
    # Genuine pass rate
    # --------------------------------------------------------

    completed_tests = (
        report["passed"]
        + report["failed"]
    )

    if completed_tests > 0:

        pass_rate = (
            report["passed"]
            / completed_tests
        ) * 100

    else:

        pass_rate = 0

    print(
        f"Completed pass rate: "
        f"{pass_rate:.1f}%"
    )

    # --------------------------------------------------------
    # Important interpretation
    # --------------------------------------------------------

    if report["skipped_quota"] > 0:

        print()
        print(
            "WARNING:"
        )

        print(
            "Some tests were skipped because "
            "Gemini quota/rate limits were reached."
        )

        print(
            "Quota-skipped tests are NOT counted "
            "as passed."
        )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2
        )

    print()
    print(
        f"Report saved to:"
    )

    print(
        REPORT_PATH
    )

    print()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()
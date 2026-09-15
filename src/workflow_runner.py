import os
from typing import Dict, Any

from src.workflow_steps import (
    step1_load_input,
    step2_extract_structured,
    step3_classify_and_route,
    step4_generate_strategy,
    step5_save_outputs,
    step6_log_run,
)


def run_workflow(
    input_file: str,
    out_dir: str = "outbox",
    log_file: str = "logs/workflow.log"
) -> Dict[str, Any]:

    # =====================================================
    # STEP 1 — LOAD INPUT
    # =====================================================

    s1 = step1_load_input(input_file)

    if not s1["ok"]:

        step6_log_run(
            log_file,
            {
                "ok": False,
                "step": 1,
                "error": s1.get("error"),
                "file": input_file,
            }
        )

        return s1


    # =====================================================
    # STEP 2 — EXTRACT STRUCTURED DATA
    # =====================================================

    s2 = step2_extract_structured(
        s1["raw_text"]
    )

    if not s2["ok"]:

        step6_log_run(
            log_file,
            {
                "ok": False,
                "step": 2,
                "error": s2.get("error"),
                "file": input_file,
            }
        )

        return s2


    extracted = s2["extracted"]


    # =====================================================
    # STEP 3 — CLASSIFY AND ROUTE
    # =====================================================

    s3 = step3_classify_and_route(
        extracted
    )

    if not s3["ok"]:

        step6_log_run(
            log_file,
            {
                "ok": False,
                "step": 3,
                "error": s3.get("error"),
                "file": input_file,
            }
        )

        return s3


    # =====================================================
    # STEP 4 — GENERATE MEDIA STRATEGY
    # =====================================================

    s4 = step4_generate_strategy(
        extracted,
        s3["priority"],
        s3["sla"]
    )

    if not s4["ok"]:

        step6_log_run(
            log_file,
            {
                "ok": False,
                "step": 4,
                "error": s4.get("error"),
                "file": input_file,
            }
        )

        return s4


    # =====================================================
    # COMBINE WORKFLOW RESULTS
    # =====================================================

    payload = {
        "input_file": input_file,
        "extracted": extracted,
        "priority": s3["priority"],
        "sla": s3["sla"],
        "draft_reply": s4["draft_reply"],
    }


    # =====================================================
    # STEP 5 — SAVE OUTPUTS
    # =====================================================

    base_name = os.path.splitext(
        os.path.basename(input_file)
    )[0]

    output_dir = os.path.join(
        out_dir,
        base_name
    )

    s5 = step5_save_outputs(
        output_dir,
        payload
    )

    if not s5["ok"]:

        step6_log_run(
            log_file,
            {
                "ok": False,
                "step": 5,
                "error": s5.get("error"),
                "file": input_file,
            }
        )

        return s5


    # =====================================================
    # STEP 6 — LOG WORKFLOW
    # =====================================================

    step6_log_run(
        log_file,
        {
            "ok": True,
            "file": input_file,
            "output_dir": output_dir,
        }
    )


    # =====================================================
    # FINAL RESULT
    # =====================================================

    return {
        "ok": True,
        "output_dir": output_dir,
        "saved": s5,
    }
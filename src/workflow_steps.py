import os
import json
import time
from typing import Dict, Any

from dotenv import load_dotenv
from google import genai
from google.genai import types


MODEL = "gemini-3.6-flash"


# ============================================================
# GEMINI CLIENT
# ============================================================

def get_gemini_client():

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY in .env"
        )

    return genai.Client(
        api_key=api_key
    )


# ============================================================
# STEP 1 — LOAD INPUT
# ============================================================

def step1_load_input(
    file_path: str
) -> Dict[str, Any]:

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            text = f.read()

        return {
            "ok": True,
            "input_path": file_path,
            "raw_text": text
        }

    except Exception as error:

        return {
            "ok": False,
            "error": str(error)
        }


# ============================================================
# STEP 2 — EXTRACT STRUCTURED BUSINESS DATA
# ============================================================

def step2_extract_structured(
    raw_text: str
) -> Dict[str, Any]:

    try:

        client = get_gemini_client()

        system_instruction = """
You are an AI business and media planning analyst.

Extract structured information from the supplied business brief.

Return ONLY valid JSON.

Use exactly these keys:

industry
business_name
location
target_audience
objective
budget
urgency
summary

Rules:

- Do not invent information.
- If information is missing, use null.
- Keep the summary concise.
- urgency must be one of:
  low
  medium
  high

Return JSON only.
"""

        response = client.models.generate_content(

            model=MODEL,

            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text=raw_text
                        )
                    ]
                )
            ],

            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0
            )
        )

        content = (
            response.text or ""
        ).strip()

        # Remove accidental markdown fences
        if content.startswith("```"):

            content = content.replace(
                "```json",
                ""
            ).replace(
                "```",
                ""
            ).strip()

        data = json.loads(content)

        required_keys = {
            "industry",
            "business_name",
            "location",
            "target_audience",
            "objective",
            "budget",
            "urgency",
            "summary"
        }

        missing = (
            required_keys
            - set(data.keys())
        )

        if missing:

            return {
                "ok": False,
                "error": (
                    f"Missing keys: "
                    f"{sorted(list(missing))}"
                ),
                "raw": content
            }

        return {
            "ok": True,
            "extracted": data
        }

    except Exception as error:

        return {
            "ok": False,
            "error": str(error)
        }


# ============================================================
# STEP 3 — CLASSIFY AND PRIORITIZE
# ============================================================

def step3_classify_and_route(
    extracted: Dict[str, Any]
) -> Dict[str, Any]:

    urgency = (
        extracted.get("urgency")
        or "medium"
    ).lower()

    if urgency not in {
        "low",
        "medium",
        "high"
    }:

        urgency = "medium"


    if urgency == "high":

        priority = "priority"
        sla = "4 hours"

    elif urgency == "medium":

        priority = "standard"
        sla = "24 hours"

    else:

        priority = "low"
        sla = "72 hours"


    return {
        "ok": True,
        "priority": priority,
        "sla": sla
    }


# ============================================================
# STEP 4 — GENERATE MEDIA STRATEGY
# ============================================================

def step4_generate_strategy(
    extracted: Dict[str, Any],
    priority: str,
    sla: str
) -> Dict[str, Any]:

    try:

        client = get_gemini_client()

        business_data = json.dumps(
            extracted,
            ensure_ascii=False,
            indent=2
        )

        prompt = f"""
You are an AI Media Strategist.

Create a preliminary media strategy based ONLY
on the supplied business information.

Business information:

{business_data}

Priority:
{priority}

Response SLA:
{sla}

Provide:

1. Business situation
2. Target audience
3. Marketing objective
4. Recommended media channels
5. Suggested budget allocation approach
6. First 30-day action plan
7. Important assumptions

Important rules:

- Do not invent market statistics.
- Do not invent competitor data.
- Do not claim guaranteed ROI.
- Clearly label assumptions.
- Keep recommendations practical for an SME.
"""

        response = client.models.generate_content(

            model=MODEL,

            contents=prompt,

            config=types.GenerateContentConfig(
                temperature=0.3
            )
        )

        draft = (
            response.text or ""
        ).strip()

        return {
            "ok": True,
            "draft_reply": draft
        }

    except Exception as error:

        return {
            "ok": False,
            "error": str(error)
        }


# ============================================================
# STEP 5 — SAVE OUTPUTS
# ============================================================

def step5_save_outputs(
    out_base: str,
    payload: Dict[str, Any]
) -> Dict[str, Any]:

    try:

        os.makedirs(
            out_base,
            exist_ok=True
        )


        # ----------------------------------------------
        # JSON RESULT
        # ----------------------------------------------

        json_path = os.path.join(
            out_base,
            "result.json"
        )

        with open(
            json_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                payload,
                f,
                ensure_ascii=False,
                indent=2
            )


        # ----------------------------------------------
        # MEDIA STRATEGY REPORT
        # ----------------------------------------------

        report_path = os.path.join(
            out_base,
            "media_strategy.txt"
        )

        with open(
            report_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                payload.get(
                    "draft_reply",
                    ""
                )
            )


        return {
            "ok": True,
            "json_path": json_path,
            "report_path": report_path
        }

    except Exception as error:

        return {
            "ok": False,
            "error": str(error)
        }


# ============================================================
# STEP 6 — LOG WORKFLOW RUN
# ============================================================

def step6_log_run(
    log_path: str,
    record: Dict[str, Any]
) -> None:

    os.makedirs(
        os.path.dirname(log_path),
        exist_ok=True
    )

    log_record = dict(record)

    log_record["timestamp"] = int(
        time.time()
    )

    with open(
        log_path,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(
                log_record,
                ensure_ascii=False
            )
            + "\n"
        )
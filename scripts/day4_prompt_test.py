import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

def load_rulebook() -> str:
    """
    Load the AI media-planning rulebook from the prompts folder.
    """
    project_root = Path(__file__).resolve().parents[1]
    rulebook_path = project_root / "prompts" / "media_rulebook.md"

    if not rulebook_path.exists():
        raise FileNotFoundError(
            f"Rulebook not found: {rulebook_path}"
        )

    return rulebook_path.read_text(encoding="utf-8")

SYSTEM_PROMPT =load_rulebook()
def main():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise SystemExit(
            "Missing GEMINI_API_KEY. Put it in your .env file."
        )

    client = genai.Client(api_key=api_key)

    user_input = """
Create a media plan for SRK Web Innovation.

Business type: B2B IT services
Location: Jaipur, Rajasthan
Target customers: Small and medium-sized businesses
Marketing objective: Generate qualified B2B leads
Marketing budget: ₹10,00,000
Planning period: 6 months

The company wants to evaluate digital, print, events, networking, SEO, and other relevant media channels.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=user_input,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT
        ),
    )

    print("\n===== DAY 4 RULEBOOK TEST =====\n")
    print(response.text)


if __name__ == "__main__":
    main()
import os

from dotenv import load_dotenv
from google import genai


def build_prompt(user_text: str) -> str:
    """
    Build a consistent instruction for the AI.
    Later, this can be replaced with media-planning instructions.
    """
    return f"""
You are a helpful AI assistant.

Rules:
- Be concise and clear.
- Give accurate information.
- Do not invent facts.
- If you are unsure, say that you are unsure.

User question:
{user_text}
"""


def call_llm(client: genai.Client, user_text: str) -> str:
    """
    Send the user's request to Gemini and return the response.
    """
    prompt = build_prompt(user_text)

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    return response.text.strip()


def main():
    # Load variables from the .env file
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise SystemExit(
            "Missing GEMINI_API_KEY. Put it in your .env file."
        )

    # Create the Gemini client
    client = genai.Client(api_key=api_key)

    print("CLI AI Assistant using Gemini")
    print("Type 'exit' to quit.")

    while True:
        user_text = input("\nYou: ").strip()

        if user_text.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not user_text:
            print("Please enter a question.")
            continue

        try:
            answer = call_llm(client, user_text)
            print("\nAssistant:", answer)

        except Exception as e:
            print("\nError calling the Gemini API:", str(e))


if __name__ == "__main__":
    main()
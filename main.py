from dotenv import load_dotenv
import os
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

while True:
    question = input("\nYou: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=question
    )

    print("AI:", response.text)
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json

from src.ai_json_function import analyze_text_to_validated_json
import json

from src.ai_json_function import analyze_text_to_validated_json


def main():
    business_input = """
Business: SRK Web Innovation
Industry: B2B IT Services
Location: Jaipur, Rajasthan
Target audience: Small and medium-sized businesses
Marketing objective: Generate qualified B2B leads
Marketing budget: ₹10,00,000
Planning period: 6 months
""".strip()

    result = analyze_text_to_validated_json(business_input)

    print("\n===== VALIDATED MEDIA RECOMMENDATION =====\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
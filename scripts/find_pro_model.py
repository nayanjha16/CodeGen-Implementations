
import google.generativeai as genai
import os
import sys

# Candidates for High Accuracy Pro models
candidates = [
    "gemini-1.5-pro",
    "gemini-1.5-pro-latest",
    "gemini-1.5-pro-001",
    "gemini-1.5-pro-002",
    "models/gemini-1.5-pro",
    "models/gemini-1.5-pro-latest",
    "models/gemini-1.5-pro-001",
    "models/gemini-1.5-pro-002",
    "gemini-pro",
    "models/gemini-pro"
]

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    # Try args
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        print("Need API Key")
        sys.exit(1)

genai.configure(api_key=api_key)

print(f"Testing {len(candidates)} model aliases...")

for model_name in candidates:
    print(f"Testing: {model_name}...", end=" ")
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Hello")
        if response.text:
            print("✅ SUCCESS!")
            print(f"\nWINNER: {model_name}")
            sys.exit(0)
    except Exception as e:
        print(f"❌ Failed ({str(e)[:50]}...)")
        
print("\nNo working Pro model found in candidates.")

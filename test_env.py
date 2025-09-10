import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
print("Looking for .env at:", dotenv_path)
load_dotenv(dotenv_path)

print("Groq Key:", os.getenv("GROQ_API_KEY"))
print("Model:", os.getenv("GROQ_MODEL"))
print("SerpAPI:", os.getenv("SERPAPI_KEY"))

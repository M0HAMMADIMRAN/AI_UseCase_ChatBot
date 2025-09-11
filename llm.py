# llm.py
import os
from langchain_groq import ChatGroq

def get_chatgroq_model():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Please set it in .env or Streamlit secrets.")
    
    model = ChatGroq(
        groq_api_key=api_key,
        model="llama-3.1-8b-instant"  # or another available model
    )
    return model

import os
from langchain_groq import ChatGroq

def get_chatgroq_model():
    """Initialize and return the Groq chat model"""
    try:
        groq_model = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        )
        return groq_model
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Groq model: {str(e)}")

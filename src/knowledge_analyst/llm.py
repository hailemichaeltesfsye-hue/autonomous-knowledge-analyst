"""
Groq LLM Client configuration for Autonomous Knowledge Analyst.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-120b"

def get_llm(model_name: str = DEFAULT_MODEL, temperature: float = 0.2) -> ChatGroq:
    """
    Get an initialized ChatGroq LLM instance.
    
    Args:
        model_name: Groq model name (default: openai/gpt-oss-120b)
        temperature: LLM sampling temperature
        
    Returns:
        ChatGroq instance
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable is not set. "
            "Please create a .env file with your GROQ_API_KEY."
        )
        
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=temperature
    )

if __name__ == "__main__":
    try:
        llm = get_llm()
        res = llm.invoke("Say 'Groq LLM initialized successfully' in 5 words.")
        print(res.content)
    except Exception as e:
        print(f"LLM initialization test note: {e}")

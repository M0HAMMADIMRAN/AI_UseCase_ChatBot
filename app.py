import streamlit as st
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from config import config  # loads .env
from llm import get_chatgroq_model
from utils.rag_utils import chunk_documents, build_index, query_index
from utils.web_search import web_search
from utils.prompt_templates import CONCISE_PROMPT, DETAILED_PROMPT

import io
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None
try:
    import docx
except Exception:
    docx = None
# --------------------------------------------------
# Human-like conversational prompt
# --------------------------------------------------
HUMAN_PROMPT = """
You are a friendly and helpful AI assistant. 
- If the user is having a casual chat (greetings, feelings, opinions), respond naturally like a human.
- If the user asks about knowledge (documents, facts, web), use the context provided and give a clear, helpful answer.
- Always keep your tone conversational and engaging.
- If unsure, politely say so instead of hallucinating.
"""

# --------------------------------------------------
# Helper function to get model response
# --------------------------------------------------
def get_chat_response(chat_model, messages, system_prompt):
    try:
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        formatted_messages = [SystemMessage(content=system_prompt)]
        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            else:
                formatted_messages.append(AIMessage(content=msg["content"]))
        response = chat_model.invoke(formatted_messages)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return f"Error getting response: {str(e)}"

# --------------------------------------------------
# Main chat page
# --------------------------------------------------
def chat_page():
    st.title("AI ChatBot with RAG + Web Search")

    # Initialize Groq model
    chat_model = None
    try:
        chat_model = get_chatgroq_model()
    except Exception as e:
        st.warning(f"Model init warning: {e}")

    # Initialize state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "saved_chats" not in st.session_state:
        st.session_state.saved_chats = []

    # Sidebar for managing chats
    st.sidebar.header("Chat Sessions")
    if st.sidebar.button("New Chat"):
        st.session_state.messages = []
    if st.sidebar.button("Save Chat"):
        if st.session_state.messages:
            st.session_state.saved_chats.append(st.session_state.messages.copy())
            st.session_state.messages = []
    for i, chat in enumerate(st.session_state.saved_chats):
        if st.sidebar.button(f"Chat {i+1}"):
            st.session_state.messages = chat.copy()

    # Upload & index documents
    with st.expander("Upload and Index Documents"):
        uploaded_files = st.file_uploader(
        "Upload documents (pdf, txt, docx). Text will be extracted and indexed.",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
        )
        if uploaded_files:
            docs = []
            for f in uploaded_files:
                fname = f.name.lower()
                try:
                    data = f.read()  # bytes
                    if fname.endswith(".pdf") and PdfReader is not None:
                    # parse PDF
                        reader = PdfReader(io.BytesIO(data))
                        text = ""
                        for p in reader.pages:
                            t = p.extract_text()
                            if t:
                                text += t + "\n"
                    elif fname.endswith(".docx") and docx is not None:
                    # parse docx
                        doc = docx.Document(io.BytesIO(data))
                        text = "\n".join([para.text for para in doc.paragraphs])
                    else:
                    # treat as text file or unknown binary -> decode best-effort
                        text = data.decode("utf-8", errors="ignore")
                except Exception as e:
                    st.warning(f"Could not parse {f.name}: {e}")
                    try:
                        text = data.decode("utf-8", errors="ignore")
                    except Exception:
                        text = ""
                if text.strip():
                    docs.append(text)
            if docs:
                try:
                    chunks = chunk_documents(docs)
                    build_index(chunks)
                    st.success("✅ Documents processed and indexed.")
                except Exception as e:
                    st.error(f"Error indexing documents: {e}")
            else:
                st.warning("No parsable text found in the uploaded files.")

    # Select response mode
    mode = st.radio("Response Mode:", ["Concise", "Detailed", "Human-like"], horizontal=True)

    # Display existing chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # RAG / Web search context
                retrieved = query_index(prompt)
                if retrieved:
                    context = "\n\n".join(retrieved)
                else:
                    search_results = web_search(prompt)
                    context = "\n\n".join(search_results) if search_results else ""

                # Choose system prompt
                if mode == "Concise":
                    system_prompt = CONCISE_PROMPT + "\n\nContext:\n" + context
                elif mode == "Detailed":
                    system_prompt = DETAILED_PROMPT + "\n\nContext:\n" + context
                else:  # Human-like
                    system_prompt = HUMAN_PROMPT + "\n\nContext:\n" + context

                # Get response
                response = get_chat_response(chat_model, st.session_state.messages, system_prompt)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

# --------------------------------------------------
# Instructions page
# --------------------------------------------------
def instructions_page():
    st.title("The Chatbot Blueprint")
    st.markdown("Follow the instructions in README.md or Streamlit secrets.")

# --------------------------------------------------
# Main entry
# --------------------------------------------------
def main():
    st.set_page_config(page_title="AI Chatbot", layout="wide")
    with st.sidebar:
        page = st.radio("Go to:", ["Chat", "Instructions"])
        if page == "Chat":
            st.button("Clear Chat History", on_click=lambda: st.session_state.pop("messages", None))

    if page == "Instructions":
        instructions_page()
    else:
        chat_page()

if __name__ == "__main__":
    main()

# import streamlit as st
# import os
# import sys
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# from models.llm import get_chatgroq_model


# def get_chat_response(chat_model, messages, system_prompt):
#     """Get response from the chat model"""
#     try:
#         # Prepare messages for the model
#         formatted_messages = [SystemMessage(content=system_prompt)]
        
#         # Add conversation history
#         for msg in messages:
#             if msg["role"] == "user":
#                 formatted_messages.append(HumanMessage(content=msg["content"]))
#             else:
#                 formatted_messages.append(AIMessage(content=msg["content"]))
        
#         # Get response from model
#         response = chat_model.invoke(formatted_messages)
#         return response.content
    
#     except Exception as e:
#         return f"Error getting response: {str(e)}"

# def instructions_page():
#     """Instructions and setup page"""
#     st.title("The Chatbot Blueprint")
#     st.markdown("Welcome! Follow these instructions to set up and use the chatbot.")
    
#     st.markdown("""
#     ## 🔧 Installation
                
    
#     First, install the required dependencies: (Add Additional Libraries base don your needs)
    
#     ```bash
#     pip install -r requirements.txt
#     ```
    
#     ## API Key Setup
    
#     You'll need API keys from your chosen provider. Get them from:
    
#     ### OpenAI
#     - Visit [OpenAI Platform](https://platform.openai.com/api-keys)
#     - Create a new API key
#     - Set the variables in config
    
#     ### Groq
#     - Visit [Groq Console](https://console.groq.com/keys)
#     - Create a new API key
#     - Set the variables in config
    
#     ### Google Gemini
#     - Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
#     - Create a new API key
#     - Set the variables in config
    
#     ##  Available Models
    
#     ### OpenAI Models
#     Check [OpenAI Models Documentation](https://platform.openai.com/docs/models) for the latest available models.
#     Popular models include:
#     - `gpt-4o` - Latest GPT-4 Omni model
#     - `gpt-4o-mini` - Faster, cost-effective version
#     - `gpt-3.5-turbo` - Fast and affordable
    
#     ### Groq Models
#     Check [Groq Models Documentation](https://console.groq.com/docs/models) for available models.
#     Popular models include:
#     - `llama-3.1-70b-versatile` - Large, powerful model
#     - `llama-3.1-8b-instant` - Fast, smaller model
#     - `mixtral-8x7b-32768` - Good balance of speed and capability
    
#     ### Google Gemini Models
#     Check [Gemini Models Documentation](https://ai.google.dev/gemini-api/docs/models/gemini) for available models.
#     Popular models include:
#     - `gemini-1.5-pro` - Most capable model
#     - `gemini-1.5-flash` - Fast and efficient
#     - `gemini-pro` - Standard model
    
#     ## How to Use
    
#     1. **Go to the Chat page** (use the navigation in the sidebar)
#     2. **Start chatting** once everything is configured!
    
#     ## Tips
    
#     - **System Prompts**: Customize the AI's personality and behavior
#     - **Model Selection**: Different models have different capabilities and costs
#     - **API Keys**: Can be entered in the app or set as environment variables
#     - **Chat History**: Persists during your session but resets when you refresh
    
#     ## Troubleshooting
    
#     - **API Key Issues**: Make sure your API key is valid and has sufficient credits
#     - **Model Not Found**: Check the provider's documentation for correct model names
#     - **Connection Errors**: Verify your internet connection and API service status
    
#     ---
    
#     Ready to start chatting? Navigate to the **Chat** page using the sidebar! 
#     """)

# def chat_page():
#     """Main chat interface page"""
#     st.title("AI ChatBot")
    
#     # Get configuration from environment variables or session state
#     # Default system prompt
#     system_prompt = ""
    
    
#     # Determine which provider to use based on available API keys
#     chat_model = get_chatgroq_model()
    
#     # Initialize chat history
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
    
#     # Display chat messages
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
    
#     # Chat input
#     # if chat_model:
#     if prompt := st.chat_input("Type your message here..."):
#         # Add user message to chat history
#         st.session_state.messages.append({"role": "user", "content": prompt})
        
#         # Display user message
#         with st.chat_message("user"):
#             st.markdown(prompt)
        
#         # Generate and display bot response
#         with st.chat_message("assistant"):
#             with st.spinner("Getting response..."):
#                 response = get_chat_response(chat_model, st.session_state.messages, system_prompt)
#                 st.markdown(response)
        
#         # Add bot response to chat history
#         st.session_state.messages.append({"role": "assistant", "content": response})
#     else:
#         st.info(" No API keys found in environment variables. Please check the Instructions page to set up your API keys.")

# def main():
#     st.set_page_config(
#         page_title="LangChain Multi-Provider ChatBot",
#         page_icon="",
#         layout="wide",
#         initial_sidebar_state="expanded"
#     )
    
#     # Navigation
#     with st.sidebar:
#         st.title("Navigation")
#         page = st.radio(
#             "Go to:",
#             ["Chat", "Instructions"],
#             index=0
#         )
        
#         # Add clear chat button in sidebar for chat page
#         if page == "Chat":
#             st.divider()
#             if st.button(" Clear Chat History", use_container_width=True):
#                 st.session_state.messages = []
#                 st.rerun()
    
#     # Route to appropriate page
#     if page == "Instructions":
#         instructions_page()
#     if page == "Chat":
#         chat_page()

# if __name__ == "__main__":
#     main()
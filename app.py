import streamlit as st
import os, sys, io

# ensure project root imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from config import config  # loads .env
from llm import get_chatgroq_model
from utils.rag_utils import chunk_documents, build_index, query_index
from utils.web_search import web_search
from utils.prompt_templates import CONCISE_PROMPT, DETAILED_PROMPT

# doc parsing
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None
try:
    import docx
except Exception:
    docx = None


def get_chat_response(chat_model, messages, system_prompt):
    """Return a response string. If model is not initialized, return a helpful message."""
    if chat_model is None:
        return "Model not initialized. Please set GROQ_API_KEY in .env or Streamlit secrets."

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


def instructions_page():
    st.title("📘 The Chatbot Blueprint")
    st.markdown(
        """
        - Upload documents (`txt`, `pdf`, `docx`) to build a knowledge base.  
        - Ask questions in natural language.  
        - If the answer isn’t in your docs → it will search the web using SerpAPI.  
        - Choose between **Concise** or **Detailed** answer modes.
        """
    )


def chat_page():
    st.title("🤖 AI ChatBot with RAG + Web Search")

    # Initialize model (may be None if API key missing)
    chat_model = None
    try:
        chat_model = get_chatgroq_model()
    except Exception as e:
        st.warning(f"Model init warning: {e}")

    # session messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Upload & index
    with st.expander("📂 Upload and Index Documents"):
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
                    data = f.read()
                    if fname.endswith(".pdf") and PdfReader is not None:
                        reader = PdfReader(io.BytesIO(data))
                        text = ""
                        for p in reader.pages:
                            t = p.extract_text()
                            if t:
                                text += t + "\n"
                    elif fname.endswith(".docx") and docx is not None:
                        document = docx.Document(io.BytesIO(data))
                        text = "\n".join([para.text for para in document.paragraphs])
                    else:
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

    # Mode selection
    mode = st.radio("Response Mode:", ["Concise", "Detailed"], horizontal=True)

    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input and response
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retrieved = query_index(prompt)
                if retrieved:
                    context = "\n\n".join(retrieved)
                else:
                    search_results = web_search(prompt)
                    context = "\n\n".join(search_results) if search_results else ""

                system_prompt = (
                    (CONCISE_PROMPT if mode == "Concise" else DETAILED_PROMPT)
                    + "\n\nContext:\n" + context
                )
                response = get_chat_response(chat_model, st.session_state.messages, system_prompt)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


def main():
    st.set_page_config(page_title="LangChain Multi-Provider ChatBot", layout="wide")
    with st.sidebar:
        page = st.radio("Go to:", ["Chat", "Instructions"])
        if page == "Chat":
            st.button("🗑️ Clear Chat History", on_click=lambda: st.session_state.pop("messages", None))

    if page == "Instructions":
        instructions_page()
    else:
        chat_page()


if __name__ == "__main__":
    main()

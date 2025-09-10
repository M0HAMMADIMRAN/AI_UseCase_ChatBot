AI USECASE CHATBOT

An AI-powered chatbot built for the NeoStats AI Engineer Internship case study.
It integrates Groq LLM, FAISS-based RAG (Retrieval-Augmented Generation), and SerpAPI web search to provide concise and detailed answers from both documents and the web.

Features:

Upload your own documents (PDF, TXT, DOCX) and query them.

Retrieval-Augmented Generation (RAG): uses FAISS + Sentence Transformers.

LLM Integration: Groq llama-3.1-8b-instant.

Web Search Fallback: queries SerpAPI if the answer isn’t in uploaded docs.

Interactive Streamlit UI.


Modular code (utils/, config/, llm.py).

📂 Project Structure:

AI_UseCase/
 ├── app.py                 # Streamlit app entry
 
 ├── llm.py                 # Groq model initialization
 
 ├── requirements.txt       # Dependencies
 
 ├── README.md              # Documentation
 
 ├── .gitignore             # Git ignored files
 
 ├── config/
 
   │    └── config.py         # Env variable loader
 
 ├── utils/
 
   │    ├── rag_utils.py      # RAG (chunking, FAISS index, query)
   
   │    ├── web_search.py     # SerpAPI web search
   
   │    └── prompt_templates.py
 
 ├── models/                # Saved FAISS index, embeddings
 
 └── .env (local only)      # API keys (ignored in GitHub)


⚙️ Setup (Local Development):

1️⃣ Clone Repo

  git clone https://github.com/<your-username>/AI_UseCase_Chatbot.git
  cd AI_UseCase_Chatbot

2️⃣ Create Virtual Environment

  python -m venv venv
  venv\Scripts\activate   # on Windows
  source venv/bin/activate # on Mac/Linux

3️⃣ Install Dependencies

  pip install -r requirements.txt

4️⃣ Setup Environment Variables

Create a .env file in the project root:

  GROQ_API_KEY=gsk_your_real_groq_key_here
  
  GROQ_MODEL=llama-3.1-8b-instant
  
  SERPAPI_KEY=your_real_serpapi_key_here


⚠️ Do NOT commit .env to GitHub.

5️⃣ Run App Locally

  streamlit run app.py



🌐 Deployment (Streamlit Cloud):

1️⃣ Push to GitHub

  Make sure .gitignore excludes .env and venv/.

2️⃣ Deploy

Go to Streamlit Cloud
.

Select your repo & app.py as entry point.

Dependencies auto-install from requirements.txt.

3️⃣ Add Secrets

In Streamlit Cloud → Settings → Secrets, paste:

  GROQ_API_KEY="gsk_your_real_groq_key_here"
  
  GROQ_MODEL="llama-3.1-8b-instant"
  
  SERPAPI_KEY="your_real_serpapi_key_here"

🎯 Usage:

Upload PDF/TXT/DOCX documents.

Type a question in the chatbox.

Get AI answers powered by Groq + RAG.

If docs don’t contain the answer → SerpAPI web search is used.


Screenshots:

<img width="1900" height="880" alt="image" src="https://github.com/user-attachments/assets/276992e1-6224-4b88-ab26-7c16e28e76e5" />



🛡️ Development Guidelines Followed:

✅ API keys secured in .env / secrets.toml.

✅ Reusable modular functions in utils/.

✅ Error handling with try/except.

✅ No API keys committed to GitHub.

✅ Debugged and tested locally.


📌 Future Improvements

Support multiple Groq models.

Add summarization mode.

Show citations for web answers.

Add authentication for multi-user deployment.

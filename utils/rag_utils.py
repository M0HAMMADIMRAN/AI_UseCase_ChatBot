import os, pickle
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

try:
    import faiss
except Exception:
    faiss = None

INDEX_FILE = "vector_index.faiss"
META_FILE = "docstore.pkl"
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_documents(docs, chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs_objs = [Document(page_content=d) for d in docs]
    return splitter.split_documents(docs_objs)

def build_index(chunks):
    if faiss is None:
        raise RuntimeError("faiss is not available. Install faiss-cpu.")
    vectors = [embedder.encode(ch.page_content) for ch in chunks]
    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype("float32"))
    with open(META_FILE, "wb") as f:
        pickle.dump(chunks, f)
    faiss.write_index(index, INDEX_FILE)

def query_index(query, top_k=5):
    if not os.path.exists(INDEX_FILE) or faiss is None:
        return []
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        chunks = pickle.load(f)
    qvec = embedder.encode([query]).astype("float32")
    D, I = index.search(qvec, top_k)
    results = []
    for idx in I[0]:
        # safety: some indices might be -1
        if idx >= 0 and idx < len(chunks):
            results.append(chunks[idx].page_content)
    return results

# utils/rag_utils.py
import os
import pickle
import logging
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False

from sentence_transformers import SentenceTransformer

INDEX_FILE = "vector_index.faiss"
META_FILE = "docstore.pkl"
VECTORS_FILE = "vectors.npy"

_model = None
def load_embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

class TextChunk:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

def chunk_text_simple(text, chunk_size=800, chunk_overlap=100):
    if not text:
        return []
    text = text.replace("\r\n", "\n")
    length = len(text)
    chunks = []
    start = 0
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - chunk_overlap
        if start < 0:
            start = 0
        if start >= length:
            break
    return chunks

def chunk_documents(docs, chunk_size=800, chunk_overlap=100):
    chunks = []
    for i, doc in enumerate(docs):
        try:
            text = doc if isinstance(doc, str) else str(doc)
            subchunks = chunk_text_simple(text, chunk_size, chunk_overlap)
            for j, ch in enumerate(subchunks):
                chunks.append(TextChunk(ch, {"doc_index": i, "chunk_id": j}))
        except Exception as e:
            logging.exception("Error chunking document %s: %s", i, e)
    return chunks

def build_index(chunks):
    if not chunks:
        logging.warning("build_index called with empty chunks list.")
        return
    try:
        model = load_embedder()
        texts = [c.page_content for c in chunks]
        vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        vectors = np.array(vectors).astype("float32")

        with open(META_FILE, "wb") as f:
            pickle.dump(chunks, f)

        if FAISS_AVAILABLE:
            dim = vectors.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(vectors)
            faiss.write_index(index, INDEX_FILE)
        else:
            np.save(VECTORS_FILE, vectors)

    except Exception as e:
        logging.exception("Error in build_index: %s", e)
        raise

def query_index(query, top_k=5):
    try:
        model = load_embedder()
        qvec = model.encode([query], convert_to_numpy=True).astype("float32")

        if FAISS_AVAILABLE and os.path.exists(INDEX_FILE):
            index = faiss.read_index(INDEX_FILE)
            D, I = index.search(qvec, top_k)
            with open(META_FILE, "rb") as f:
                chunks = pickle.load(f)
            return [chunks[idx].page_content for idx in I[0] if 0 <= idx < len(chunks)]

        if os.path.exists(VECTORS_FILE) and os.path.exists(META_FILE):
            vectors = np.load(VECTORS_FILE)
            with open(META_FILE, "rb") as f:
                chunks = pickle.load(f)
            dists = np.linalg.norm(vectors - qvec, axis=1)
            idxs = np.argsort(dists)[:top_k]
            return [chunks[i].page_content for i in idxs if 0 <= i < len(chunks)]

        return []
    except Exception as e:
        logging.exception("Error querying index: %s", e)
        return []

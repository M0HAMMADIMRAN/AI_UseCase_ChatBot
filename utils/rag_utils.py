# utils/rag_utils.py
import os
import pickle
import logging
import numpy as np

# Try to import faiss (faiss-cpu). If not available we fallback to numpy
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False

from sentence_transformers import SentenceTransformer

# Files used for persistence
INDEX_FILE = "vector_index.faiss"
META_FILE = "docstore.pkl"
VECTORS_FILE = "vectors.npy"

# lazy-loaded embedder
_model = None
def load_embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

class TextChunk:
    """Simple chunk object with page_content and metadata"""
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

def chunk_text_simple(text, chunk_size=800, chunk_overlap=100):
    """Simple character-based chunking with overlap."""
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
    """
    docs: list of strings
    returns: list of TextChunk objects
    """
    chunks = []
    for i, doc in enumerate(docs):
        try:
            text = doc if isinstance(doc, str) else str(doc)
            subchunks = chunk_text_simple(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for j, ch in enumerate(subchunks):
                chunks.append(TextChunk(ch, metadata={"doc_index": i, "chunk_id": j}))
        except Exception as e:
            logging.exception("Error chunking document %s: %s", i, e)
    return chunks

def build_index(chunks):
    """
    Build or overwrite a vector index from a list of TextChunk objects.
    Saves metadata to META_FILE and vectors to FAISS index (or numpy fallback).
    """
    if not chunks:
        logging.warning("build_index called with empty chunks list.")
        return

    try:
        model = load_embedder()
        texts = [c.page_content for c in chunks]

        # Encode all texts (returns list or numpy array)
        vectors = model.encode(texts, show_progress_bar=True)
        vectors = np.array(vectors, dtype="float32")

        # Save metadata (chunks)
        with open(META_FILE, "wb") as f:
            pickle.dump(chunks, f)
        logging.info("Saved chunk metadata to %s", META_FILE)

        if FAISS_AVAILABLE:
            dim = vectors.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(vectors)  # vectors must be float32 and shape (N, dim)
            faiss.write_index(index, INDEX_FILE)
            logging.info("Wrote FAISS index to %s", INDEX_FILE)
        else:
            # fallback: save vectors as numpy array
            np.save(VECTORS_FILE, vectors)
            logging.info("faiss not available; saved vectors to %s", VECTORS_FILE)

    except Exception as e:
        logging.exception("Error in build_index: %s", e)
        raise

def query_index(query, top_k=5):
    """
    Query the index with a text query. Returns list of chunk strings (page_content).
    """
    try:
        model = load_embedder()
        qvec = model.encode([query])
        qvec = np.array(qvec, dtype="float32")
        if qvec.ndim == 1:
            qvec = qvec.reshape(1, -1)

        # FAISS path
        if FAISS_AVAILABLE and os.path.exists(INDEX_FILE):
            index = faiss.read_index(INDEX_FILE)
            D, I = index.search(qvec, top_k)
            with open(META_FILE, "rb") as f:
                chunks = pickle.load(f)
            results = []
            for idx in I[0]:
                if idx >= 0 and idx < len(chunks):
                    results.append(chunks[idx].page_content)
            return results

        # Numpy fallback
        if os.path.exists(VECTORS_FILE) and os.path.exists(META_FILE):
            vectors = np.load(VECTORS_FILE)
            with open(META_FILE, "rb") as f:
                chunks = pickle.load(f)
            dists = np.linalg.norm(vectors - qvec, axis=1)
            idxs = np.argsort(dists)[:top_k]
            results = []
            for i in idxs:
                if 0 <= i < len(chunks):
                    results.append(chunks[i].page_content)
            return results

        # Nothing available
        return []

    except Exception as e:
        logging.exception("Error querying index: %s", e)
        return []

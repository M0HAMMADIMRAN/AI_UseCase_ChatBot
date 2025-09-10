from sentence_transformers import SentenceTransformer

# simple wrapper - loads the model once
_model = None

def load_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def get_embedding(text):
    model = load_embedding_model()
    return model.encode(text)

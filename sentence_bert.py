from sentence_transformers import SentenceTransformer

def generate_sentence_embeddings (docs):
    docs_strings = [" ".join(doc) for doc in docs]
    sentence_embeddings = SentenceTransformer.encode(docs_strings, show_progress_bar=True)

    return sentence_embeddings


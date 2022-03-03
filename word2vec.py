import numpy as np

def computeWord2Vec (word2vec, tokens):
    return [np.linalg.norm(word2vec[word]) for word in tokens]

def computeWord2VecScores (word2vec, embeddings):
    embeddings_average = embeddings.mean(axis=0)
    normalized_average_embedding = (embeddings_average - np.min(embeddings_average))/np.ptp(embeddings_average)
    
    return [np.arccos(np.clip(np.dot(embeddings, normalized_average_embedding), -1, 1)) for embedding in embeddings]
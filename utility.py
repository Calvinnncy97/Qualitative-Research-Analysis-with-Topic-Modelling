import heapq
import pandas as pd
from gensim.models import CoherenceModel
from nltk.cluster import KMeansClusterer
from nltk.cluster.util import cosine_distance


def wordFreqGenerator (tokens):
    wordfreq = {}
    for sentence in tokens:
        for token in sentence:
            if token not in wordfreq.keys():
                wordfreq[token] = 1
            else:
                wordfreq[token] += 1

    #get the words with the highest freq
    most_freq = heapq.nlargest(10, wordfreq, key=wordfreq.get)
    print (f"Top 20 words: {most_freq}")

    return wordfreq

def TFIDFGenerator (docs):
    #TF-IDF
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfIdfVectorizer = TfidfVectorizer(use_idf=True)
    tfIdf = tfIdfVectorizer.fit_transform([" ". join(i) for i in docs])
    df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
    df = df.sort_values('TF-IDF', ascending=False)

    print ("Top words with the highest TFIDF scores")
    print (df.head(25))
    print (df.describe())

    return df

# supporting function
def computeCoherenceValues(tokens, id2word, model=None, topics=None, process_num=-1):
    process_param = process_num
        
    if model != None:
        cv_coherence = CoherenceModel(model=model, texts=tokens, dictionary=id2word, coherence='c_v', processes=process_param)
        cuci_coherence = CoherenceModel(model=model, texts=tokens, dictionary=id2word, coherence='c_uci', processes=process_param)
    else:
        cv_coherence = CoherenceModel(topics=topics.values(), texts=tokens, dictionary=id2word, coherence='c_v', processes=process_param) 
        cuci_coherence = CoherenceModel(topics=topics.values(), texts=tokens, dictionary=id2word, coherence='c_uci', processes=process_param) 

    return cv_coherence.get_coherence(), cuci_coherence.get_coherence()

def cluster (tokens, embeddings, num_of_topics):
    kclusterer = KMeansClusterer(num_of_topics, distance=cosine_distance, repeats=25, avoid_empty_clusters=True)
    assigned_clusters = kclusterer.cluster(embeddings, assign_clusters=True)

    clusters = {}
    for i, word in enumerate(tokens):  
        if assigned_clusters[i] in clusters.keys():
            clusters[assigned_clusters[i]].append(word)
        else:
            clusters[assigned_clusters[i]] = [word]

    # import collections
    # ordered_clusters = collections.OrderedDict(sorted(clusters.items()))

    return clusters
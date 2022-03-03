# TODO: combine stopwords list

import os
import multiprocessing as mp
from multiprocessing.spawn import freeze_support, import_main_path
from concurrent.futures import ProcessPoolExecutor as Pool
from nltk import corpus

from numpy.core.fromnumeric import take
from preprocess import *
from lda import *
from word2vec import *
from bert import *
from utility import *

def executeLDA (input):
    tokens, corpus, dictionary, k, a, b, passes, path, filename = input
    model = computeLDA (corpus, dictionary, k, a, b, passes)
    cv_score, cuci_score = computeCoherenceValues (tokens, dictionary, model=model, process_num=1)
    
    if not os.path.exists(os.path.join(path, filename)):
        os.mkdir(os.path.join(path, filename))
        df_header = pd.DataFrame(columns=["Model", "Corpus", "Alpha", "Beta", "Number of Topics", "Topics - Top 10 Words", "CV Score", "C_UCI Score"])
        df_header.to_csv(f'{path}\\{filename}-Results', header=True)

    df = pd.DataFrame([["LDA", filename, a, b, k, model.print_topics(num_topics=-1, num_words=10), cv_score, cuci_score]])
    print (df)

    df.to_csv(f'{path}\\LDA-Results.csv', mode='a', header=False)

def ldaProcess (corpuses):
    lda_input = [[corpus[0], ldaPreprocess(corpus[0]), corpus[1]] for corpus in corpuses]

    min_topics = 2
    max_topics = 52
    step_size = 2
    topics_range = range(min_topics, max_topics, step_size)

    # Alpha parameter
    alpha = list(np.arange(0.01, 1, 0.3))
    alpha.append('symmetric')
    alpha.append('asymmetric')

    # Beta parameter
    beta = list(np.arange(0.01, 1, 0.3))
    beta.append('symmetric')

    import tqdm
    tasks = [(input[0], input[1][0], input[1][1], k, a, b, 50, "Results", input[2]) for input in lda_input for a in alpha for b in beta for k in topics_range]
    pbar = tqdm.tqdm(total=len(tasks))
    
    with Pool(8) as pool:
        for _ in tqdm.tqdm(pool.map(executeLDA, tasks), total=len(tasks)):
            pbar.update(1)

def executeWord2Vec (input):
    corpus, filtered_corpus, embeddings, id2word, num_of_topics, path, filename = input
    topics = cluster(filtered_corpus, embeddings, num_of_topics)

    print ("Calculating coherence")
    cv_score, cuci_score = computeCoherenceValues(tokens=corpus, id2word=id2word, topics=topics, process_num=1)

    print ("Writing file")
    if not os.path.exists(os.path.join(path, filename)):
        #table columnns: model, corpus name, Number of Topics, Topics - Top 10 Words, CV Score, C_UCI Score
        df_header = pd.DataFrame(columns=["Model", "Corpus", "Number of Topics", "Topics - Top 10 Words", "CV Score", "C_UCI Score"])
        df_header.to_csv(f'{path}\\{filename}-Results', header=True)

    df = pd.DataFrame([["Word2Vec", filename, num_of_topics, [topic[:10] for topic in topics.values()], cv_score, cuci_score]])
    print (df)

    df.to_csv(f'{path}\\Word2Vec-Results.csv', mode='a', header=False)

def word2VecProcess (corpuses):
    #load Word2Vec model
    from gensim.models import KeyedVectors
    print ("Loading Word2Vec Model")
    word2vec = KeyedVectors.load("Google-News-Vectors")

    #initialize parameters
    min_topics = 2
    max_topics = 52
    step_size = 2
    topics_range = range(min_topics, max_topics, step_size)
    
    tasks = []

    for corpus in corpuses:
        _, id2word = ldaPreprocess(corpus[0])
        filename = corpus[1]

        flattened_corpus = []
        for token in corpus[0]:
            flattened_corpus.extend(token)

        filtered_corpus = [word for word in flattened_corpus if word2vec.has_index_for(word)] 
        
        for i in topics_range:
            tasks.append((corpus[0], filtered_corpus, word2vec[filtered_corpus], id2word, i, "Results", filename))

    import tqdm
    pbar = tqdm.tqdm(total=len(tasks))
    
    with Pool(8) as pool:
        for _ in tqdm.tqdm(pool.map(executeWord2Vec, tasks), total=len(tasks)):
            pbar.update(1)

def bertProcess (corpuses, tokenizer, model):
    #initialize parameters
    min_topics = 2
    max_topics = 52
    step_size = 2
    topics_range = range(min_topics, max_topics, step_size)
    
    tasks = []
    index = 0

    for corpus in corpuses:
        _, id2word = ldaPreprocess(corpus[0])
        filename = corpus[1]

        flattened_corpus = []
        for token in corpus[0]:
            flattened_corpus.extend(token)

        bert_sequences = splitBertInputSequences(corpus[0], tokenizer)
        split_embeddings = computeBertEmbeddings(bert_sequences, tokenizer, model)
        combined_embeddings = combineBertOutput(split_embeddings)

        for i in topics_range:
            tasks.append((corpus[0], deepcopy(flattened_corpus), deepcopy(combined_embeddings), id2word, i, "Results", filename))
        
    import tqdm
    pbar = tqdm.tqdm(total=len(tasks))
    
    with Pool(8) as pool:
        for _ in tqdm.tqdm(pool.map(executeBert, tasks), total=len(tasks)):
            pbar.update(1)

    pbar.close()

    
def executeBert (input):
    corpus, flattened_corpus, embeddings, id2word, num_of_topics, path, filename = input
    print ("Clustering")
    topics = cluster(flattened_corpus, embeddings, num_of_topics)

    print ("Calculating coherence")
    cv_score, cuci_score = computeCoherenceValues(tokens=corpus, id2word=id2word, topics=topics, process_num=1)

    print ("Writing file")
    if not os.path.exists(os.path.join(path, filename)):
        #table columnns: model, corpus name, Number of Topics, Topics - Top 10 Words, CV Score, C_UCI Score
        df_header = pd.DataFrame(columns=["Model", "Corpus", "Number of Topics", "Top Words", "CV Score", "C_UCI Score"])
        df_header.to_csv(f'{path}\\Bert-Results', header=True)

    df = pd.DataFrame([["BERT - Transformer", filename, num_of_topics, [topic[:10] for topic in topics.values()], cv_score, cuci_score]])

    df.to_csv(f'{path}\\Bert-Results.csv', mode='a', header=False)

def executeLFLDA (input):
    corpus, k, a, b, l, filename = input

    filename = corpus.replace("LFTM-Master\\", "")

    import subprocess
    subprocess.run(f"java -jar LFTM-master\\jar\\LFTM.jar -model LFLDA -corpus {corpus} -vectors LFTM-Master\\Google-News-Vectors.txt -ntopics {k} -alpha {a} -beta {b} -lambda {l} -initers 500 -niters 50 -name {filename}-{a}-{b}-{l}-{k}", capture_output=True, start_new_session=True)

def lfldaProcess (corpuses):
    #initialize parameters
    min_topics = 2
    max_topics = 52
    step_size = 2
    topics_range = range(min_topics, max_topics, step_size)

     # Alpha parameter
    alpha = list(np.arange(0.01, 1, 0.3))
    alpha.append('symmetric')
    alpha.append('asymmetric')

    # Beta parameter
    beta = list(np.arange(0.01, 1, 0.3))
    beta.append('symmetric')

    #Lambda parameter
    lamda = list(np.arange(0.01, 1, 0.3))

    import tqdm
    tasks = [(corpus[0], k, a, b, l,  corpus[1]) for corpus in corpuses for a in alpha for b in beta for l in lamda for k in topics_range]
    pbar = tqdm.tqdm(total=len(tasks))
    
    with Pool(8) as pool:
        for _ in tqdm.tqdm(pool.map(executeLFLDA, tasks), total=len(tasks)):
            pbar.update(1)

def lfldaComputeCoherence ():
        pass

if __name__ == "__main__":
    print ("Starting Processes")
    folder_path = "C:\\Users\\ncy_k\\Desktop\\EMBRACE Redacted Transcripts\\Compiled Data\\"
    files = ["Compiled Data Challenges - SAC Staff.docx", "Compiled Data Challenges.docx", "Compiled Data Challenges - Volunteer.docx"]
    file_paths = [folder_path+file for file in files]

    # process docs, filename
    input_corpus = [[preprocess(file), files[index].replace(".docx", "")] for index, file in enumerate(file_paths)]

    from transformers import BertTokenizer, BertModel
    model = BertModel.from_pretrained("bert-based-uncased-pytorch", output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased-tokenizer')
    model.eval()

    bertWorker = mp.Process(target=bertProcess, args=(input_corpus, tokenizer, model,))
    bertWorker.start()
    bertWorker.join()

    # word2vecWorker = mp.Process(target=word2VecProcess, args=(input_corpus,))

    # word2vecWorker.start()
    # word2vecWorker.join()
    
    # ldaWorker = mp.Process(target=ldaProcess, args=(input_corpus,))
    # ldaWorker.start()
    # ldaWorker.join()

    # lflda_files = ["LFTM-Master\\LFDLA-Challenges-SAC-Staff.txt", "LFTM-Master\\LFDLA-Challenges-Full.txt", "LFTM-Master\\LFDLA-Challenges-Volunteers.txt"]
    # lflda_corpuses = [[file, file.replace(".txt",  "")] for file in lflda_files]
    # lfldaWorker = mp.Process(target=lfldaProcess, args=(lflda_corpuses,))
    # lfldaWorker.start()
    # lfldaWorker.join()
import glob
import docx

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')

stopwords_list = ["ok", 'u', "1", "one", "yea", "2", "u", "oh", "also", "yeah", "00", "okay", "", "hm", "haha", "5l", "mm", 
                    "k", "b", "le", "u", "lo", "p", "unintelligible", "actually", "subtitle"]

def preprocess (path):
    print (f"Path: {path}")
    files = glob.glob(path)

    para_list = []

    for file in files:
        paras = docx.Document(file).paragraphs
        if paras != None:
            for para in paras:
                para = para.text
                colon_index = para.find(":")
                speaker = para[0:colon_index]
                if colon_index != -1 and (speaker == "LD" or speaker == "LD2" or speaker == "LD1"):
                    continue
                para = para[colon_index+1:].strip().lower()
                for word in para:
                    if word.isdigit():
                        para = para.replace(word,"")
                if para.replace(":","").isdigit():
                    continue
                if len(para)>0:
                    para_list.append(para)
        
    #tokenize and remove stop words by sentence
    all_sw = stopwords.words('English')
    all_sw.extend(stopwords_list)
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    wordnet_lemmatizer = WordNetLemmatizer()

    tokenized_and_cleaned_doc = []
    print (f"Length: {len(para_list)}")
    for i in para_list:
        tokens = tokenizer.tokenize(i)
        tokens_without_sw = []
        tokens_without_sw = [wordnet_lemmatizer.lemmatize(word.strip()) for word in tokens if len(word) > 3 and not (word.strip() in all_sw)]
        if len(tokens_without_sw) > 0: 
            tokenized_and_cleaned_doc.append(tokens_without_sw)

    print(f"{len(tokenized_and_cleaned_doc)} documents generated.")
    
    return tokenized_and_cleaned_doc
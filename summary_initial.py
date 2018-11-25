import json

# json_filename = 'preprocessed/big_relevant.json'
# json_file = open(json_filename)
# json_str = json_file.read()
# json_data = json.loads(json_str)
# #text = json_data[1]['Sentences']
# text = ''
# for i in range(0,int(len(json_data))):
#     text+= json_data[i]['Sentences']
# # print(text)
# #for record in json_data:
# #    text += record['Sentences']

# # print(text)

import os
import re
# import nltk
# nltk.download('brown')
# from nltk.corpus import brown
import pickle
import nltk
import numpy as np
import datetime
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
stop = stopwords.words('english')
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Noun Part of Speech Tags used by NLTK
NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def clean_document(document):
    """Cleans document by removing unnecessary punctuation. It also removes
    any extra periods and merges acronyms to prevent the tokenizer from
    splitting a false sentence

    """
    # Remove all characters outside of Alpha Numeric
    # and some punctuation
    document = re.sub('[^A-Za-z .-]+', ' ', document)
    document = document.replace('-', '')
    document = document.replace('...', '')
    document = document.replace('Mr.', 'Mr').replace('Mrs.', 'Mrs')

    # Remove Ancronymns M.I.T. -> MIT
    # to help with sentence tokenizing
    document = merge_acronyms(document)

    # Remove extra whitespace
    document = ' '.join(document.split())
    return document

def remove_stop_words(document):
    """Returns document without stop words"""
    document = ' '.join([i for i in document.split() if i not in stop])
    return document

def similarity_score(t, s):
    """Returns a similarity score for a given sentence.

    similarity score = the total number of tokens in a sentence that exits
                        within the title / total words in title

    """

    t = remove_stop_words(t.lower())
    s = remove_stop_words(s.lower())
    t_tokens = t.split()
    s_tokens = s.split()
    similar = [w for w in s_tokens if w in t_tokens]
    score = (len(similar) * 0.1 ) / len(t_tokens)
    return score

def merge_acronyms(s):
    """Merges all acronyms in a given sentence. For example M.I.T -> MIT"""
    r = re.compile(r'(?:(?<=\.|\s)[A-Z]\.)+')
    acronyms = r.findall(s)
    for a in acronyms:
        s = s.replace(a, a.replace('.',''))
    return s

def rank_sentences(doc, doc_matrix, feature_names, top_n=25):
    """Returns top_n sentences. Theses sentences are then used as summary
    of document.

    input
    ------------
    doc : a document as type str
    doc_matrix : a dense tf-idf matrix calculated with Scikits TfidfTransformer
    feature_names : a list of all features, the index is used to look up
                    tf-idf scores in the doc_matrix
    top_n : number of sentences to return

    """
    sents = nltk.sent_tokenize(doc)
    sentences = [nltk.word_tokenize(sent) for sent in sents]
    sentences = [[w for w in sent if nltk.pos_tag([w])[0][1] in NOUNS]
                  for sent in sentences]
    tfidf_sent = [[doc_matrix[feature_names.index(w.lower())]
                   for w in sent if w.lower() in feature_names]
                 for sent in sentences]

    # Calculate Sentence Values
    doc_val = sum(doc_matrix)
    sent_values = [sum(sent) / doc_val for sent in tfidf_sent]

    # Apply Similariy Score Weightings
    similarity_scores = [similarity_score(title, sent) for sent in sents]
    scored_sents = np.array(sent_values) + np.array(similarity_scores)

    ranked_sents = [pair for pair in zip(range(len(sent_values)), sent_values)]
    ranked_sents = sorted(ranked_sents, key=lambda x: x[1] *-1)

    return ranked_sents[:top_n]

if __name__ == '__main__':
    for t in range(5):
        print('Start ranking topic '+str(t))
        createFolder('preprocessed/Topics/Topic'+str(t)+'_Sents')
        IDs_topic = np.loadtxt(open("preprocessed/Topics/Topic"+str(t)+".csv",'rb'),delimiter=",")
        # Load corpus data used to train the TF-IDF Transformer
        # data = brown.words()
        sents = ''
        NoDAPL_topic_file = 'preprocessed/Topics/Topic'+str(t)+'.json'
        with open(NoDAPL_topic_file, 'r') as data_file:
            raw = json.load(data_file)
            for doc in raw:
                sents += ' '
                sents += doc['Sentences']
            data = nltk.word_tokenize(sents)
            print('-----finished corpus tokenization-----')
            # Load the document you wish to summarize
            title = 'American Missouri River Dakota Access Pipeline Fort Yates Standing Rock America Bakkan Sioux Youth Army Corps Engineer North Obama Trump Native DAPL Radio Energy Transfer Gonacon'
            count = 0
            for ele in raw:
                document = ele['Sentences']
                cleaned_document = clean_document(document)
                doc = remove_stop_words(cleaned_document)

                # Merge corpus data and new document data
                data = [' '.join(document) for document in data]
                train_data = set(data + [doc])

                # Fit and Transform the term frequencies into a vector
                count_vect = CountVectorizer()
                count_vect = count_vect.fit(train_data)
                freq_term_matrix = count_vect.transform(train_data)
                feature_names = count_vect.get_feature_names()

                # Fit and Transform the TfidfTransformer
                tfidf = TfidfTransformer(norm="l2")
                tfidf.fit(freq_term_matrix)

                # Get the dense tf-idf matrix for the document
                story_freq_term_matrix = count_vect.transform([doc])
                story_tfidf_matrix = tfidf.transform(story_freq_term_matrix)
                story_dense = story_tfidf_matrix.todense()
                doc_matrix = story_dense.tolist()[0]

                # Get Top Ranking Sentences and join them as a summary
                top_sents = rank_sentences(doc, doc_matrix, feature_names,top_n=1)
                # print(top_sents)
                summary = '.'.join([cleaned_document.split('.')[i]
                                    for i in [pair[0] for pair in top_sents]])
                summary = ' '.join(summary.split())
                summary += '.'
                outname = 'preprocessed/Topics/Topic'+str(t)+'_Sents/'+str(int(IDs_topic[count]))+'.txt'
                with open(outname, "w") as text_file:
                    text_file.write(summary)
                count += 1
                print("Completed {0}%".format(round(count/len(IDs_topic)*100)))


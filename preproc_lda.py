import spacy
# spacy.load('en')
from spacy.lang.en import English
import numpy as np
import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import json
from gensim import corpora
import pickle
import gensim
import pyLDAvis.gensim


parser = English()
def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

if __name__=='__main__':
    en_stop = set(nltk.corpus.stopwords.words('english'))
    text_data = []
    # json_filename = 'data/NoDAPLsmall.json'
    json_filename = 'preprocessed/big_relevant.json'
    json_file = open(json_filename)
    json_str = json_file.read()
    json_data = json.loads(json_str)
    sm_corpus = list()
    for record in json_data:
        sm_corpus.append(record['Sentences'])
    for doc in sm_corpus:
        text_data.append(prepare_text_for_lda(doc))
    # np.save('preprocessed_corpus_small.npy',text_data)
    # print(text_data[1])
    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]
    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    dictionary.save('dictionary.gensim')
    # LDA
    NUM_TOPICS = 10
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
    ldamodel.save('model10.gensim')
    topics = ldamodel.print_topics(num_words=4)
    for topic in topics:
        print(topic)
    # visualization of LDA topics
    dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
    corpus = pickle.load(open('corpus.pkl', 'rb'))
    lda = gensim.models.ldamodel.LdaModel.load('model10.gensim')
    lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
    pyLDAvis.show(lda_display)
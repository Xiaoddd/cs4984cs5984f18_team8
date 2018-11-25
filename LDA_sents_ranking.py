import numpy as np
import json
from gensim import corpora
import pickle
import gensim
import nltk
import spacy
from spacy.lang.en import English
from gensim import corpora
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
# import pyLDAvis.gensim

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
    TOP_N = 5
    en_stop = set(nltk.corpus.stopwords.words('english'))
    dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
    corpus = pickle.load(open('corpus.pkl', 'rb'))
    lda = gensim.models.ldamodel.LdaModel.load('LDAresults/model5.gensim')
    sents_topics = list()
    for t in range(5):
        tmp_sim = list()
        tmp_sents = list()
        NoDAPL_topic_file = 'preprocessed/Topics/Topic'+str(t)+'.json'
        with open(NoDAPL_topic_file, 'r') as data_file:
            raw = json.load(data_file)
            sents = ''
            for doc in raw:
                sents += ' '
                sents += doc['Sentences']
            data = nltk.sent_tokenize(sents)
        for sent in data:
            bow = dictionary.doc2bow(prepare_text_for_lda(sent))
            tmplda = lda[bow]
            tmpOrder = np.zeros(len(tmplda))
            for j in range(len(tmplda)):
                tmpOrder[j] = tmplda[j][1]
            topicIdx = tmplda[np.argmax(tmpOrder)][0]
            if topicIdx == t:
                tmp_sim.append(tmplda[np.argmax(tmpOrder)][1])
                tmp_sents.append(sent)
        tmpSimilarity = np.asarray(tmp_sim)
        sorted_order = tmpSimilarity.argsort()[::-1][:TOP_N]
        paragraph_topic = ''
        for ord in sorted_order:
            paragraph_topic += tmp_sents[int(ord)]
            paragraph_topic += ' '
        sents_topics.append(paragraph_topic)
        print(sents_topics)
        outname = 'preprocessed/Topics/LDA_ranking/Topic_'+str(t)+'_paragraph.txt'
        with open(outname, "w") as text_file:
            text_file.write(paragraph_topic)

    
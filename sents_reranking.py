import pandas as pd
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
from sklearn.preprocessing import normalize
import csv

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

def similarity_scores(lda, filename):
    abst = pd.read_csv(filename)
    abst = abst.values.tolist()
    sents_abst = list()
    sim_abst = list()
    for ele in abst:
        data = nltk.sent_tokenize(ele[0])
        for sent in data:
            bow = dictionary.doc2bow(prepare_text_for_lda(sent))
            tmpldascore = lda[bow]
            for j in range(len(tmpldascore)):
                if tmpldascore[j][0] == t:
                    sim_abst.append(tmpldascore[j][1])
                    sents_abst.append(sent)
    return sents_abst, sim_abst

def frequence_scores(ners, sents):
    freq = list()
    for sent in sents:
        sent = sent.lower()
        words = nltk.word_tokenize(sent)
        tmpfreq = np.zeros(len(ners))
        for i in range(len(ners)):
            matchers = ners[i]
            matching = list()
            for word in words:
                if word == matchers:
                    matching.append(word)
            tmpfreq[i] = len(matching)/len(words)
        freq.append(np.sum(tmpfreq))
    return freq

def remove_replicated_sents(sents):
    out = list()
    for sent in sents:
        if sent not in out:
            out.append(sent)
    return out
if __name__=='__main__':
    TOP_N = 20
    en_stop = set(nltk.corpus.stopwords.words('english'))
    dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
    corpus = pickle.load(open('corpus.pkl', 'rb'))
    lda = gensim.models.ldamodel.LdaModel.load('LDAresults/model5.gensim')
    for t in range(5):
        filename = 'NER/NER'+str(t)+'.txt'
        with open(filename, 'r') as f:
            ners = f.readlines()
            for i in range(len(ners)):
                ners[i] = ners[i].rstrip()
                ners[i] = ners[i].lower()
        # Abstractive Similarity
        filename = 'preprocessed/Topics/LDA_ranking/Abstractive/Abstractve_Topic'+str(t)+'.csv'
        sents_abst, sim_abst = similarity_scores(lda, filename)
        sim_abst = np.array(sim_abst)
        sim_abst = normalize(sim_abst).ravel()
        print('Finished calculating abstractive similarity for Topic'+str(t))
        # Extractive Similarity
        filename = 'preprocessed/Topics/LDA_ranking/Extractive/Extractve_Topic'+str(t)+'.csv'
        sents_ext, sim_ext = similarity_scores(lda, filename)
        sim_ext = np.array(sim_ext)
        sim_ext = normalize(sim_ext).ravel()
        print('Finished calculating extractive similarity for Topic'+str(t)+'!')
        # Abstractive NER frequences
        freq_abst = np.array(frequence_scores(ners, sents_abst))
        freq_abst = normalize(freq_abst).ravel()
        print('Finished calculating abstractive frequency for Topic'+str(t)+'!')
        # Extractive NER frequences
        freq_ext = np.array(frequence_scores(ners, sents_ext))
        freq_ext = normalize(freq_ext).ravel()
        print('Finished calculating extractive frequency for Topic'+str(t)+'!')
        # Reranking score calculation
        weight_abst = 0.7
        score_abst = weight_abst*sim_abst + (1-weight_abst)*freq_abst
        score_ext = weight_abst*sim_ext + (1-weight_abst)*freq_ext
        scores = np.concatenate((score_abst,score_ext),axis=None)
        sents_ae = sents_abst + sents_ext
        print('Finished calculating reranking scores for Topic'+str(t)+'!')
        sorted_order = scores.argsort()[::-1][:TOP_N]
        sents_para = list()
        paragraph = ''
        for ord in sorted_order:
            sents_para.append(sents_ae[ord])
        sents_para = remove_replicated_sents(sents_para)
        print(sents_para)
        for sent in sents_para:
            paragraph+=sent
            paragraph+=' '
        outname = 'preprocessed/Topics/reranking/Topic'+str(t)+'.txt'
        with open(outname, "w") as output:
            writer = csv.writer(output)
            writer.writerow([paragraph])
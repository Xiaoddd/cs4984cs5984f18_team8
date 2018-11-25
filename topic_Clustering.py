import numpy as np
import json
from gensim import corpora
import pickle
import gensim
# import pyLDAvis.gensim

if __name__=='__main__':
    
    dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
    corpus = pickle.load(open('corpus.pkl', 'rb'))
    lda = gensim.models.ldamodel.LdaModel.load('LDAresults/model5.gensim')
    # lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
    # pyLDAvis.show(lda_display)
    IDs = np.loadtxt(open("preprocessed/rel_id_start_from_0.csv",'rb'),delimiter=",")
    clusters = list()
    for i in range(5):
        tmp = list()
        count = 0
        for bow in corpus:
            tmplda = lda[bow]
            tmpOrder = np.zeros(len(tmplda))
            for j in range(len(tmplda)):
                tmpOrder[j] = tmplda[j][1]
            topicIdx = tmplda[np.argmax(tmpOrder)][0]
            if topicIdx == i:
                tmp.append(IDs[count])
            count += 1
        filename_i = 'preprocessed/Topic'+str(i)+'.csv'
        clusters.append(tmp)
        np.savetxt(filename_i,tmp,delimiter=",")
        NoDAPL_file = "data/part-00000-66d9f78f-37f9-4dea-985c-6e2c040632ef-c000.json"
        with open(NoDAPL_file, 'r') as data_file:
            data = json.load(data_file)
            TopicCorpus = [data[int(i)] for i in tmp]
        print(len(TopicCorpus))
        filename_t = 'preprocessed/Topics/Topic'+str(i)+'.json'
        with open(filename_t, 'w') as outfile:
            json.dump(TopicCorpus,outfile)

    np.save('preprocessed/Topics.npy',clusters)
    
###
## Feature Extraction for cs4984/5984 Project
## Team 8 | Oct 2018
###

import json, nltk, string, requests, numpy, itertools
import nltk
import string
import pprint
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from bs4 import BeautifulSoup
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

def filter_synsets(synsets):
    syns = set()
    for syn in synsets:
        for lemma in syn.lemmas():
            syns.add(lemma.name().lower())
    return list(syns)

def get_synsets(keyword):
    '''
    create synsets from NLTK
    '''
    syns = set()
    for syn in wordnet.synsets(keyword):
        for lemma in syn.lemmas():
            syns.add(lemma.name().lower())
    return list(syns)


def count_frequency(keyword, synsets, document):
    '''
    count total frequency for all synsets in a cleaned document
    :param synsets: a list of synsets for a specific word
    :param document: cleaned documents
    :param stopwords: optional stopwords list
    :return: totally frequency for all synsets in the document
    '''
    docLength = len(document)
    # print docLength, len(synsets)
    count = 0
    # we found input document can be empty, may need double-check the document cleaning process
    if docLength > 0:
        # combine keyword and its synsets
        set(synsets).add(keyword)
        for synset in synsets:
            for word in document:
                word = word.lower()
                # filter and compare
                # if word not in stopwords and word not in string.punctuation and len(word) > 2 and word == synset.lower():
                if len(word) > 2 and word == synset.lower():
                    count += 1
        count = float(count) / float(docLength)
        print count
    return count


def count_frequency_keyword_only(keywords, document):
    '''
    count total frequency for all synsets in a cleaned document
    :param synsets: a list of synsets for a specific word
    :param document: cleaned documents
    :param stopwords: optional stopwords list
    :return: totally frequency for all synsets in the document
    '''
    docLength = len(document)
    # print docLength, len(synsets)
    count = 0
    # we found input document can be empty, may need double-check the document cleaning process
    if docLength > 0:
        # combine keyword and its synsets
        for keyword in keywords:
            for word in document:
                word = word.lower()
                # filter and compare
                # if word not in stopwords and word not in string.punctuation and len(word) > 2 and word == synset.lower():
                if len(word) > 2 and word == keyword.lower():
                    count += 1
        count = float(count) / float(docLength)
        print count
    return count


""" CONFIG """
NoDAPL_file = "data\NoDAPL-top50.json"
stopwords_file = "data\stopwords_new.txt"
stopwords_list = ["'s", '...', '-800', 'mr.', 'de', 'een', '-0800', 'www.senate.gov', "''", "``", "\\n", "'s", "'re", "the"]
wikipage_link = 'https://en.wikipedia.org/wiki/NODAPL' # 'https://en.wikipedia.org/wiki/Dakota_Access_Pipeline'

""" LOAD DATA """
test_document = "This movement tribe is not just about a pipeline.  You folk are not fighting for a reroute, or a better process in the white man courts.  We are fighting for our rights as the indigenous peoples of this land; we are fighting for our liberation, and the liberation of Unci Maka, Mother Earth.  We want every last oil and gas pipe removed from her body.  We want healing.  We want clean water.  We want to determine our own future. Each one of us is fighting for our grandchildren, and their grandchildren, and for our relatives who cannot speak or fight back. Imagine if we had stood together on October 27th, the day they pushed us out the Treaty Camp we built in the very path of the Black Snake -- our most powerful position in this entire struggle. What if our own people had not negotiated away our power? What if our people had not opened the roads and then turned to march against us with outstretched arms, in line with the riot police and armored vehicles?  Why pass resolutions calling federal agents to attack our people and evict the camps as the drill digs beneath our sacred water? How powerful could we be if we agree to stand our ground on our treaty land where we have laid thousands of prayers?"
test_keywords = ["access", "army", "bakkan", "campaign", "climate", "companies", "construction", "corp", "dakota",
                 "dapl", "donation", "drill", "duty", "energy", "fossil", "fuel", "indian", "indigenous", "keystone", "klp",
                 "mission", "missouri", "morton", "movement", "native", "nyemah", "obama", "oil", "nodapl", "patriots",
                 "pipeline","police", "protestor", "protest", "reservation", "resistance", "sacred", "sioux", "supplies", "tribal",
                 "tribe", "trump", "veteran", "violence", "volunteer", "water"]

clean = list()
with open(NoDAPL_file, 'r') as json_data:
    input_data = json.load(json_data)
    for sentence in input_data:
        print sentence
        clean.append(sentence['Sentences'][0])
json_data.close()

input = list()
for entry in clean:
    # note: type of "Sentences" value is list in NoDAPL-top50.json
    print len(entry),entry
    input.append(word_tokenize(entry))

""" STOP WORDS """
with open(stopwords_file, 'r') as stop_file:
    stopword_custom = stop_file.read().split("\n")
stop_file.close()
# concanete all the stopwords sources
stop_words = stopwords.words('english') + list(string.punctuation) + stopword_custom + stopwords_list

""" LEMMATIZATION """
lemmatizedList = list()
lemmat = nltk.WordNetLemmatizer()
for i in range(len(input)):
    lemmatizedList.append([lemmat.lemmatize(x, j[0].lower()) if j[0].lower() in ['a', 'n', 'v'] else lemmat.lemmatize(x) for x, j in pos_tag(input[i])])

print len(lemmatizedList), lemmatizedList[0]

cleandata = list()
for data in lemmatizedList:
    cleandata.append([w for w in data if w.lower() not in stop_words])
print len(cleandata), cleandata[2]

""" GRAB WIKIPEDIA CONTENT """
page_response = requests.get(wikipage_link, timeout=5)
# here, we fetch the content from the url, using the requests library
page_content = BeautifulSoup(page_response.content, "html.parser")
# we use the html parser to parse the url content and store it in a variable.
textContent = []
for i in range(1, 7):
    paragraphs = page_content.find_all("p")[i].text
    textContent.append(paragraphs)

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(textContent)

# GET KEYWORDS / FREQUENT WORDS FOR NODAPL
json_data = json.dumps(textContent)
keywords = list()


words = word_tokenize(json_data.lower())  # countin in lower
wordsFiltered = []
for w in words:
    if w not in stopwords:
        wordsFiltered.append(w)

fd = nltk.FreqDist(wordsFiltered)
most_freq = fd.most_common()[0:50]
for words in most_freq:
    keywords.append(words[0])
print(keywords)
pp.pprint(most_freq)
print(len(most_freq))


keywords_synsets = list()

for keyword in keywords:
    keyword_synsets = get_synsets(keyword)
    keywords_synsets.append(keyword_synsets)
print(keywords_synsets)

# use dummy data
##calculate frequency using generated synsets from keywords
# wordFreq = dict()
# for i in range(len(keywords)):
#     synsetsList = get_synsets(keywords[i])
#     print synsetsList
#     freq = count_frequency(synsetsList,document,stopWords)
#     wordFreq[keywords[i]] = freq
# print wordFreq

print keywords[1], keywords_synsets[1], cleandata[1]

# calculate frequency using provided synsets
wordFreq = numpy.zeros((len(cleandata), len(keywords)))
for n in range(len(cleandata)):
    for i in range(len(keywords)):
        freq = count_frequency(keywords[i],keywords_synsets[i],cleandata[n])

        wordFreq[n][i] = freq

print wordFreq

numpy.save("features", wordFreq)
numpy.savetxt("wordFreq.csv", wordFreq, delimiter=",")

import json

json_filename = 'small_relevant.json'
json_file = open(json_filename)
json_str = json_file.read()
json_data = json.loads(json_str)
#text = json_data[1]['Sentences']
#for i in range(0,int(len(json_data))):
#    text+= json_data[i]['Sentences']
#print(text)
#for record in json_data:
#    text += record['Sentences']
print(text)

import spacy
# Load the spacy model that you have installed
nlp = spacy.load('en')
#process a sentence using the model
# Get the mean vector for the entire sentence (useful for sentence classification etc.)

split_sent = text.split(".")
#print(split_sent)
i=1
j=1
for new_text in split_sent:
    #print(new_text)
    doc = nlp(new_text)
    for dup_text in split_sent:
        #print(dup_text)
        if(i <= j):
            i=i+1
            continue
        doc2 = nlp(dup_text)
        if(doc.similarity(doc2)>0.5):
            split_sent.remove(dup_text)
    j=j+1
    i=1
    break
print(split_sent)

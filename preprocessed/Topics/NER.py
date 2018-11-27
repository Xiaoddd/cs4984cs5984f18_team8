#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 22:04:13 2018

@author: namanchhikara
"""
def fileimport(string):
    with open(string,'r') as jsondata:
        input = json.load(jsondata)
    data=''
    for i in range(0,(len(input)-1)):
        data+= input[i]['Sentences']    
    return data

import json
from collections import Counter
import en_core_web_sm

input2= fileimport('big_relevant.json')
nlp = en_core_web_sm.load() 
doc = nlp(input2) 
pair= [(X.text, X.label_) for X in doc.ents] 
print(Counter(pair))
labels = [x.label_ for x in doc.ents]
Counter(labels)
items = [x.text for x in doc.ents]
print(Counter(items).most_common(40))




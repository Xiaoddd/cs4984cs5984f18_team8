#mapping among document relevancy, decoded file id, big data entry id
#Haitao Wang (wanght@vt.edu)

import os, json, hashlib
from numpy import genfromtxt

def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode())
    return h.hexdigest()

jsonFile = 'data/part-00000-66d9f78f-37f9-4dea-985c-6e2c040632ef-c000.json'
decode_folder = "log/pretrained_model\decode_test_400maxenc_4beam_35mindec_120maxdec_ckpt-238410/decoded"
relevancy = genfromtxt('data/rel_id_start_from_0.csv', dtype=int)

# pair document id with topic id
topic_numbers = 5
topic_ids = []
doc_ids = []
for n in range(topic_numbers):
    doc_ids += genfromtxt('data/topics_ID/Topic' + str(n) + '.csv', dtype=int).tolist()
    topic_ids += [n] * len(doc_ids)
topics = dict(zip(doc_ids, topic_ids))


# Mapping the ids
with open(jsonFile, 'r') as json_data, open('data/relevancy_mapping.csv', 'w') as rel_mapping:
    input_data = json.load(json_data)
    rel_mapping.write('Doc_id, HASH, Relavency, Topic_id, Doc_length, Decode_id, Decoded_file, Abstract, Sentence_excerpt, URL' + '\n')
    i = 0
    j = 0
    for entry in input_data:
        decode_id = 0
        decoded_file = ""
        print (i)
        url = entry['URL']
        sentence = entry['Sentences'].strip()
        length = len(sentence)
        if length > 3:
            decode_id = j
            #pad file name with zeros
            decoded_file = (str(decode_id) + '_decoded.txt').rjust(18, '0')
            abstract = open(os.path.join(decode_folder, decoded_file),'r').read().strip().replace('\n', ' ').replace('\r', '').replace(',', ' ')
            j += 1
        else:
            abstract = ""
        h = hashhex(url)

        rel = 1 if i in relevancy else 0
        topic = str(topics[i]) if i in topics else ''

        rel_mapping.write(','.join([str(i), h, str(rel), topic, str(length), str(decode_id), decoded_file, abstract, sentence[:100].lower().replace(',', ' '), url]) + '\n')
        i += 1
    rel_mapping.close()
json_data.close()

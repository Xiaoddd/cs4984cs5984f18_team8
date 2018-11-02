# process data for pointer summarization for CS4984/5984
# code from https://github.com/chmille3/process_data_for_pointer_summrizer/blob/master/json_to_hash.py

#import getopt
#import sys
import json
import hashlib
import os


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s)
    return h.hexdigest()

# as each document for NoDAPL data in the JSON file takes multiple lines, so use a different way to extract info

jsonFile = 'data/top100_bigger_corpus.json'
outputDir = 'output'
outputUrlDir = 'url'

# Make sure there is an output directory
if not os.path.exists(outputDir): os.makedirs(outputDir)
if not os.path.exists(outputUrlDir): os.makedirs(outputUrlDir)

URL_FILE = open(os.path.join(outputUrlDir, 'all_urls.txt'), 'w')

# iterate each object (rather than lines)
with open(jsonFile, 'r') as json_data:
    input_data = json.load(json_data)
    for entry in input_data:
        url = entry['URL'][0]
        sentences = entry['Sentences'][0]
        print (url, sentences)
        h = hashhex(url)

        fileName = outputDir + '/' + str(h) + '.story'  # create the. story file version of the article
        FILE = open(fileName, 'w')
        FILE.write(sentences)
        FILE.close()

        URL_FILE.write(str(url) + '\n')
    URL_FILE.close()

json_data.close()


'''
def processJSON(jsonFile, outputDir):
    # Make sure there is an output directory
    if not os.path.exists(outputDir): os.makedirs(outputDir)

    URL_FILE = open(os.path.join(outputDir, 'all_urls.txt'), 'w')

    # go through the JSON and pull each entry into separate line
    lines = []
    for line in open(jsonFile, 'r'):
        lines.append(json.loads(line))

    # at this point each line is a json dictionary for each entry in the json file

    for line in lines:
        # define here for URL and article text what the JSON dictionary keys are
        url = line['URL']
        # url = line['URL_s']
        # url = line['originalUrl']

        sentences = line['Sentences']
        # sentences = line['Sentences_t']
        # sentences = line['mergedsent']

        h = hashhex(url)

        fileName = outputDir + '/' + h + '.story'  # create the. story file version of the article
        FILE = open(fileName, 'w')
        FILE.write(sentences)
        FILE.close()

        URL_FILE.write(url + '\n')

    URL_FILE.close()


if __name__ == '__main__':

    print(sys.argv)
    print

    try:
        opts, args = getopt.getopt(sys.argv[1:], "f:h:o:")
    except getopt.GetoptError:

        print("opts:")
        print(opts)

        print('\n')
        print("args:")
        print(args)

        print("Incorrect usage of command line: ")
        print('python json_to_hash.py -f <file name> -o <output directory>')

        sys.exit(2)

    # initialize cmd line variables with default calues
    jsonFile = None
    outputDir = None

    for opt, arg in opts:
        print(opt, '\t', arg)
        if opt == '-h':
            print('python json_to_hash.py -f <file name> -o <output directory>')
            sys.exit()
        elif opt in ("-f"):
            jsonFile = arg
        elif opt in ("-o"):
            outputDir = arg

    print('\n')
    print("JSON file:", jsonFile)
    print("Output directory:", outputDir)
    print('\n')

    processJSON(jsonFile, outputDir)
'''

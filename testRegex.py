# * Partial solution to Unit 8 for ClassEvent
# * by Xuan Zhang and Tarek Kanan, Nov. 10, 2014
# * Teams are expected to learn from this, not to just use it.
# * A suitable solution for Unit 8 should be richer and tailored to YourSmall, YourBig.

from nltk.tokenize import sent_tokenize, word_tokenize
import re, os, operator, nltk


# from TextUtilsU3 import *


def is_ascii(s):
    return all(ord(c) < 128 for c in s)

# The directory location for ClassEvent documents.
classEventDir = r'log\pretrained_model\decode_test_400maxenc_4beam_35mindec_120maxdec_ckpt-238410\decoded'
#classEventDir = r'D:\CS5984\cs4984cs5984f18_team8\test'
# The set of stopwords.
stopwords = nltk.corpus.stopwords.words('english')


def main():
    '''
    A pattern that matches "in" OR "at" followed by a single word, two words, or one word + a comma one or two words.
    This is intended to match common location occurrences, like: "in Islip, New York", or "at Islip"
    '''
    locationPatternString = "((in|at)\s([A-Z][a-zA-Z]{4,}|[A-Z][a-zA-Z]{2,}\s[A-Z][a-zA-Z]{3,}))|\s+[A-Z][a-zA-Z]{3,},\s[A-Z][a-zA-Z]{2,}\s[A-Z][a-zA-Z]{3,}"

    '''
    A pattern that matches phrases relating to the Islip flood's 'girth'. This matches occurrences of
    one word followed by the word "flood", as well as "___ area" or "area of ____", which also may
    describe a flood's girth.
    '''
    girthPatternString = "([a-zA-Z]{5,}\swater|[a-zA-Z]{4,}\sarea|area\sof\s[A-Za-z]{4,})"

    '''
    A pattern that matches possible causes (or 'source') of the event. Phrases matched are: 
    "affected by ____", "result of ____", "caused by _____", "by ____", or "heavy ____" (which is more specific
    to the Islip event)
    '''
    causePatternString = "(due\sto(\s[A-Za-z]{3,}){1,3}|result\sof(\s[A-Za-z]{3,}){1,3}|caused\sby(\s[A-Za-z]{3,}){1,3}|by\s([A-Za-z]{4,}){1,2})|heavy\s([A-Za-z]{3,})"

    '''
    A pattern that matches context describing possible 'waterways' affect by the event. Phrases matched include:
    "affected ____", "water from _____", and "overflow of _____".
    '''
    waterwaysPatternString = "(affected(\s[A-Za-z]{3,}){1,3})|water\sfrom(\s[A-Za-z]{3,}){1,3}|overflow\sof(\s[A-Za-z]{3,}){1,3}"
    protestPatternString = "(protest(\s[A-Za-z]{3,}){1,3})|protect(\s[A-Za-z]{3,}){1,3}|threat\s(\s[A-Za-z]{3,}){1,3}"

    '''
    A pattern for 4-digit years
    '''
    yearPatternString = "\s\d{4}"

    '''
    A pattern for months
    '''
    monthPatternString = "(?:January|February|March|April|May|June|July|August|September|October|November|December)"

    # Compilation of regex patterns to improve repeated query efficiency.
    locationPattern = re.compile(locationPatternString)
    girthPattern = re.compile(girthPatternString)
    causePattern = re.compile(causePatternString)
    waterwaysPattern = re.compile(waterwaysPatternString)
    protestPattern = re.compile(protestPatternString)
    yearPattern = re.compile(yearPatternString)
    monthPattern = re.compile(monthPatternString)

    # A list of all files in the Class Event Directory
    listOfFiles = os.listdir(classEventDir)

    # A Dictionary to store a word and it's associated type as a tuple for the key, and the associated frequency
    # for the value.
    D = dict()

    # Loop through all of the files in the Class Event directory.
    for fileName in listOfFiles:

        # Ignores any non .txt files
        if not fileName.endswith('.txt'):
            continue

        # Stores the file's absolute path
        filePath = os.path.join(classEventDir, fileName)

        # Reads the file contents and tokenizes by sentence.
        fileContents = open(filePath.strip(), 'r').read()
        # Remove non-English words
        words = fileContents.split()
        fileContents = ""
        for w in words:
            if is_ascii(w):
                fileContents = fileContents + " " + w
        fileSentences = sent_tokenize(fileContents)

        # Calls the searchMatches function to
        searchMatches(D, locationPattern, fileSentences, fileName, "location")
        searchMatches(D, girthPattern, fileSentences, fileName, "girth")
        searchMatches(D, causePattern, fileSentences, fileName, "cause")
        searchMatches(D, waterwaysPattern, fileSentences, fileName, "waterways")
        searchMatches(D, protestPattern, fileSentences, fileName, "protest")
        searchMatches(D, yearPattern, fileSentences, fileName, "year")
        searchMatches(D, monthPattern, fileSentences, fileName, "month")

    print

    '''
    The following code is used to filter words by their Parts of Speech (POS) tag. This is useful because, for example,
    we can ignore any "location" data that is not a noun, as we know that a verb would not be useful in describing a location.
    However, the reason that is not used is because when tagging words individually with POS, we lose context.
    So if we were to tag the phrase "between 5 and 8", which describes the time of, we would tag each word individually, and lose
    the context.

    # Each element in the list is of the form: [Attribute Type, List of Words, POS Tag]
    listOfResults = []

    # Stores the result of listOfResults AFTER filtering unneeded parts of speech.
    refinedListOfResults = []

    waterwaysList = []
    causeList = []
    timeList = []
    locationList = []
    girthList = []

    # From the frequency dictionary, grab the attribute type and word, split the word (which could be several words
    # long, like "between 5 and 8") into words, and append a list containing [Attribute Type, List of Words, POS Tag]
    for typeAndWordTuple, freq in D.iteritems():
        typeOfInfo, word = typeAndWordTuple
        word = word_tokenize(word)
        listOfResults.append([typeOfInfo, word, nltk.pos_tag(word)])
        if (typeOfInfo == "time"):
            timeList.append(word[1:])

    # Sorts list of results by the Attribute Type
    listOfResults = sorted(listOfResults)


    # Creates a dictionary storing the Parts of Speech that are valuable for each specific attribute.
    typeOfInfoPOS = {}
    typeOfInfoPOS["location"] = {"NNP"}
    typeOfInfoPOS["girth"] = {"NN", "JJ"}
    typeOfInfoPOS["cause"] = {"NN", "JJ"}
    typeOfInfoPOS["waterways"] = {"NN"}


    for result in listOfResults:
        typeOfInfo = result[0]
        for resultTuple in result[2]:

            if typeOfInfoPOS.has_key(typeOfInfo) and resultTuple[1] in typeOfInfoPOS[typeOfInfo]:

                refinedListOfResults.append([typeOfInfo, resultTuple])

    print 

    for result in refinedListOfResults:
        typeOfInfo = result[0]
        if typeOfInfo == "location":
            locationList.append(result[1][0])
        elif typeOfInfo == "waterways":
            waterwaysList.append(result[1][0])
        elif typeOfInfo == "cause":
            causeList.append(result[1][0])
        elif typeOfInfo == "girth":
            girthList.append(result[1][0])

    for typeAndWordTuple, freq in D.iteritems():
        typeOfInfo, word = typeAndWordTuple
        if (typeOfInfo == "time"):
            timeList.append(word)
        elif (typeOfInfo == "location"):
            locationList.append(word)
        elif (typeOfInfo == "girth"):
            girthList.append(word)
        elif (typeOfInfo == "cause"):
            causeList.append(word)
        elif (typeOfInfo == "waterways"):
            waterwaysList.append(word)
    '''

    # Creates a frequency dictionary for each attribute type
    locationFreqDict = dict()
    waterwaysFreqDict = dict()
    protestFreqDict = dict()
    causeFreqDict = dict()
    girthFreqDict = dict()
    yearFreqDict = dict()
    monthFreqDict = dict()

    # Loops through the original frequency dictionary, and adds the correspond word and frequency
    # to the dictionary for the appropriate attribute type.
    #for typeAndWordTuple, freq in D.iteritems(): #python 2
    for typeAndWordTuple, freq in D.items():
        typeOfInfo, result = typeAndWordTuple

        if (typeOfInfo == "location" and result not in stopwords):
            try:
                locationFreqDict[result] += freq
            except:
                locationFreqDict[result] = freq

        if (typeOfInfo == "waterways" and result not in stopwords):
            try:
                waterwaysFreqDict[result] += freq
            except:
                waterwaysFreqDict[result] = freq
        if (typeOfInfo == "protest" and result not in stopwords):
            try:
                protestFreqDict[result] += freq
            except:
                protestFreqDict[result] = freq

        if (typeOfInfo == "cause" and result not in stopwords):
            try:
                causeFreqDict[result] += freq
            except:
                causeFreqDict[result] = freq

        if (typeOfInfo == "girth" and result not in stopwords):
            try:
                girthFreqDict[result] += freq
            except:
                girthFreqDict[result] = freq
        if (typeOfInfo == "year" and result not in stopwords):
            try:
                yearFreqDict[result] += freq
            except:
                yearFreqDict[result] = freq
        if (typeOfInfo == "month" and result not in stopwords):
            try:
                monthFreqDict[result] += freq
            except:
                monthFreqDict[result] = freq

    print("")

    # Sorts all of the frequency dictionaries by their frequency values in reverse order, so the greatest
    # frequency is first.
    locationFreqDict = sorted(locationFreqDict.items(), key=operator.itemgetter(1), reverse=True)
    waterwaysFreqDict = sorted(waterwaysFreqDict.items(), key=operator.itemgetter(1), reverse=True)
    protestFreqDict = sorted(protestFreqDict.items(), key=operator.itemgetter(1), reverse=True)
    causeFreqDict = sorted(causeFreqDict.items(), key=operator.itemgetter(1), reverse=True)
    girthFreqDict = sorted(girthFreqDict.items(), key=operator.itemgetter(1), reverse=True)
    yearFreqDict = sorted(yearFreqDict.items(), key=operator.itemgetter(1), reverse=True)
    monthFreqDict = sorted(monthFreqDict.items(), key=operator.itemgetter(1), reverse=True)

    # Prints the top 10 words for each attribute.
    print("Top 10 frequent values for each attribute:")
    print("Location:", locationFreqDict[:10], "\n")
    print("Waterways:", waterwaysFreqDict[:10], "\n")
    print("Protest:", protestFreqDict[:10], "\n")
    print("Cause:", causeFreqDict[:10], "\n")
    print("Girth:", girthFreqDict[:10], "\n")
    print("Year:", yearFreqDict[:10], "\n")
    print("Month:", monthFreqDict[:10], "\n")

    # Prints the original template.
    print("Template before filling-out:")
    print("On {Time} a {Girth} caused by {Cause} {Waterways} in {Location}.\n")
    # Prints the highest frequency result for each attribute in the formated template.
    print("Template after filling-out:")
    print("On {0} {1} a {2} caused by {3} {4} in {5}.".format(monthFreqDict[0][0], yearFreqDict[0][0],
                                                              girthFreqDict[0][0], causeFreqDict[0][0],
                                                              waterwaysFreqDict[1][0], locationFreqDict[0][0]))


# Prints any matches in the files with their corresponding filename and location in the file.
# Also creates a frequency dictionary for words and their attributes.
def searchMatches(D, pattern, fileSentences, fileName, typeOfInfo):
    # Loop over all sentences in the file.
    for sentence in fileSentences:

        # Loop over all matches from the regex object.
        for match in pattern.finditer(sentence):

            # Splits the match into words
            result = match.group().split()

            # Filters any words of length 2 or less.
            result = [w for w in result if len(w) > 2]

            # Joins filtered set words with spaces
            result = " ".join(w for w in result)

            # Display the filename and location in the file at which a match was found.
            # print "{4}: {0}: {1}-{2}: {3}".format(fileName, match.start(), match.end(), result, typeOfInfo)

            # Increment the frequency for this attribute and word pair
            try:
                D[typeOfInfo, result] += 1
            except:
                D[typeOfInfo, result] = 1


#if __name__ == "__main__": main()
main()

import sys
import json
import os
import re
import codecs
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import percepclassify
from io import TextIOWrapper


def formwordshape(word):
    word = re.sub('[A-Z]+','A',word)
    word = re.sub('[a-z]+','a',word)
    word = re.sub('[0-9]+','d',word)
    word = re.sub('[^0-9a-zA-Z]+','#',word)
    return word


def createfeaturevector(window, withclass = False):
    f = {}

    if(withclass == True):
        word, pos1, ner = window[1].rsplit('/', 2)
        lword, pos2, ner1 = window[0].rsplit('/', 2)
        rword, pos3, ner2 = window[2].rsplit('/', 2)
    else:
        word, pos1  = window[1].rsplit('/', 1)
        lword, pos2 = window[0].rsplit('/', 1)
        rword, pos3 = window[2].rsplit('/', 1)

    f['__1__' + word.lower()] = 1
    f['__2__' + lword.lower()] = 1
    f['__3__' + rword.lower()] = 1
    f['__4__' + formwordshape(word)] = 1
    f['__5__' + pos1] = 1
    f['__6__' + pos2] = 1
    f['__7__' + pos3] = 1

    if(withclass == True):
        return ner, f

    return f


def tokenize(line):
    tokens = line.rstrip('\n ').split(' ')
    tokens.insert(0, '__UNK__/__UNK__/__UNK__')
    tokens.append('__END__/__END__/__END__')
    return tokens


if __name__ == '__main__':

    if(len(sys.argv) != 2):
        print('Format -> python3 netag.py MODEL')
        sys.exit()

    modelfilepath = sys.argv[1]
    model_data = open(modelfilepath)
    model_obj = json.load(model_data)
    classifier = percepclassify.PercepClassifier(model_obj)

    for line in TextIOWrapper(sys.stdin.buffer, encoding='iso-8859-1').readlines():
        tokens = tokenize(line)
        numtokens = len(tokens) - 2
        for i in range(0, numtokens):
            window = tokens[i:i+3]
            f = createfeaturevector(window)
            label = classifier.classify(f)
            out = window[1] + '/' + label
            print(out, end=' ')
        print('')
    # classifier.calculatestats()

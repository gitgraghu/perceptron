import sys
import json
import os
import re
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import percepclassify


def formwordshape(word):
    word = re.sub('[A-Z]+','A',word)
    word = re.sub('[a-z]+','a',word)
    word = re.sub('[0-9]+','d',word)
    word = re.sub('[^0-9a-zA-Z]+','#',word)
    return word


def derivesuffixes(word):
    suffix1 = ''
    suffix2 = ''
    if(len(word) >= 3):
        suffix1 = word[len(word)-2:]
    if(len(word) >= 4):
        suffix2 = word[len(word)-3:]
    return suffix1, suffix2


def createfeaturevector(window, withclass = False):
    f = {}

    start = i
    end = i+3
    window = tokens[start:end]

    if(withclass == True):
        word, pos = window[1].split('/')
        lword = window[0].split('/')[0]
        rword = window[2].split('/')[0]
    else:
        word  = window[1]
        lword = window[0]
        rword = window[2]

    f['__1__' + word.lower()] = 1
    f['__2__' + lword.lower()] = 1
    f['__3__' + rword.lower()] = 1
    f['__4__' + formwordshape(word)] = 1

    suffix1, suffix2 = derivesuffixes(word.lower())
    f['__5__' + suffix1] = 1
    f['__6__' + suffix2] = 1

    if(withclass == True):
        return pos, f

    return f

def tokenize(line):
    tokens = line.rstrip('\n ').split(' ')
    tokens.insert(0, '__UNK__/__UNK__')
    tokens.append('__END__/__END__')
    return tokens

if __name__ == '__main__':

    if(len(sys.argv) != 2):
        print('Format -> python3 postag.py MODEL')
        sys.exit()

    modelfilepath = sys.argv[1]
    model_data = open(modelfilepath)
    model_obj = json.load(model_data)

    classifier = percepclassify.PercepClassifier(model_obj)

    for line in sys.stdin.readlines():
        tokens = tokenize(line)
        numtokens = len(tokens) - 2
        for i in range(0, numtokens):
            window = tokens[i:i+3]
            f = createfeaturevector(window)
            label = classifier.classify(f)
            print(window[1] + '/' + label, end=' ')
        print('')

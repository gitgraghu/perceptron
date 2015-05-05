import sys
import json
import os
import re
import getopt
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import perceplearn

def formwordshape(word):
    word = re.sub('[A-Z]+','A',word)
    word = re.sub('[a-z]+','a',word)
    word = re.sub('[0-9]+','d',word)
    word = re.sub('[^0-9a-zA-Z]+','#',word)
    return word

def developfeatureset(datafile):
    featureset = []

    for line in datafile:
        tokens = line.rstrip('\n ').split(' ')
        numoftokens = len(tokens)
        tokens.insert(0, '__UNK__/__UNK__/__UNK__')
        tokens.append('__END__/__END__/__END__')

        for i in range(0, numoftokens):
            f = {}

            start = i
            end = i+3
            window = tokens[start:end]

            word, pos1, ner = window[1].rsplit('/', 2)
            lword, pos2, ner1 = window[0].rsplit('/', 2)
            rword, pos3, ner2 = window[2].rsplit('/', 2)

            f['__1__' + word.lower()] = 1
            f['__2__' + lword.lower()] = 1
            f['__3__' + rword.lower()] = 1
            f['__4__' + formwordshape(word)] = 1
            f['__5__' + pos1] = 1
            f['__6__' + pos2] = 1
            f['__7__' + pos3] = 1

            featureset.append((ner, f))

    return featureset


if __name__ == '__main__':

    if(len(sys.argv) < 3):
        print('Format -> python3 nelearn.py <TRAININGFILE> <MODEL> -h <DEVFILE>')
        sys.exit()

    devset=[]
    optlist, args = getopt.gnu_getopt(sys.argv, 'h:')
    for o, a in optlist:
        if o == '-h':
            devfile = open(a, 'r', encoding = 'iso-8859-1')
            devset = developfeatureset(devfile)

    trainingfilename = sys.argv[1]
    modelfilepath = sys.argv[2]

    trainingfile = open(trainingfilename, 'r', encoding = 'iso-8859-1')
    trainingset = developfeatureset(trainingfile)
    avgmodel = perceplearn.createpercepmodel(trainingset, devset)

    with open(modelfilepath, 'w+') as modelfile:
            json.dump(avgmodel, modelfile)

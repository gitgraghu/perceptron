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


def derivesuffixes(word):
    suffix1 = ''
    suffix2 = ''
    if(len(word) >= 3):
        suffix1 = word[len(word)-2:]
    if(len(word) >= 4):
        suffix2 = word[len(word)-3:]
    return suffix1, suffix2


def developfeatureset(datafile):
    featureset = []

    for line in datafile:
        tokens = line.rstrip('\n ').split(' ')
        numoftokens = len(tokens)
        tokens.insert(0, '__UNK__/__UNK__')
        tokens.append('__END__/__END__')

        for i in range(0, numoftokens):
            f = {}

            start = i
            end = i+3
            window = tokens[start:end]

            word, pos = window[1].split('/')
            lword = window[0].split('/')[0]
            rword = window[2].split('/')[0]

            f['__1__' + word.lower()] = 1
            f['__2__' + lword.lower()] = 1
            f['__3__' + rword.lower()] = 1
            f['__4__' + formwordshape(word)] = 1

            suffix1, suffix2 = derivesuffixes(word.lower())
            f['__5__' + suffix1] = 1
            f['__6__' + suffix2] = 1

            featureset.append((pos, f))

    return featureset

if __name__ == '__main__':

    if(len(sys.argv) < 3):
        print('Format -> python3 postrain.py <TRAININGFILE> <MODEL> -h <DEVFILE>')
        sys.exit()

    devset=[]
    optlist, args = getopt.gnu_getopt(sys.argv, 'h:')
    for o, a in optlist:
        if o == '-h':
            devfile = open(a, 'r', errors='ignore')
            devset = developfeatureset(devfile)

    trainingfilename = args[1]
    modelfilepath = args[2]

    trainingfile = open(trainingfilename, 'r', errors='ignore')
    trainingset = developfeatureset(trainingfile)
    avgmodel = perceplearn.createpercepmodel(trainingset,devset)

    with open(modelfilepath, 'w+') as modelfile:
            json.dump(avgmodel, modelfile)

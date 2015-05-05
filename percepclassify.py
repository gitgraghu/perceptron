import sys
import json


def dotproduct(a, b):
    result = 0
    for key in b:
        if key in a:
            result = result + b[key]*a[key]
    return result

class PercepClassifier():

    def __init__(self, model_obj):
        self.model = model_obj
        self.stats = {}

    def classifyforstats(self, f, y):
        cclass = self.classify(f)

        correctcount = self.stats.setdefault('correctcount', {})
        classifiedcount = self.stats.setdefault('classifiedcount', {})
        actualcount = self.stats.setdefault('actualcount',{})

        actualcount[y] = actualcount.setdefault(y, 0) + 1
        classifiedcount[cclass] = classifiedcount.setdefault(cclass, 0) + 1

        if(cclass == y):
            correctcount[y] = correctcount.setdefault(y, 0) + 1

    def calculatestats(self):
        correctcount = self.stats.setdefault('correctcount', {})
        classifiedcount = self.stats.setdefault('classifiedcount', {})
        actualcount = self.stats.setdefault('actualcount',{})
        Fscore = {}
        precision = {}
        recall = {}
        for classification in actualcount:
            ccorrectcount = correctcount[classification]
            cclassifiedcount = classifiedcount[classification]
            cactualcount = actualcount[classification]
            p = ccorrectcount/cclassifiedcount
            r = ccorrectcount/cactualcount
            F = (2*p*r)/(p+r)
            Fscore[classification] = F
            precision[classification] = p
            recall[classification] = r
        print(Fscore, precision, recall)

    def classify(self, f):
        z = 0
        Z = {}

        for classification in self.model:
           weights = self.model[classification]
           z = dotproduct(weights, f)
           Z[classification] = z

        max_value = max(Z.values())
        max_result = [key for key, value in Z.items() if value == max_value]
        max_class = max_result[0]
        return max_class


def createfeaturevector(line):
    tokens = line.rstrip('\n').split(' ')
    f = {}
    for token in tokens:
        f[token] = f.setdefault(token, 0) + 1

    return f


if __name__ == '__main__':

    # Check if arguments are passed correctly
    if(len(sys.argv) != 2):
        print('Format -> python3 percepclassify.py MODEL')
        sys.exit()

    # Load model data
    modelfilepath = sys.argv[1]
    model_data = open(modelfilepath)
    model_obj = json.load(model_data)

    # Initialize Classifier Object
    classifier = PercepClassifier(model_obj)

    # Go through input line by line
    for line in sys.stdin.readlines():
        f = createfeaturevector(line)
        label = classifier.classify(f)
        print(label)

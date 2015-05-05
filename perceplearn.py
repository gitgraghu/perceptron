import sys
import json
import random
import percepclassify
import getopt


def dotproduct(a, b):
    result = 0
    for key in b:
        if key in a:
            result = result + b[key]*a[key]
    return result


def createpercepmodel(trainingset, devset=[]):

    # Initialize weight and cached weights
    model  = {}
    cached = {}
    c = 1

    # Perceptron learn iterations
    for i in range(0, 20):

        # Initializing data to calculate statistics
        total = 0
        inaccurate = 0

        # Shuffle training set
        random.shuffle(trainingset)

        # Go through all samples in training set
        for trainingsample in trainingset:

            # 'y' is actual class and 'f' is feature vector
            y = trainingsample[0]
            f = trainingsample[1]
            total = total + 1

            # Make sure that class 'y' is present in weight models
            model.setdefault(y, {})
            cached.setdefault(y, {})

            Z = {}

            # Calculate perceptron value of every class in model
            for classification in model:
                weights = model[classification]
                z = dotproduct(weights, f)
                Z[classification] = z

            # max_class is the class which has maximum perceptron value
            max_value = max(Z.values())
            max_result = [key for key, value in Z.items() if value == max_value]
            max_class = max_result[0]

            # If perceptron was inaccurate adjust weight vectors
            if(max_class != y):
                inaccurate = inaccurate + 1
                wz = model[max_class]
                wy = model[y]
                uz = cached[max_class]
                uy = cached[y]
                for feature in f:
                    wz[feature] = wz.setdefault(feature, 0) - f[feature]
                    wy[feature] = wy.setdefault(feature, 0) + f[feature]
                    uz[feature] = uz.setdefault(feature, 0) - c*f[feature]
                    uy[feature] = uy.setdefault(feature, 0) + c*f[feature]

        c = c + 1
        accuracy = 1.0 - (inaccurate/total)
        print('Training Accuracy: ' + str(accuracy) + ' Innacurate: ' + str(inaccurate))

        avgmodel = {}

        for classification in model:
            u = cached[classification]
            avgweights = dict(model[classification])
            for feature in avgweights:
                avgweights[feature] = avgweights[feature] - (u[feature]/c)
            avgmodel[classification] = avgweights

        if(len(devset) > 0):
            inaccurate = 0
            total = len(devset)
            classifier = percepclassify.PercepClassifier(avgmodel)
            random.shuffle(devset)
            for devsample in devset:
                y = devsample[0]
                f = devsample[1]
                cclass = classifier.classify(f)
                if(y != cclass):
                    inaccurate = inaccurate + 1
            accuracy = 1.0 - (inaccurate/total)
            print('Development Accuracy: ' + str(accuracy) + ' Innacurate: ' + str(inaccurate))

    return avgmodel


def createfeaturevector(line):
    tokens = line.rstrip('\n').split(' ')
    y = tokens.pop(0)

    f = {}
    for token in tokens:
        f[token] = f.setdefault(token, 0) + 1

    return (y,f)


def developfeatureset(datafile):
    featureset = []

    for line in datafile:
        y, f = createfeaturevector(line)
        featureset.append((y, f))

    return featureset


if __name__ == '__main__':

    if(len(sys.argv) < 3):
        print('Format -> python3 perceplearn.py TRAININGFILE MODELFILE -h DEVFILE')
        sys.exit()

    devset = []
    optlist, args = getopt.gnu_getopt(sys.argv, 'h:')
    for o, a in optlist:
        if o == '-h':
            devfile = open(a, 'r', errors='ignore')
            devset = developfeatureset(devfile)

    trainingfilepath = args[1]
    modelfilepath = args[2]

    trainingfile = open(trainingfilepath, 'r', errors='ignore')

    trainingset = developfeatureset(trainingfile)

    avgmodel = createpercepmodel(trainingset, devset)

    with open(modelfilepath, 'w+') as modelfile:
        json.dump(avgmodel, modelfile)

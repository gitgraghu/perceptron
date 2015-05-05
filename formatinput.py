import sys
import os

# Check if args are supplied in command line
if(len(sys.argv) != 4):
    print('Format -> python3 formatinput.py <dir> <outfile> <formattype>')
    sys.exit()

# Get the training directory and output file name from args
trainingdir = sys.argv[1]
outputfilename = sys.argv[2]
formattype = sys.argv[3]

# Check if training_dir is a directory
if(not os.path.isdir(trainingdir)):
    print(trainingdir + ' is not a directory.')
    sys.exit()

# Open the output file for writing
outputfile = open(outputfilename, 'w')

# Get the list of files in training directory
files = os.listdir(trainingdir)

# Go through all files, format input and write to outputfile
for f in files:
    trainingfilepath = trainingdir + os.path.sep + f
    with open(trainingfilepath, 'r', errors='ignore') as trainingfile:
        splitstrings = f.split('.')
        if(formattype == 'training'):
            label = splitstrings[0]
            outputfile.write(label + ' ')
        for line in trainingfile:
            outputfile.write(line.rstrip('\n'))
        outputfile.write('\n')
outputfile.close()

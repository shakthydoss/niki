import json
import sys

import nltk
import numpy as np
from nltk.stem.lancaster import LancasterStemmer

from model import classify
from model import train

words = []
classes = []
documents = []
ignore_words = ['?']
stemmer = LancasterStemmer()

with open('train-data.txt') as f:
    content = f.readlines()
for item in content:
    sents, cls = item.strip().split(",,,")
    sents = sents.strip()
    cls = cls.strip()
    # tokenize
    w = nltk.word_tokenize(sents)
    words.extend(w)
    documents.append((w, cls))
    if cls not in classes:
        classes.append(cls)

words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = list(set(words))
classes = list(set(classes))

training = []
output = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    training.append(bag)
    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    output.append(output_row)

X = np.array(training)
y = np.array(output)

# probability threshold
ERROR_THRESHOLD = 0.0005
# load our calculated synapse values
synapse_file = 'synapses.json'
with open(synapse_file) as data_file:
    synapse = json.load(data_file)
    synapse_0 = np.asarray(synapse['synapse0'])
    synapse_1 = np.asarray(synapse['synapse1'])

print sys.argv

if len(sys.argv) <= 1:
    print("Error : Argument missing. Run like...\npython process.py train\npython process.py test\n")
    sys.exit()

if sys.argv[1] == 'train':
    train(X, y, classes, words, hidden_neurons=10, alpha=0.021, epochs=50000, dropout=True, dropout_percent=0.2)
if sys.argv[1] == 'test':
    with open('test-data.txt') as f:
        content = f.readlines()
        for item in content:
            sents = item.strip()
            predicted_cls = classify(sents, stemmer, classes, words, synapse_0, synapse_1, ERROR_THRESHOLD)
            print predicted_cls, sents

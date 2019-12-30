import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tensorflow
import tflearn
import random
import json
import pickle

with open('data.json') as file:
    data = json.load(file)
try:
    with open('data.pickle', 'rb') as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for item in data['data']:
        for pattern in item['patterns']:
            splitword = nltk.word_tokenize(pattern)
            words.extend(splitword)
            docs_x.append(splitword)
            docs_y.append(item['tag'])

        if item['tag'] not in labels:
            labels.append(item['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w != '?']
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        splitword = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in splitword:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open('data.pickle', 'wb') as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load('models/model.tflearn')
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("models/model.tflearn")
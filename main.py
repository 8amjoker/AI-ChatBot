import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tensorflow
import tflearn
import random
import json

with open('data.json') as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y =[]

for item in data['data']:
    for pattern in item['patterns']:
        splitword = nltk.word_tokenize(pattern)
        words.extends(splitword)
        docs_x.append(pattern)
        docs_y.append(item['tag'])

    if item['tag'] not in labels:
        labels.append(item['tag'])

words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))
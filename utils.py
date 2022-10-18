import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
# nltk.download('punkt')

stemmer = PorterStemmer()

tokenize = lambda sentence : nltk.word_tokenize(sentence)

"""stemming - find the root form of the word"""
stem = lambda word: stemmer.stem(word.lower())

def bag_of_words(token_sent, words):
    stems = [stem(word) for word in token_sent]
    bag = np.zeros(len(words), dtype=np.float32)
    for i, word in enumerate(words):
        if word in stems:
            bag[i] = 1.0
    return bag
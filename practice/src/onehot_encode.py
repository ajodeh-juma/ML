#!/usr/bin/env python

from __future__ import print_function, division
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import euclidean_distances
import nltk
#nltk.download()
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
from sys import argv



"""explanatory variables encoded using one binary feature for each of the variable's possible values"""

onehot_encoder = DictVectorizer()
instances = [
{'city': 'New York'},
{'city': 'San Francisco'},
{'city': 'Chapel Hill'}
]

print (onehot_encoder.fit_transform(instances).toarray())

# extracting features from text - bag-of-words representation, document classification and retrieval

corpus = ['UNC played Duke in basketball', 'Duke lost the basketball game', 'I ate a sandwich', 'Duke lost the basketball game']
vectorizer = CountVectorizer()
print (vectorizer.fit_transform(corpus).todense())
print (vectorizer.vocabulary_)

# compare feature vectors
counts = vectorizer.fit_transform(corpus).todense()
print('Distance between 1st and 2nd documents:', euclidean_distances(counts[0], counts[1]))
print('Distance between 1st and 3rd documents:', euclidean_distances(counts[0], counts[2]))
print('Distance between 2nd and 3rd documents:', euclidean_distances(counts[1], counts[2]))
print('Distance between 2nd and 4th documents:', euclidean_distances(counts[1], counts[3]))

# stemming and lemmatization
copus = ['The dog ate a sandwich, the wizard transfigured a sandwich, and I ate a sandwich']
vectorizer = CountVectorizer(stop_words='english')
print (vectorizer.fit_transform(copus).todense())
print (vectorizer.vocabulary_)
#lemmatizer = WordNetLemmatizer()
#print (lemmatizer.lemmatize('gathering', 'v'))
#print (lemmatizer.lemmatize('gathering', 'n'))


def features(seqfile):
	seqs = []
	for line in seqfile:
		if line.startswith('>'):
			continue
		else:
			seqs.append(line.strip())

	
	#vectorizer = TfidfVectorizer()
	#vectorizer = CountVectorizer()
	vectorizer = HashingVectorizer(n_features=20)
	counts = vectorizer.fit_transform(seqs).todense()
	print (counts)





if __name__ == '__main__':
	sq = features(open(argv[1]))
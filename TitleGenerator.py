# -*- coding: utf-8 -*-
# Henry Dinh - HXD130130@UTDallas.edu
# Title Generator
# Takes in an article and outputs a relevant title


from unidecode import unidecode
import sys
import nltk
import string
import copy
import math
import random


# Gets the tokens in a string and strips end punctuation
def tokenize(phrase):
	words = []
	for line in phrase.split("\n"):
		for word in line.split(" "):
			words.append(word.strip(string.punctuation).lower())
	return words
	

# prints a matrix in a neat format
def printMatrix(table):
	col_width = [max(len(str(x)) for x in col) for col in zip(*table)]
	for line in table:
		print " | " + " | ".join("{:{}}".format(x, col_width[i]) for i, x in enumerate(line)) + " |"
	print	


# Gets the dot product between two vectors
def dot(vector1, vector2):
	if len(vector1) != len(vector2):
		return 0.0
	return sum(i[0] * i[1] for i in zip(vector1, vector2))
	

# Gets the magnitude of a vector (double bars)	
def magnitude(vector):
	return math.sqrt(sum(i**2 for i in vector))
	

# Returns a similarity matrix based on Cosine Similarity
def similarityMatrix(matrix):
	similarity_matrix = [[0] * len(matrix) for i in range(len(matrix))]
	for i in range(len(similarity_matrix)):
		for j in range(len(similarity_matrix[i])):
			similarity_matrix[i][j] = round(1.0 * dot(matrix[i], matrix[j]) / (magnitude(matrix[i]) * magnitude(matrix[j])), 5)
	return similarity_matrix
	
	
def textRank(damper, matrix, iterations):
	# Get probability of going from a node to its neighbor based on matrix
	probability = copy.deepcopy(matrix)
	for i in range(len(matrix)):
		total = sum(x if x != i else 0.0 for x in range(len(matrix[i])))
		for j in range(len(matrix[i])):
			probability[i][j] = 0.0 if i == j else matrix[i][j] / total
	# 0.0 initial score for each sentence
	page_rank = [0] * len(matrix)
	# start walking at a random node
	start_node = random.randint(0, len(page_rank) - 1)
	iteration = 0
	while iteration < iterations:
		rand = random.uniform(0.0, 1.0)
		if rand <= damper:
			# go to neighbor node randomly based on probability
			prob = random.uniform(0.0, 1.0)
			for i in range(len(page_rank)):
				prob -= probability[start_node][i]
				if prob <= 0:
					start_node = i
					break
			page_rank[start_node] += 1
		else:
			# start a new walk at a random node. Only increase iteration when a new walk is started
			iteration += 1
			start_node = random.randint(0, len(page_rank) - 1)
			#page_rank[start_node] += 1
	return page_rank
	

# prints the title formatted
def printTitle(title):
	print string.capwords(title.strip(string.punctuation))
	

# replaces uncommon unicode characters	
def cleanText(text):
	text = text.replace('“','"').replace('”','"').replace('–','-').replace('’','\'')
	return text
	

# Make sure user provides an article
if len(sys.argv) != 2:
	print "Usage: python TitleGenerator.py <article-name>"
	sys.exit()

	
# Get the article and clean it
article = open(sys.argv[1], 'r').read()
article = cleanText(article)

# Get unique tokens in article ignoring punctuation. inverse_words is word : index
words = dict(list(enumerate(list(set(tokenize(article))))))
inverse_words = {y:x for x,y in words.iteritems()}

# get the unigram counts of all words
word_counts = copy.deepcopy(inverse_words)
word_counts = dict.fromkeys(word_counts, 0)

# Separate the article into sentences
sentences = dict(list(enumerate(nltk.sent_tokenize(article))))
for i in sentences:
	sentences[i] = sentences[i].replace('\n', ' ')

# Get words in each sentence. keep duplicates for counting later
sentence_words = dict(list(enumerate([tokenize(sentences[s]) for s in sentences])))

# Build matrix of words and count times it appears in each sentence
# Row is the sentence #. Column is the word #
matrix = [[0] * len(words) for i in range(len(sentences))]
for s in sentence_words:
	for word in sentence_words[s]:
		matrix[s][inverse_words[word]] += 1
		word_counts[word] += 1

# counts the number of sentences a word appears in
word_sent_count = copy.deepcopy(inverse_words)
word_sent_count = dict.fromkeys(word_counts, 0)
for j in range(len(matrix[0])):
	count = 0
	for i in range(len(matrix)):
		if matrix[i][j] > 0:
			count += 1
	word_sent_count[words[j]] = count	

# Normalize the matrix with term frequency (tf) and inverse doc frequency (idf)
for i in range(len(matrix)):
	for j in range(len(matrix[i])):
		tf = 1.0 * matrix[i][j] / len(sentence_words[i])
		idf = math.log(1.0 * len(sentences) / (1 + word_sent_count[words[j]]))
		tf_idf = round(tf * idf, 5)
		matrix[i][j] = tf_idf

# Construct Cosine Similarity Matrix with the normalized tf-idf matrix
similarity_matrix = similarityMatrix(matrix)

# Use TextRank algorithm to score the sentences in the graph
text_rank = textRank(.85, matrix, 1000)

# Get the best ranked sentence and use it as the title
best = 0
for i in range(len(text_rank)):
	if text_rank[i] >= best:
		best = i
title = sentences[best]
printTitle(title)


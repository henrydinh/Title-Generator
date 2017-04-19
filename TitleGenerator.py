# Henry Dinh - HXD130130@UTDallas.edu
# Title Generator
# Takes in an article and outputs a relevant title


import sys
import nltk
import string
import copy
import math


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
	

# Make sure user provides an article
if len(sys.argv) != 2:
	print "Usage: python TitleGenerator.py <article-name>"
	sys.exit()
	
# Get the article
article = open(sys.argv[1], 'r').read()
print "Article: "
print article
print

# Get unique tokens in article ignoring punctuation. inverse_words is word : index
words = dict(list(enumerate(list(set(tokenize(article))))))
inverse_words = {y:x for x,y in words.iteritems()}
print "Words: "
print words
print
print "Inverse words: "
print inverse_words
print

# get the unigram counts of all words
word_counts = copy.deepcopy(inverse_words)
word_counts = dict.fromkeys(word_counts, 0)

# Separate the article into sentences
sentences = dict(list(enumerate(nltk.sent_tokenize(article))))
for i in sentences:
	sentences[i] = sentences[i].replace('\n', ' ')
print "Sentences: "
print sentences
print

# Get words in each sentence. keep duplicates for counting later
sentence_words = dict(list(enumerate([tokenize(sentences[s]) for s in sentences])))
print "Words in sentences as a list: "
print sentence_words
print

# Build matrix of words and count times it appears in each sentence
# Row is the sentence #. Column is the word #
matrix = [[0] * len(words) for i in range(len(sentences))]
for s in sentence_words:
	for word in sentence_words[s]:
		matrix[s][inverse_words[word]] += 1
		word_counts[word] += 1
print "Matrix: "
printMatrix(matrix)
print
print "Word counts: "
print word_counts
print

# counts the number of sentences a word appears in
word_sent_count = copy.deepcopy(inverse_words)
word_sent_count = dict.fromkeys(word_counts, 0)
for j in range(len(matrix[0])):
	count = 0
	for i in range(len(matrix)):
		if matrix[i][j] > 0:
			count += 1
	word_sent_count[words[j]] = count
print "Number of documents each word appears in: "			
print word_sent_count
print		

# Normalize the matrix with term frequency (tf) and inverse doc frequency (idf)
for i in range(len(matrix)):
	for j in range(len(matrix[i])):
		tf = 1.0 * matrix[i][j] / len(sentence_words[i])
		idf = math.log(1.0 * len(sentences) / (1 + word_sent_count[words[j]]))
		tf_idf = round(tf * idf, 5)
		matrix[i][j] = tf_idf
print "Normalized matrix: "
printMatrix(matrix)
print

# Construct Cosine Similarity Matrix with the normalized tf-idf matrix
similarity_matrix = similarityMatrix(matrix)
print "Similarity Matrix: "
printMatrix(similarity_matrix)
print

# Use PageRank algorithm to score the sentences in the graph








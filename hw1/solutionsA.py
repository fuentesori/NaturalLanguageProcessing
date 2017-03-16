from __future__ import division
import math
import nltk
import time
import numpy as np
from collections import Counter

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):

    uni_corpus = [s[:-2] + STOP_SYMBOL for s in training_corpus]
    bi_corpus = [START_SYMBOL + ' ' + s[:-2] + STOP_SYMBOL for s in training_corpus]
    tri_corpus = [START_SYMBOL+ ' ' + START_SYMBOL + ' '+ s[:-2] + STOP_SYMBOL for s in training_corpus]

    unigram_tuples=[]
    bitokens=[]
    tritokens=[]
    for a in uni_corpus:
        unigram_tuples.extend(a.split(' '))
    for a in bi_corpus:
        bitokens.extend(a.split(' '))
    for a in tri_corpus:
        tritokens.extend(a.split(' '))
    bigram_tuples = list(nltk.bigrams(bitokens))
    trigram_tuples = list(nltk.trigrams(tritokens))
    trigram_bi_dict = list(nltk.bigrams(tritokens))


    uni_count = Counter(gram for gram in unigram_tuples)
    bi_count = Counter(gram for gram in bigram_tuples)
    bi_dict = Counter(token for token in bitokens)
    tri_count = Counter(gram for gram in trigram_tuples)
    tri_dict = Counter(gram for gram in trigram_bi_dict)


    uni_total = sum(uni_count.values())
    bi_total = sum(bi_count.values())
    tri_total = sum(tri_count.values())

    unigram_p = Counter()
    bigram_p = Counter()
    trigram_p = Counter()
    for k in uni_count.keys():
        unigram_p[k] = np.log2(uni_count[k] /uni_total)
    for k,b in bi_count.keys():
        bigram_p[k,b] = np.log2(bi_count[k,b]/bi_dict[k])
    for k,b,c in tri_count.keys():
        trigram_p[k,b,c] = np.log2(tri_count[k,b,c]/tri_dict[k,b])

    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc.
def score(ngram_p, n, corpus):

    if n==1:
        corpus = [s[:-2] + STOP_SYMBOL for s in corpus]
    elif n==2:
        corpus = [START_SYMBOL + ' ' + s[:-2] + STOP_SYMBOL for s in corpus]
    elif n==3:
        corpus = [START_SYMBOL+ ' ' + START_SYMBOL + ' '+ s[:-2] + STOP_SYMBOL for s in corpus]

    corpus = [a.split(' ') for a in corpus]
    prob = 0

    tuples = corpus

    if n==2:
        tuples = [list(nltk.bigrams(a)) for a in corpus]
    if n==3:
        tuples = [list(nltk.trigrams(a)) for a in corpus]
    scores = []
    for a in tuples:
        for k in a:
            prob = prob + ngram_p[k]
        if prob == 0:
            prob = MINUS_INFINITY_SENTENCE_LOG_PROB
        scores.append(prob)
        prob = 0
    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

#TODO: IMPLEMENT THIS FUNCTION
# Calculcates the perplexity of a language model
# scores_file: one of the A2 output files of scores
# sentences_file: the file of sentences that were scores (in this case: data/Brown_train.txt)
# This function returns a float, perplexity, the total perplexity of the corpus
def calc_perplexity(scores_file, sentences_file):

    words =[]
    infile = open(scores_file, 'r')
    scores = infile.readlines()
    infile.close()
    scores = [s[:-1] for s in scores]
    scores = [float(score) for score in scores]
    infile = open(sentences_file, 'r')
    sentences = infile.readlines()
    infile.close()
    sentences = [a[:-2]+' '+STOP_SYMBOL for a in sentences]
    for a in sentences:
        words.extend(a.split(' '))
    s = Counter(word for word in words)
    totalwords = sum(s.values())
    perplexity = 2**((-1/totalwords)*(np.sum(scores)))

    return perplexity

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):

    corpus = [START_SYMBOL+ ' ' + START_SYMBOL + ' '+ s[:-2] + STOP_SYMBOL for s in corpus]
    corpus = [a.split(' ') for a in corpus]

    newprobs = Counter()
    for k,b,c in trigrams:
        newprobs[k,b,c] = np.log2((2**trigrams[k,b,c])/3+(2**bigrams[b,c])/3+(2**unigrams[c])/3)
    tuples = [list(nltk.trigrams(a)) for a in corpus]

    scores = []
    prob = 0
    for a in tuples:
        for k in a:
            if k not in newprobs:
                prob = MINUS_INFINITY_SENTENCE_LOG_PROB
                break
            prob = prob + newprobs[k]
        scores.append(prob)
        prob = 0


    return scores

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close()

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()

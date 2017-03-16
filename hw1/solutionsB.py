from __future__ import division
import sys
import nltk
import math
import time
import numpy as np
from collections import Counter

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# TODO: IMPLEMENT THIS FUNCTION
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):

    brown_train = [START_SYMBOL+'/*'+' '+START_SYMBOL+'/*'+' '+s[:-2]+STOP_SYMBOL +'/STOP' for s in brown_train]
    split_corpus = [a.split(' ') for a in brown_train]
    brown_words = []
    brown_tags = []
    wtemp = []
    ttemp = []
    for a in range(len(split_corpus)):
        for k in split_corpus[a]:
            wtemp.extend(k.rsplit('/',1)[0::2])
            ttemp.extend(k.rsplit('/',1)[1::2])
        brown_words.append(wtemp)
        brown_tags.append(ttemp)
        wtemp =[]
        ttemp =[]


    # brown_train = [s[:-2]+STOP_SYMBOL +'/STOP_SYMBOL' for s in brown_train]
    # split_corpus = [a.split(' ') for a in brown_train]
    # brown_words = []
    # brown_tags = []
    # for a in split_corpus:
    #     for k in a:
    #         brown_words.extend(k.rsplit('/',1)[0::2])
    #         brown_tags.extend(k.rsplit('/',1)[1::2])
    return brown_words, brown_tags


# TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    brown_tags2 =[]
    for a in brown_tags:
        brown_tags2.extend(a)
    q_values = Counter()
    tri_tag_tuples = list(nltk.trigrams(brown_tags2))
    trigram_bi_dict = list(nltk.bigrams(brown_tags2))
    tri_dict = Counter(gram for gram in trigram_bi_dict)
    tri_count = Counter(gram for gram in tri_tag_tuples)
    tri_total = sum(tri_count.values())

    for k,b,c in tri_count.keys():
        q_values[k,b,c] = np.log2(tri_count[k,b,c]/tri_dict[k,b])
    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    trigrams.sort()
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    brown_words2 =[]
    for a in brown_words:
        brown_words2.extend(a)

    word_counts = Counter(word for word in brown_words2)
    known_words = set([])
    for a in word_counts:
        if word_counts[a]>5:
            known_words.add(a)
    return known_words

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = []
    trare =[]
    for a in range(len(brown_words)):
        for k in brown_words[a]:
            if k not in known_words:
                trare.append(RARE_SYMBOL)
            else:
                trare.append(k)
        brown_words_rare.append(trare)
        trare =[]
    return brown_words_rare

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    brown_words_rare2 =[]
    brown_words_rare3 =[]
    for a in brown_words_rare:
        brown_words_rare2.extend(a)
    brown_tags2 =[]
    for a in brown_tags:
        brown_tags2.extend(a)

    for a in range(len(brown_words_rare2)):
        brown_words_rare3.append((brown_words_rare2[a],brown_tags2[a]))

    e_values = {}
    taglist = set([])
    brown_probs = Counter()
    brown_tags_count = Counter(tag for tag in brown_tags2)
    brown_words_rare_count = Counter(word for word in brown_words_rare3)
    for a in range(len(brown_words_rare3)):
        e_values[(brown_words_rare2[a],brown_tags2[a])]= np.log2(brown_words_rare_count[brown_words_rare3[a]]/brown_tags_count[brown_tags2[a]])
        taglist.add(brown_tags2[a])
    return e_values, taglist

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()

def forward(brown_dev_words,taglist, known_words, q_values, e_values):
    brown_dev_words2 = [s + [STOP_SYMBOL] for s in brown_dev_words]
    #brown_dev_words2 = brown_dev_words2[0:5]
    probs = []
    for s in brown_dev_words2:
        total = 0
        probsm = Counter()
        for w in range(len(s)):
            word = s[w]
            if word not in known_words:
                word = '_RARE_'
            for t in taglist:
                if w == 0:
                    if ('*','*',t) not in q_values:
                        tr_p = LOG_PROB_OF_ZERO
                    else:
                        tr_p = q_values['*','*',t]
                    if (word,t) not in e_values:
                        ob_p = LOG_PROB_OF_ZERO
                    else:
                        ob_p = e_values[word,t]
                    probsm[w,'*',t]=tr_p+ob_p
                else:
                    for t2 in taglist:
                        if w == 1:
                            if ('*',t2,t) not in q_values:
                                tr_p1 = LOG_PROB_OF_ZERO
                            else:
                                tr_p1 = q_values['*',t2,t]
                            if (word,t) not in e_values:
                                ob_p1 = LOG_PROB_OF_ZERO
                            else:
                                ob_p1 = e_values[word,t]
                            probsm[w,t2,t]=tr_p1+ob_p1+probsm[w-1,'*',t2]
                        else:
                            if (word,t) not in e_values:
                                probsm[w,t2,t] = LOG_PROB_OF_ZERO
                                continue
                            probs_res = 0
                            probs_sum = 0
                            if (word,t) not in e_values:
                                ob_p2 = LOG_PROB_OF_ZERO
                            else:
                                ob_p2 = e_values[word,t]
                            for t3 in taglist:
                                if (t3,t2,t) not in q_values:
                                    tr_p2 = LOG_PROB_OF_ZERO
                                else:
                                    tr_p2 = q_values[t3,t2,t]
                                probs_sum = ob_p2 + probsm[w-1,t3,t2] + tr_p2

                                probs_res = probs_res + 2**probs_sum
                            if probs_res == 0.0:
                                probs_res = LOG_PROB_OF_ZERO
                            else:
                                probs_res = np.log2(probs_res)
                            probsm[w,t2,t]=probs_res
                            if w==len(s)-1:
                                total = total + 2**probs_res
        probs.append((np.log2(total).astype('str')+' \n'))

    return probs
# This function takes the output of forward() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!
def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    brown_dev_words2 = [s + [STOP_SYMBOL] for s in brown_dev_words]
    #brown_dev_words2 = brown_dev_words2[0:5]
    #create a probability matrix
    probs = []

    tagged = []
    for s in brown_dev_words2:
        fintags = Counter()
        total = 0
        probsm = Counter()
        for w in range(len(s)):
            fintags[w] = [-1000,'WORD','TAG']
            wordmax = 0
            word = s[w]
            if word not in known_words:
                word = '_RARE_'
            for t in taglist:
                if w == 0:
                    if ('*','*',t) not in q_values:
                        tr_p = LOG_PROB_OF_ZERO
                    else:
                        tr_p = q_values['*','*',t]
                    if (word,t) not in e_values:
                        ob_p = LOG_PROB_OF_ZERO
                    else:
                        ob_p = e_values[word,t]
                    probsm[w,'*',t]=tr_p+ob_p
                    if (probsm[w,'*',t]) > fintags[w][0]:
                        fintags[w] = [probsm[w,'*',t], s[w],t]
                else:
                    for t2 in taglist:
                        if w == 1:
                            if ('*',t2,t) not in q_values:
                                tr_p1 = LOG_PROB_OF_ZERO
                            else:
                                tr_p1 = q_values['*',t2,t]
                            if (word,t) not in e_values:
                                ob_p1 = LOG_PROB_OF_ZERO
                            else:
                                ob_p1 = e_values[word,t]
                            probsm[w,t2,t]=tr_p1+ob_p1+probsm[w-1,'*',t2]
                            if (probsm[w,t2,t]) > fintags[w][0]:
                                fintags[w] = [probsm[w,t2,t],s[w], t]
                        else:
                            if (word,t) not in e_values:
                                probsm[w,t2,t] = LOG_PROB_OF_ZERO
                                continue
                            probs_res = 0
                            probs_sum = 0
                            if (word,t) not in e_values:
                                ob_p2 = LOG_PROB_OF_ZERO
                            else:
                                ob_p2 = e_values[word,t]
                            for t3 in taglist:
                                if (t3,t2,t) not in q_values:
                                    tr_p2 = LOG_PROB_OF_ZERO
                                else:
                                    tr_p2 = q_values[t3,t2,t]
                                probs_sum = ob_p2 + probsm[w-1,t3,t2] + tr_p2
                                probs_res = probs_res + 2**probs_sum
                            if probs_res == 0.0:
                                probs_res = LOG_PROB_OF_ZERO
                            else:
                                probs_res = np.log2(probs_res)
                            probsm[w,t2,t]=probs_res
                            if (probsm[w,t2,t]) > fintags[w][0]:
                                fintags[w] = [probsm[w,t2,t], s[w],t]
                            if w==len(s)-1:
                                total = total + 2**probs_res
        probs.append((np.log2(total).astype('str')+' \n'))


        sent =""
        for tag in range(len(fintags)-1):
            sent = sent+(fintags[tag][1]+ '/'+fintags[tag][2]+' ')
        sent = sent+' \n'
        tagged.append(sent)
    return tagged

# This function takes the output of viterbi() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a
# terminal newline, not a list of tokens.
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
    training = [ zip(brown_words[i],brown_tags[i]) for i in xrange(len(brown_words)) ]
    #brown_dev_words3 = brown_dev_words[0:4]
    # IMPLEMENT THE REST OF THE FUNCTION HERE
    default_tagger = nltk.DefaultTagger('NOUN')
    bigram_tagger = nltk.BigramTagger(training, backoff=default_tagger)
    trigram_tagger = nltk.TrigramTagger(training, backoff=bigram_tagger)
    #tokens = nltk.word_tokenize(brown_dev_words)
    tagged = []
    interim = []
    interim2 = ""
    for sentence in brown_dev_words:
        interim = trigram_tagger.tag(sentence)
        for a in interim:
            interim2 = interim2+a[0]+'/'+a[1]+' '
        interim2 = interim2+'\n'
        tagged.append(interim2)
        interim = []
        interim2 = ""
    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q7_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare



    # open Brown development data (question 6)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # question 5
    forward_probs = forward(brown_dev_words,taglist, known_words, q_values, e_values)
    q5_output(forward_probs, OUTPUT_PATH + 'B5.txt')

    # do viterbi on brown_dev_words (question 6)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 6 output
    q6_output(viterbi_tagged, OUTPUT_PATH + 'B6.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 7 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B7.txt')

    # print total time to run Part B
    print "Part B time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()

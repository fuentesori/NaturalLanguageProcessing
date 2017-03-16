import gensim
from collections import OrderedDict
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic

RESOURCES = ""
RESOURCES = '/home/coms4705/resources/'

# Given a filename where each line is in the format "<word1>  <word2>  <human_score>", 
# return a dictionary of {(word1, word2):human_score), ...}
# Note that human_scores in your dictionary should be floats.
def parseFile(filename):
    similarities = OrderedDict()
    # Fill in your code here
    f = open(filename,'r')
    for line in f:
         words = line.split()
         similarities[(words[0],words[1])]=float(words[2])
    ########################
    return similarities
    


# Given a list of tuples [(word1, word2), ...] and a wordnet_ic corpus, return a dictionary 
# of Lin similarity scores {(word1, word2):similarity_score, ...}
def linSimilarities(word_pairs, ic):
    similarities = OrderedDict()
    # Fill in your code here
    for a in word_pairs:
	 i = wn.synsets(a[0].lower(), pos=wn.NOUN)
	 j = wn.synsets(a[1].lower(), pos=wn.NOUN)
	 if (i ==[]) or (j==[]):
	     i = wn.synsets(a[0].lower(), pos=wn.VERB)
	     j = wn.synsets(a[1].lower(), pos=wn.VERB) 
         similarities[(a[0],a[1])]=(float(i[0].lin_similarity(j[0],ic))*10)
    ########################
    return similarities

# Given a list of tuples [(word1, word2), ...] and a wordnet_ic corpus, return a dictionary 
# of Resnik  similarity scores {(word1, word2):similarity_score, ...}
def resSimilarities(word_pairs, ic):
    similarities = {}
    # Fill in your code here
    for a in word_pairs:
	 i = wn.synsets(a[0].lower(), pos=wn.NOUN)
	 j = wn.synsets(a[1].lower(), pos=wn.NOUN)
	 if (i==[]) or (j==[]):
	      i = wn.synsets(a[0].lower(), pos=wn.VERB)
	      j = wn.synsets(a[1].lower(), pos=wn.VERB)
	 similarities[(a[0],a[1])]=float(i[0].res_similarity(j[0],ic))
    ########################
    return similarities

# Given a list of tuples [(word1, word2), ...] and a word2vec model, return a dictionary 
# of word2vec similarity scores {(word1, word2):similarity_score, ...}
def vecSimilarities(word_pairs, model):
    similarities = {}
    # Fill in your code here
    for a in word_pairs:
	 similarities[(a[0],a[1])]=float(model.similarity(a[0].lower(),a[1].lower())*10)
    ########################
    return similarities


def main():
    brown_ic = wordnet_ic.ic('ic-brown.dat')

    human_sims = parseFile("input.txt")

    lin_sims = linSimilarities(human_sims.keys(), brown_ic)
    res_sims = resSimilarities(human_sims.keys(), brown_ic)

    model = None
    model = gensim.models.Word2Vec()
    model = model.load_word2vec_format(RESOURCES+'glove_model.txt', binary=False)
    vec_sims = vecSimilarities(human_sims.keys(), model)
    
    lin_score = 0
    res_score = 0
    vec_score = 0

    print '{0:15} {1:15} {2:10} {3:20} {4:20} {5:20}'.format('word1','word2', 
                                                             'human', 'Lin', 
                                                             'Resnik', 'Word2Vec')
    for key, human in human_sims.items():
        try:
            lin = lin_sims[key]
        except:
            lin = 0
        lin_score += (lin - human) ** 2
        try:
            res = res_sims[key]
        except:
            res = 0
        res_score += (res - human) ** 2
        try:
            vec = vec_sims[key]
        except:
            vec = 0
        vec_score += (vec - human) ** 2
        print '{0:15} {1:15} {2:10} {3:20} {4:20} {5:20}'.format(key[0], key[1], human, 
                                                                 lin, res, vec)

    num_examples = len(human_sims)
    print "\nMean Squared Errors"
    print "Lin method error: %0.2f" % (lin_score/num_examples) 
    print "Resnick method error: %0.2f" % (res_score/num_examples)
    print "Vector-based method error: %0.2f" % (vec_score/num_examples)

if __name__ == "__main__":
    main()

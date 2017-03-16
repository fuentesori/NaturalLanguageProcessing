import nltk
import sys

squirrel = 0
girl = 0

greeting = sys.stdin.read()
print greeting

token_list = nltk.word_tokenize(greeting)
print "The tokens in the greeting are"
for token in token_list:
    print token
    token=token.lower()
    if token =='squirrel':
         squirrel += 1
    if token == 'girl':
         girl += 1
print "There were %d instances of the word 'squirrel' and %d instances of the word 'girl.'" % (squirrel, girl)

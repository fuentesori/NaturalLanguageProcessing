import sys
import fileinput
from providedcode.transitionparser import TransitionParser
from providedcode import dataset
from providedcode.dependencygraph import DependencyGraph

englishfile = sys.stdin.read()
lines = englishfile.split('\n')

model =  sys.argv[1]

tp = TransitionParser.load(model)


for line in lines:
    sentence = DependencyGraph.from_sentence(line.strip())
    parsed = tp.parse([sentence])
    print parsed[0].to_conll(10).encode('utf-8'), ('\n')

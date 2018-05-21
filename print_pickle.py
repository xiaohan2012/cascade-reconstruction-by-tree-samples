import sys
import pickle as pkl

path = sys.argv[1]
print(pkl.load(open(path, 'rb')))

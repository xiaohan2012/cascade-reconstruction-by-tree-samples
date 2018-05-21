import sys
import pandas as pd
import os
import pickle as pkl
from glob import glob
from collections import defaultdict


dirname = sys.argv[1]

running_time_by_method = defaultdict(list)

for subdir in glob(os.path.join(dirname, '*')):
    method_name = os.path.basename(subdir)
    for path in glob(os.path.join(subdir, '*.meta.pkl')):
        data = pkl.load(open(path, 'rb'))
        running_time_by_method[method_name].append(data['time_elapsed'])

df = pd.DataFrame.from_dict(running_time_by_method)

print('running time summary')
print('--' * 10)
print(df.describe())


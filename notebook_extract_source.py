import json

data = json.loads(
    open('compare_inference_method.ipynb', 'r').read())

for c in data['cells']:
    line = ''.join(c['source'])
    print(line)

import json
import os
import numpy as np

os.chdir('/home/irr_jinwook/EAI/EAI_midterm/')

with open('result_mp.json', 'r') as f:
    jsonDict = json.load(f)

result = jsonDict[-1]
keys = list(result.keys())
keys.pop(0)
items = {}

for i in jsonDict:
    for key in keys:
        i[key] = i[key][:-4]

jsonString = json.dumps(jsonDict, indent=4)
with open('result_mp.json', 'w') as f:
    f.write(jsonString)
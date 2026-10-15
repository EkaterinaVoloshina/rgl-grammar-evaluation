import json 
from glob import glob

def open_json(file):
    with open(file) as f:
        t = json.load(f)
    return t 

ts = []
filepath = "eval_results_2/results_Mistral-Small-24B-Instruct-2501"
for file in glob(f"{filepath}_*.json"):
    ts.extend(open_json(file))

print(len(ts))

with open(filepath+".json", "w") as f:
    json.dump(ts, f)
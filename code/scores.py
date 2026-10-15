import json 
import pandas as pd
from krippendorff import alpha
from langid.langid import LanguageIdentifier, model, set_languages
from qq import Database, IdType
import evaluate
from Levenshtein import ratio
from tqdm import tqdm
from sys import argv

tqdm.pandas()

chrf = evaluate.load("chrf")

METRICS = ["chrf", "levenshtein", "exact"]
SYSTEMS = {"qwen": "_qwen", "mistral": "_mistral", "gemma": ""}


db = Database.load()
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
identifier.set_languages(['de', 'en', 'fo', 'mk', 'hu', 'te', 'ru', 'mt', 'fa', 'ar', 'ca'])

def check_lang(sent, lang, src):
    if lang == "som":
        return sent 
    if sent.lower().strip() == src: 
        return ""
    langCode = identifier.classify(sent)[0]
    langName = db.get(langCode, id_type=IdType.ISO_639_1)
    if langName.iso_639_2b == "eng": 
        return ""
    return sent 

def open_file(json_file):
    with open(json_file) as f: 
        text = json.load(f)
    df = pd.DataFrame(text)
    df["corrected"] = df.apply(lambda x: check_lang(x["corrected"][0].split("</corrected>")[0],x["lang"], x["src"]) if x["corrected"] and x["corrected"] != "" else None, axis=1)
    df["exact"] = df.progress_apply(lambda x: exact(x["mt"], x["corrected"]) if x["corrected"] else None, axis=1)
    df["chrf"] = df.progress_apply(lambda x: chrf.compute(predictions=[x["mt"]], references=[[x["corrected"]]])["score"] if x["corrected"] else None, axis=1)
    df["levenshtein"] = df.progress_apply(lambda x: ratio(x["mt"], x["corrected"]) if x["corrected"] else None, axis=1)
    return df

def exact(mt, ref):
    return int(mt.lower() == ref.lower())

def get_results(lang_code):
    gemma = open_file(f"../raw_data/eval_results_{lang_code}/results_gemma-3-27B-it_0_11000.json")
    mistral = open_file(f"../raw_data/eval_results_{lang_code}/results_Mistral-Small-24B-Instruct-2501_0_11000.json")
    qwen = open_file(f"../raw_data/eval_results_{lang_code}/results_Qwen3-30B-A3B-Instruct-2507.json")

    df = qwen.merge(mistral, on=["src", "mt", "lang"], suffixes=("_qwen", "_mistral"))
    models = df.merge(gemma, on=["src", "mt", "lang"])

    models = models.replace('', None)
    models["chrf_avg"] = models[["chrf_qwen", "chrf_mistral", "chrf"]].mean(axis=1)
    models["chrf_std"] = models[["chrf_qwen", "chrf_mistral", "chrf"]].std(axis=1)
    models["leven_avg"] = models[["levenshtein_qwen", "levenshtein_mistral", "levenshtein"]].mean(axis=1)
    models["leven_std"] = models[["levenshtein_qwen", "levenshtein_mistral", "levenshtein"]].std(axis=1)
    models.to_csv(f"models_{lang_code}.csv")

def get_scores(lang_code):
    models = pd.read_csv(f"../corrections/models_{lang_code}.csv")
    data = models[["exact_qwen", "exact_mistral", "exact"]].values.T
    print("Alpha: ", round(alpha(reliability_data=data, level_of_measurement="nominal"), 3))
    
    long = pd.concat([
        models[["lang"] + [m + suf for m in METRICS]]
        .rename(columns={m + suf: m for m in METRICS})
        .assign(system=name)
        for name, suf in SYSTEMS.items()
    ])

    by_lang = long.groupby(["lang", "system"])[METRICS].agg(["mean", "count"])
    by_lang.to_csv(f"scores_{lang_code}.csv")


if __name__=="__main__":
    lang_code = argv[1]
    print(lang_code)
    get_results(lang_code)
    get_scores(lang_code)








import json
import os 
import re 
import argparse
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def mapLang(lang):
	LANGS = {
		"hun": "Hungarian", 
		"mkd": "Macedonian",
		"som": "Somali",
		"ger": "German",
		"fao": "Faroese"
			 }
	return LANGS[lang]

def get_prompt(sentence, translation, lang):
	
	PROMPT = f"""Please make minimal modifications to the given sentence in {mapLang(lang)} to achieve all of the properties below:
    - Perfect grammaticality: The sentence is native-sounding. It has no grammatical errors, but may contain very minor typographical and/or collocation errors.
    - Perfect fluency: The sentence sounds extremely natural and native-like.
    - Same language: The sentence must remain in the same language as the original (do not translate or change language).
    Instructions:
    1. Return ONLY the corrected sentence.
    2. Wrap the corrected sentence in <corrected> and </corrected> tags.
	3. Ensure that the corrected sentence preserves the meaning. For the reference, you are provided with English translation below. 
    4. If the original sentence is already perfect, return it AS IS inside the <corrected> tags.
    5. Do NOT include any explanations, extra text, or formatting.
    Example output format:
    <corrected>Your corrected sentence here.</corrected>
    Sentence: {sentence}
	Translation (in English): {translation}
    Output:"""
	return PROMPT

def generate_batch(tokenizer, model, prompts):
	input_ids = tokenizer(prompts, add_special_tokens=False, return_tensors="pt", padding=True)["input_ids"]
	input_ids = input_ids.to(model.device)
	output = model.generate(
            input_ids,
            max_new_tokens=100,
            repetition_penalty=1.1,
            num_return_sequences=1,
        )
	output = output[:, input_ids.shape[1]:]
	generation = tokenizer.batch_decode(output.cpu(), skip_special_tokens=True)
	return generation

def format_results(examples, results):
	updExamples = []
	for n, ex in enumerate(examples):
		ex["corrected"] = re.findall("<corrected>(.*)</corrected>", results[n], re.DOTALL)
		updExamples.append(ex)
	return updExamples 

def run_model(model_name, examples, batch_size):
	if "Mistral" in model_name:
		tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", fix_mistral_regex=True)
	else:
		tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
	model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
	
	if "Mistral" in model_name or "Llama" in model_name:
		tokenizer.pad_token = tokenizer.eos_token
		model.generation_config.pad_token_id = model.generation_config.eos_token_id
	model.eval()
	all_generations = []
	for i in tqdm(range(0, len(examples), batch_size)):
		batch = examples[i:i + batch_size]
		prompts = [get_prompt(sentence["mt"], sentence["src"], sentence["lang"]) for sentence in batch]
		results = generate_batch(tokenizer, model, prompts)
		generations = format_results(batch, results)
		all_generations.extend(generations)
	return all_generations

def load_dataset(filename):
	with open(filename) as f:
		examples = json.load(f)
	return examples[:1000]

def save_dataset(output, model, result):
	model_name = model.split("/")[-1]
	with open(os.path.join(output,f"results_{model_name}.json"), "w") as f:
		json.dump(result, f)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("model")
	parser.add_argument("--input", default="data/results.json") #json file with examples
	parser.add_argument("--output", default="eval_results")
	parser.add_argument("--batch_size", default=8)
	args = parser.parse_args()
	print(f"Reading {args.input}..")
	examples = load_dataset(args.input)
	print("Generating..")
	result = run_model(args.model, examples, int(args.batch_size))
	print(f"Saving to {args.output}")
	save_dataset(args.output, args.model, result)
	
if __name__=="__main__":
	main()

    
    
    
		
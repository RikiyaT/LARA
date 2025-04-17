import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import csv
from tqdm import tqdm
import re
import gc
import xml.etree.ElementTree as ET
from datasets import load_dataset
import time
from utils.utils import prompt_func
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"  # Changed to generic GPU IDs


def get_specified_gpus():
    gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpu_ids:
        return [int(gpu) for gpu in gpu_ids.split(',')]
    return []

def load_model_and_tokenizer():
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    cache_dir = "/path/to/model/cache"  # Changed to generic path
    # make cache_dir if it does not exist
    os.makedirs(cache_dir, exist_ok=True)

    token = "YOUR_HUGGINGFACE_TOKEN"  # Changed to placeholder token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        cache_dir=cache_dir,
        token=token,
        torch_dtype=torch.bfloat16 
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=token)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded. Device map: {model.hf_device_map}")
    return model, tokenizer

def load_qrels(qrels_path):
    qrels = {}
    with open(qrels_path, 'r') as f:
        for line in f:
            query_id, _, doc_id, relevance = line.strip().split()
            qrels[(query_id, doc_id)] = int(relevance)
    return qrels

def load_topics(topics_path):
    tree = ET.parse(topics_path)
    root = tree.getroot()
    topics = {}
    for topic in root.findall('topic'):
        number = topic.get('number')
        query = topic.find('query').text
        question = topic.find('question').text
        narrative = topic.find('narrative').text
        topics[number] = {
            'query': query,
            'question': question,
            'narrative': narrative
        }
    return topics

def load_corpus():
    os.environ['HF_DATASETS_CACHE'] = "/path/to/dataset/cache"  # Changed to generic path
    dataset = load_dataset("BeIR/trec-covid", "corpus")
    return dataset['corpus']

def get_document(doc_id, corpus):
    doc_ids = corpus['_id']
    if doc_id not in doc_ids:
        print(f"Document not found: {doc_id}")
        return None
    
    index = doc_ids.index(doc_id)
    title = corpus['title'][index]
    text = corpus['text'][index]
    
    formatted_content = f"Title: {title}\n\n{text}"
    return formatted_content

def truncate_text(text, max_tokens, tokenizer):
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if len(encoded) <= max_tokens:
        return text
    
    truncated = tokenizer.decode(encoded[:max_tokens], skip_special_tokens=True)
    last_sentence = re.findall(r'.*?[.!?]', truncated[::-1])
    if last_sentence:
        last_complete_sentence = last_sentence[0][::-1]
        truncated = truncated[:truncated.rfind(last_complete_sentence) + len(last_complete_sentence)]
    
    return truncated.strip()

def preprocess_document(content, max_tokens, tokenizer):
    return truncate_text(content, max_tokens, tokenizer)

def format_prompt(dataset, text, prompt_type, query, desc, narrative):
    text = repr(text)[1:-1]
    query = repr(query)[1:-1]
    desc = repr(desc)[1:-1]
    return prompt_func(data = dataset, text=text, prompt_type=prompt_type, query=query, desc=desc, narrative=narrative)

def calculate_probabilities(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,
            top_k=50,
            temperature=3.0
        )
    
    scores = outputs.scores[0][0]
    probabilities = torch.nn.functional.softmax(scores, dim=0)
    
    def get_word_probability(word):
        variations = [f" {word}", f" {word.lower()}", f" {word.upper()}", word, word.lower(), word.upper()]
        token_ids = []
        for var in variations:
            token_ids.extend(tokenizer.encode(var, add_special_tokens=False))
        return max(probabilities[token_id].item() for token_id in set(token_ids) if token_id in range(len(probabilities)))

    prob_0 = get_word_probability("0")
    prob_1 = get_word_probability("1")
    prob_2 = get_word_probability("2")
    
    total_prob = prob_0 + prob_1 + prob_2
    normalized_prob_0 = prob_0 / total_prob if total_prob > 0 else 0.0
    normalized_prob_1 = prob_1 / total_prob if total_prob > 0 else 0.0
    normalized_prob_2 = prob_2 / total_prob if total_prob > 0 else 0.0

    return normalized_prob_0, normalized_prob_1, normalized_prob_2


def evaluate_relevance(model, tokenizer, topics, qrels, corpus, output_file, prompt_type, dataset, batch_size=100):
    max_prompt_tokens = 3000 
    static_prompt_part = format_prompt(dataset = dataset, text="", prompt_type=prompt_type, query="", desc="", narrative="")
    max_doc_tokens = max_prompt_tokens - len(tokenizer.encode(static_prompt_part))

    file_exists = os.path.isfile(output_file)
    mode = 'a' if file_exists else 'w'
    
    with open(output_file, mode, newline='') as csvfile:
        fieldnames = ['topic_id', 'doc_id', 'annotation', 'prob_0', 'prob_1', 'prob_2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()

        for batch_start in tqdm(range(0, len(qrels), batch_size), desc="Processing batches"):
            batch_items = list(qrels.items())[batch_start:batch_start + batch_size]
            
            for (query_id, doc_id), relevance in batch_items:

                if query_id not in topics:
                    print(f"Query {query_id} not found.")
                    continue

                topic = topics[query_id]
                query = topic['query']
                desc = topic['question']
                narrative = topic['narrative']

                doc_content = get_document(doc_id, corpus)
                
                if doc_content is None:
                    continue

                doc_content = preprocess_document(doc_content, max_doc_tokens, tokenizer)

                prompt = format_prompt(dataset=dataset, text=doc_content, prompt_type=prompt_type, query=query, desc=desc, narrative=narrative)
                
                prob_0, prob_1, prob_2 = calculate_probabilities(model, tokenizer, prompt)

                writer.writerow({
                    'topic_id': query_id,
                    'doc_id': doc_id,
                    'annotation': relevance,
                    'prob_0': f"{prob_0:.4f}",
                    'prob_1': f"{prob_1:.4f}",
                    'prob_2': f"{prob_2:.4f}"
                })
                
                if csvfile.tell() % 10 == 0:
                    csvfile.flush()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Evaluate relevance using a specified prompt type.")
    parser.add_argument("--prompt_type", type=str, choices=["utility", "rational", "simple"], required=True, 
                        help="Specify the type of prompt to use: 'utility', 'rational', or 'simple'.")
    args = parser.parse_args()
    
    dataset = "gcovid5"
    
    prompt_type = args.prompt_type
    print(f"Selected prompt type: {prompt_type}")
    
    model, tokenizer = load_model_and_tokenizer()
    
    qrels_path = f"../results/{dataset}/qrels.true"
    topics_path = f"../data/{dataset}/topics.gcovid5"
    
    qrels = load_qrels(qrels_path)
    topics = load_topics(topics_path)
    
    print("Loading BEIR/trec-covid corpus...")
    corpus = load_corpus()
    
    output_file = f"../results/{dataset}/results/{prompt_type}-Llama-3.1-70B-Instruct.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Evaluating relevance and writing results to {output_file}...")
    
    evaluate_relevance(model, tokenizer, topics, qrels, corpus, output_file, prompt_type, dataset)
    
    print("Evaluation complete. Results have been written to the output file.")

if __name__ == "__main__":
    main()
import json
import torch
import re
import argparse
import os
import time
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# =======================================================
# 1. Experiment Matrix & Configuration
# =======================================================
CONFIG = {
    "LOCAL_PATH": {
        "citeagent": "./path/to/CiteLocator",
        "qwen4b": "../path/to/model_weights/qwen/Qwen3-4B-Instruct-2507"
    },
    "API_KEYS": {
        "deepseek": "YOUR_DEEPSEEK_API_KEY",
        "qwen_api": "YOUR_QWEN_API_KEY",
        "openai": "YOUR_OPENAI_API_KEY",
        "zhipu": "YOUR_ZHIPU_API_KEY",
        "moonshot": "YOUR_MOONSHOT_API_KEY",
        "minimax": "YOUR_MINIMAX_API_KEY"
    },
    "BASE_URLS": {
        "deepseek": "url",
        "qwen_api": "url",
        "openai": "url",
        "zhipu": "url",
        "moonshot": "url",
        "minimax": "url"
    }
}

MODEL_MAP = {
    "gpt-mini": ("gpt-4o-mini", "openai"),
    "kimi": ("kimi-k2-0905-preview", "moonshot"),
    "minimax": ("MiniMax-M2.5", "minimax"),
    "zhipu": ("glm-4-flash", "zhipu"),
    "qwen-27b": ("qwen3.5-27b", "qwen_api")
}

CITE_TOKEN = "[#CITE#]" 
MAX_WORKERS = 15        

# =======================================================
# 2. Core Algorithmic Logic
# =======================================================

def get_citation_anchors(text, token, window_size=5):
    """Extracts a character window preceding each citation token for positional matching."""
    anchors = []
    # Core fix: Use '*' to aggressively consume all preceding spaces and tabs before the token.
    pattern = r'(?:~|\s)*' + re.escape(token) 
    for m in re.finditer(pattern, text):
        # Clean residual trailing whitespace from the extracted prefix
        prefix = text[:m.start()].replace(token, "").replace("~", "").rstrip()
        anchor = prefix[-window_size:].strip()
        if anchor:
            clean_anchor = re.sub(r'\s+', ' ', anchor.lower()).strip()
            anchors.append(clean_anchor)
    return anchors

def evaluate_categorical(gt_text, pred_text, gt_types, token):
    """Evaluates prediction accuracy separated by mandatory vs. optional citation requirements."""
    gt_anchors = get_citation_anchors(gt_text, token)
    pred_anchors = get_citation_anchors(pred_text, token)
    res = {"mandatory": {"tp": 0, "fn": 0}, "optional": {"tp": 0, "fn": 0}, "overall": {"tp": 0, "fp": 0, "fn": 0}}
    temp_pred_set = list(pred_anchors)
    
    for i, anchor in enumerate(gt_anchors):
        raw_val = gt_types[i] if i < len(gt_types) else "mandatory"
        if isinstance(raw_val, dict):
            c_type = str(raw_val.get("classification", raw_val.get("type", list(raw_val.values())[0]))).lower()
        else:
            c_type = str(raw_val).lower()
            
        if c_type not in ["mandatory", "optional"]: 
            c_type = "mandatory"
        
        if anchor in temp_pred_set:
            res["overall"]["tp"] += 1
            res[c_type]["tp"] += 1
            temp_pred_set.remove(anchor)
        else:
            res["overall"]["fn"] += 1
            res[c_type]["fn"] += 1
            
    res["overall"]["fp"] = len(temp_pred_set)
    return res


# =======================================================
# 3. Inference Engines
# =======================================================

def get_prompt(strategy): 
    if strategy == "direct":
        return (f"Task: Act as a professional academic copy-editor. Insert the specified marker {CITE_TOKEN} "
                f"at places that require citations, such as when mentioning prior works, existing methods, "
                f"algorithms, models, datasets or other places where citations are needed. "
                f"Rules: Do not alter any original content. Do not add extra preamble. Output only the processed text.")
    
    elif strategy == "cot":
        return (f"""[Role] You are an Expert Academic Copy-Editor and Peer Reviewer.

[Task] Audit the provided academic text and systematically insert the exact marker {CITE_TOKEN} at places that require citations.

[Decision Logic - Internal Chain of Thought]
Step 1. Claim Identification: Read the text sentence by sentence to identify specific technical assertions, such as mentions of prior works, existing methods, algorithms, models, or datasets.
Step 2. Necessity Evaluation: For each identified claim, determine if it relies on external literature or established knowledge that logically requires an academic citation for attribution.
Step 3. Precise Placement: Once a citation is deemed necessary, insert the {CITE_TOKEN} marker exactly at the end of that specific claim, without breaking the original sentence structure.

[Constraints]
- Output ONLY the final processed text with the {CITE_TOKEN} markers inserted.
- DO NOT alter, add, or delete any original words, spaces, or punctuation.
- DO NOT output any internal reasoning, explanations, or preamble.""")
    
    return ""


def local_inference(model_key, test_data, strategy, batch_size=32, device="cuda:1"):
    path = CONFIG["LOCAL_PATH"][model_key]
    print(f"[*] Loading batch inference model: {path} | Batch Size: {batch_size}")
    
    # Must load the vocabulary associated with the fine-tuned model
    tokenizer_path = path
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, padding_side='left')
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    
    # Ensure special tokens are ONLY added to the fine-tuned citeagent, NOT the base model
    if model_key == "citeagent":
        if CITE_TOKEN not in tokenizer.all_special_tokens:
            tokenizer.add_special_tokens({'additional_special_tokens': [CITE_TOKEN]})
        
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    
    prompt_template = get_prompt(strategy)

    results = []
    for i in tqdm(range(0, len(test_data), batch_size), desc=f"Local-Batch-{model_key}"):
        batch_items = test_data[i : i + batch_size]
        batch_prompts = [f"<|im_start|>system\n{prompt_template}<|im_end|>\n<|im_start|>user\n{item['input']}<|im_end|>\n<|im_start|>assistant\n" for item in batch_items]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=False, 
                pad_token_id=tokenizer.pad_token_id
            )
            
        for j, item in enumerate(batch_items):
            generated_ids = outputs[j][inputs.input_ids.shape[1]:]
            
            # Must be False to prevent the tokenizer from swallowing the [#CITE#] token
            pred = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
            
            # Manually strip system termination tokens
            pred = pred.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

            eval_metrics = evaluate_categorical(item["output"], pred, item.get("citation_types", []), CITE_TOKEN)
            results.append({
                "input": item["input"],
                "ground_truth": item["output"],
                "prediction": pred,
                "eval": eval_metrics
            })
            
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return results

def api_inference(model_name, model_id, api_type, test_data, strategy):
    client = OpenAI(api_key=CONFIG["API_KEYS"][api_type], base_url=CONFIG["BASE_URLS"][api_type])
    prompt_template = get_prompt(strategy)
    
    if model_name == "kimi":
        current_workers = 3
    elif model_name == "qwen-27b":
        current_workers = 15
    else:
        current_workers = MAX_WORKERS
    
    def process_item(index, item):
        for attempt in range(5):
            try:
                params = {
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": prompt_template},
                        {"role": "user", "content": item["input"]}
                    ],
                    "temperature": 0.1,
                    "timeout": 120
                }

                if "qwen" in model_name:
                    params["extra_body"] = {"enable_thinking": False}

                resp = client.chat.completions.create(**params)
            
                pred = resp.choices[0].message.content.strip()
                
                # Specific thought-block silencer for Minimax models
                if model_name == "minimax":
                    # Use re.DOTALL to ensure the regex matches multiline <think> blocks
                    pred = re.sub(r'<think>.*?</think>\s*', '', pred, flags=re.DOTALL)
                    
                eval_metrics = evaluate_categorical(item["output"], pred, item.get("citation_types", []), CITE_TOKEN)
                return index, {
                    "input": item["input"],
                    "ground_truth": item["output"],
                    "prediction": pred,
                    "eval": eval_metrics
                }
            except Exception as e:
                print(f"\n[DEBUG] {model_name} Error details: {e}") 
                time.sleep((attempt + 1) * 10) 
        return index, None

    results_map = {}
    print(f"[*] Starting parallel API evaluation: {model_name} (Concurrency: {current_workers})...")
    
    with ThreadPoolExecutor(max_workers=current_workers) as executor:
        futures = {executor.submit(process_item, i, item): i for i, item in enumerate(test_data)}
        for future in tqdm(as_completed(futures), total=len(test_data), desc=f"API-{model_name}"):
            idx, res = future.result()
            results_map[idx] = res
            
    return [results_map.get(i) for i in range(len(test_data))]

# =======================================================
# 4. Execution Entry Point
# =======================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CiteAgent Evaluation Suite")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["citeagent", "qwen4b", "ds_v3", "qwen3-max", "gpt5.1-chat", 
                                 "kimi", "minimax", "zhipu", "gpt-mini", "qwen-27b"])
    parser.add_argument("--strategy", type=str, default="direct", choices=["direct", "cot"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    data_path = "test_CiteLocator.json"
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), "..", data_path)

    with open(data_path, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]
    
    if args.limit: 
        test_data = test_data[:args.limit]

    if args.model in ["citeagent", "qwen4b"]:
        raw_res = local_inference(args.model, test_data, args.strategy)
    elif args.model == "ds_v3":
        raw_res = api_inference("DeepSeek-V3", "deepseek-chat", "deepseek", test_data, args.strategy)
    elif args.model == "qwen3-max":
        raw_res = api_inference("Qwen3-Max", "qwen-max", "qwen_api", test_data, args.strategy)
    elif args.model == "gpt5.1-chat":
        raw_res = api_inference("GPT-5.1-Chat", "gpt-5.1-chat", "openai", test_data, args.strategy)
    elif args.model in MODEL_MAP:
        m_id, a_type = MODEL_MAP[args.model]
        raw_res = api_inference(args.model, m_id, a_type, test_data, args.strategy)

    stats = {"mandatory": {"tp": 0, "fn": 0}, "optional": {"tp": 0, "fn": 0}, "overall": {"tp": 0, "fp": 0, "fn": 0}}
    detailed_results = [] 
    
    for r in raw_res:
        if r:
            eval_metrics = r["eval"]
            for cat in stats:
                for k in stats[cat]: 
                    stats[cat][k] += eval_metrics[cat].get(k, 0)
            
            detailed_results.append({
                "input": r["input"],
                "ground_truth": r["ground_truth"],
                "prediction": r["prediction"],
                "metrics": eval_metrics
            })

    final_report = {
        "meta": {
            "model": args.model,
            "strategy": args.strategy,
            "total_samples": len(test_data)
        },
        "metrics": {},      
        "raw_counts": stats,
        "detailed_results": detailed_results 
    }

    print("\n" + "="*65)
    print(f"Evaluation Report: {args.model} | Strategy: {args.strategy}")
    print("-" * 65)
    
    for cat in ["mandatory", "optional", "overall"]:
        tp = stats[cat]["tp"]
        fp = stats["overall"]["fp"]
        fn = stats[cat]["fn"]
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        final_report["metrics"][cat] = {
            "precision": f"{p:.2%}",
            "recall": f"{r:.2%}",
            "f1_score": f"{f1:.2%}"
        }
        
        print(f"{cat.upper():<15} | Prec: {p:>7.2%} | Rec: {r:>7.2%} | F1: {f1:>7.2%}")
    print("="*65)

    output_filename = f"result_{args.model}_{args.strategy}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)
    
    print(f"\n[*] Evaluation complete! Results saved to: {output_filename}")

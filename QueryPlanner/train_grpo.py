import re
import json
import torch
import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

# Environment Optimization
os.environ["VLLM_USE_V1"] = "0" 
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# =======================================================
# 1. Paths & Alignment Configuration
# =======================================================
MODEL_PATH = "./path/to/sft_output_model" 
DATA_FILE = "./path/to/train.jsonl"
OUTPUT_DIR = "./path/to/grpo_output"

SYSTEM_PROMPT = (
    "You are an expert academic citation analyst.\n\n"
    "### Task\n"
    "Analyze the context of the given text. For each citation marker [#CITE#], infer what keywords should be used to search for the paper that needs to be cited. Each [#CITE#] marker in the input must correspond to one [Index n] in the output in order, and the total number of indices must exactly match the number of citation markers.\n\n"
    "### Output Components\n"
    "1. <reasoning>: A 1-2 sentence logical analysis of the context explaining the core role of the citation.\n"
    "   MUST start with the fixed phrase: \"The purpose of this citation is to...\"\n"
    "2. <keywords>: 3-5 precise keywords to retrieve the target document.\n"
    "   - Focus on: contextually relevant keywords, specific research methods, exclusive technical terminology, problem formulation, model or dataset names, etc.\n\n"
    "### Constraints\n"
    "- Sequential Indexing: Start with [Index 0] and increment by 1 for each marker. Do NOT skip any numbers.\n"
    "- Strict Mapping: The total number of [Index n] blocks must exactly match the count of [#CITE#] markers in the input.\n"
    "- Multi-Ref Handling: If one [#CITE#] represents multiple references, provide multiple consecutive <reasoning> and <keywords> pairs under that single [Index n].\n"
    "- Output ONLY the structured content. Strictly NO conversational filler.\n\n"
    "### Output Format\n"
    "[Index X]\n"
    "<reasoning>...</reasoning>\n"
    "<keywords>[\"keyword 1\", \"keyword 2\", ...]</keywords>"
)

# =======================================================
# 2. Reward Function Collection
# =======================================================

def reasoning_start_reward_func(completions, **kwargs) -> list[float]:
    """[Fixed Prefix Reward] Ensures the model follows the specific phrase constraints defined in the Prompt."""
    fixed_phrase = "The purpose of this citation is to"
    rewards = []
    for content in completions:
        content = str(content)
        blocks = re.findall(r'<reasoning>(.*?)</reasoning>', content, re.DOTALL)
        if not blocks:
            rewards.append(-1.0); continue
        # Penalize if any reasoning block deviates from the required starting phrase.
        score = 1.0 if all(b.strip().startswith(fixed_phrase) for b in blocks) else -1.0
        rewards.append(score)
    return rewards

def hybrid_content_reward_func(prompts, completions, answer, **kwargs):
    rewards = []
    for content, gold_output in zip(completions, answer):
        try:
            p_blocks = re.findall(r'<keywords>(.*?)</keywords>', str(content), re.DOTALL)
            g_blocks = re.findall(r'<keywords>(.*?)</keywords>', str(gold_output), re.DOTALL)
            p_list = []; [p_list.extend(json.loads(b)) for b in p_blocks]
            g_list = []; [g_list.extend(json.loads(b)) for b in g_blocks]
            if not g_list: rewards.append(0.0); continue
            p_set, g_set = set([x.lower() for x in p_list]), set([x.lower() for x in g_list])
            f1_strict = 2 * len(p_set & g_set) / (len(p_set) + len(g_set) + 1e-8) if p_set else 0
            p_ws = set(re.findall(r'\w+', " ".join(p_list).lower()))
            g_ws = set(re.findall(r'\w+', " ".join(g_list).lower()))
            f1_soft = 2 * len(p_ws & g_ws) / (len(p_ws) + len(g_ws) + 1e-8) if p_ws else 0
            rewards.append(f1_strict * 2.5 + f1_soft * 1.0)
        except: rewards.append(0.0)
    return rewards

def local_context_window_reward_func(prompts, completions, **kwargs):
    rewards = []
    STOP_WORDS = {'the', 'and', 'in', 'of', 'with', 'for', 'based', 'using', 'by', 'on', 'from', 'a', 'an'}
    for prompt, content in zip(prompts, completions):
        input_text = prompt[-1]["content"]
        cite_matches = list(re.finditer(r'\[#CITE#\]', input_text))
        pred_kws_blocks = re.findall(r'<keywords>(.*?)</keywords>', str(content), re.DOTALL)
        score = 0
        for i, match in enumerate(cite_matches):
            if i >= len(pred_kws_blocks): break
            words_before = input_text[:match.start()].strip().split()
            window = words_before[-4:] if len(words_before) >= 4 else words_before
            try:
                k_list = [k.lower().strip() for k in json.loads(pred_kws_blocks[i])]
                combined_kws = " ".join(k_list)
                match_count = 0
                for word in window:
                    clean_word = word.strip(".,()[]")
                    lower_word = clean_word.lower()
                    if lower_word in STOP_WORDS or len(lower_word) <= 2: continue
                    if lower_word in combined_kws:
                        weight = 1.0 if clean_word[0].isupper() else 0.5
                        match_count += weight
                score += min(1.5, match_count)
            except: continue
        rewards.append(score / len(cite_matches) if cite_matches else 0.0)
    return rewards

def context_relevance_reward_func(prompts, completions, **kwargs):
    rewards = []
    for prompt, content in zip(prompts, completions):
        source_words = set(re.findall(r'\w+', prompt[-1]["content"].lower()))
        kw_blocks = re.findall(r'<keywords>(.*?)</keywords>', str(content), re.DOTALL)
        all_pred_words = set(re.findall(r'\w+', "".join(kw_blocks).lower()))
        if not all_pred_words: rewards.append(0.0); continue
        hit_rate = sum(1 for w in all_pred_words if w in source_words) / len(all_pred_words)
        rewards.append(1.5 if hit_rate > 0.4 else -1.0)
    return rewards

def count_consistency_reward_func(prompts, completions, **kwargs):
    rewards = []
    for prompt, content in zip(prompts, completions):
        target = prompt[-1]["content"].count("[#CITE#]")
        pred = len(re.findall(r'\[Index \d+\]', str(content)))
        rewards.append(2.0 if pred == target else -2.0)
    return rewards

def format_reward_func(completions, **kwargs):
    rewards = []
    for content in completions:
        if "<reasoning>" in str(content) and "<keywords>" in str(content):
            try:
                kw_blocks = re.findall(r'<keywords>(.*?)</keywords>', str(content), re.DOTALL)
                if kw_blocks and all(isinstance(json.loads(b), list) for b in kw_blocks):
                    rewards.append(1.0); continue
            except: pass
        rewards.append(-1.0)
    return rewards

def index_logic_reward_func(completions, **kwargs):
    rewards = []
    for content in completions:
        nums = re.findall(r'\[Index (\d+)\]', str(content))
        expected = [str(i) for i in range(len(nums))]
        rewards.append(1.0 if nums and nums == expected else -1.0)
    return rewards

# =======================================================
# 3. Main Training Pipeline
# =======================================================

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=True, fix_mistral_regex=True)
    tokenizer.pad_token = tokenizer.eos_token
    if "[#CITE#]" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["[#CITE#]"]})

    raw_dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    
    # [Length Filter] Intercept overly long samples to prevent token limit OOM or 4097 errors
    SAFE_PROMPT_LIMIT = 2400 
    def filter_by_token_length(example):
        full_text = SYSTEM_PROMPT + example["input"]
        return len(tokenizer.encode(full_text)) < SAFE_PROMPT_LIMIT

    print(f"[*] Original sample count: {len(raw_dataset)}")
    dataset = raw_dataset.filter(filter_by_token_length)
    print(f"[*] Remaining samples after filtering: {len(dataset)}")

    def prepare_dataset(example):
        return {
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": example["input"]}],
            "answer": example["output"]
        }
    dataset = dataset.map(prepare_dataset)

    # 1. Keep the base configuration clean to avoid initialization validation errors.
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-7,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_generations=4, 
        bf16=True,
        use_vllm=True,
        vllm_mode='colocate', 
        vllm_gpu_memory_utilization=0.3,
        vllm_max_model_length=4096,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        save_strategy="steps",
        save_steps=100,
        logging_steps=1,
        report_to="none"
    )

    # 2. [Version Patch] Dynamically override the legacy 256 default token limit.
    # If GRPOConfig lacks native support for these attributes in older versions, inject them manually.
    if not hasattr(training_args, 'max_prompt_length'):
        print("[!] Older TRL version detected. Manually injecting length parameters...")
        # Forcibly set internal attributes to bypass initialization checks.
        setattr(training_args, 'max_prompt_length', 2048)
        setattr(training_args, 'max_completion_length', 2048)

    trainer = GRPOTrainer(
        model=MODEL_PATH,
        processing_class=tokenizer,
        reward_funcs=[
            format_reward_func,
            count_consistency_reward_func,
            index_logic_reward_func,
            reasoning_start_reward_func, # Enforces prefix constraint reward
            local_context_window_reward_func,
            hybrid_content_reward_func,
            context_relevance_reward_func
        ],
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

if __name__ == "__main__":
    main()
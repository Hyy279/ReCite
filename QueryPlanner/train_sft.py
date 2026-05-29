import torch
import torch.nn as nn
import os
import random
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from accelerate import Accelerator
from torch.optim import AdamW

# =======================================================
# 1. Base Configuration
# =======================================================
set_seed(42)  
MODEL_DIR = '/path/to/model_weights/qwen/Qwen3-4B-Instruct-2507' 
DATA_FILE = "/path/to/train.jsonl"
OUTPUT_DIR = "./output_sft" 
LOG_FILE = "training_log.json"

MAX_LEN = 4096 
BATCH_SIZE = 1  
GRAD_ACCUM = 32 
LEARNING_RATE = 1e-5
EPOCHS = 3

# Initialize Accelerator for distributed training and gradient accumulation
accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCUM)
device = accelerator.device

# Initialize Tokenizer and ensure the special token is registered
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
if "[#CITE#]" not in tokenizer.all_special_tokens:
    tokenizer.add_special_tokens({'additional_special_tokens': ["[#CITE#]"]})

# =======================================================
# 2. Data Processing Pipeline
# =======================================================
raw_dataset = load_dataset("json", data_files=DATA_FILE, split="train")
dataset_dict = raw_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset_raw = dataset_dict["train"]

def tokenize_fn(example):
    prompt_header = (
        "You are an expert academic citation analyst.\n\n"
        "### Task\n"
        "Analyze the context of the given text. For each citation marker [#CITE#], infer what keywords should be used to search for the paper that needs to be cited. Each [#CITE#] marker in the input must correspond to one [Index n] in the output in order, and the total number of indices must exactly match the number of citation markers.\n\n"
        "### Output Components\n"
        "1. <reasoning>: A 1-2 sentence logical analysis of the context explaining the core role of the citation.\n"
        "   MUST start with the fixed phrase: \"The purpose of this citation is to...\"\n"
        "2. <keywords>: 3-5 precise keywords to retrieve the target document.\n"
        "   - STRICTLY PROHIBITED: Do NOT include any author names or surnames.\n"
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
    
    system_text = f"<|im_start|>system\n{prompt_header}<|im_end|>\n"
    user_text = f"<|im_start|>user\n{example['input']}<|im_end|>\n"
    assistant_header = f"<|im_start|>assistant\n"
    
    full_prompt = system_text + user_text + assistant_header
    assistant_text = f"{example['output']}<|im_end|>"
    
    prefix_enc = tokenizer(full_prompt, add_special_tokens=False)
    full_enc = tokenizer(full_prompt + assistant_text, truncation=True, max_length=MAX_LEN, padding="max_length", add_special_tokens=False)
    
    labels = list(full_enc["input_ids"])
    prefix_len = len(prefix_enc["input_ids"])
    
    # Apply label masking (-100) for the prompt section and padding tokens
    for i in range(len(labels)):
        if i < prefix_len or labels[i] == tokenizer.pad_token_id:
            labels[i] = -100
            
    return {"input_ids": full_enc["input_ids"], "attention_mask": full_enc["attention_mask"], "labels": labels}

train_dataset = train_dataset_raw.map(tokenize_fn, remove_columns=train_dataset_raw.column_names, num_proc=8)
train_dataset.set_format(type='torch')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# =======================================================
# 3. Model Loading & Optimization
# =======================================================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, 
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True
)
model.resize_token_embeddings(len(tokenizer))
model.config.use_cache = False
model.gradient_checkpointing_enable()

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
num_training_steps = (len(train_loader) * EPOCHS) // GRAD_ACCUM
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.05 * num_training_steps), num_training_steps=num_training_steps
)

model, optimizer, train_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader, lr_scheduler)

# =======================================================
# 4. Training Loop
# =======================================================
if accelerator.is_main_process and not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for step, batch in enumerate(pbar):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.detach().item()
            pbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    if accelerator.is_main_process:
        avg_loss = total_loss / len(train_loader)
        save_path = os.path.join(OUTPUT_DIR, f"epoch_{epoch}")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(save_path, safe_serialization=True)
        tokenizer.save_pretrained(save_path)
        print(f"[*] Epoch {epoch} Saved. Avg Loss: {avg_loss:.4f}")

print("[*] Training successfully completed!")
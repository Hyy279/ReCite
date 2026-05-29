import json
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from openai import OpenAI
from tqdm import tqdm

# =======================================================
# 1. Global Configuration Section
# =======================================================
# Insert your LLM API configuration here
YOUR_API_KEY = "YOUR_API_KEY"  
BASE_URL = "YOUR_LLM_BASE_URL" 

# Insert your Semantic Scholar API Keys here
S2_KEYS = [
    "YOUR_S2_API_KEY_1", 
    "YOUR_S2_API_KEY_2"
]

FILE_TEST = "test_200.jsonl"           # File containing the original text
FILE_PRED = "eval_results.jsonl"     # File containing predictions and ground truth
FILE_OUTPUT = "lenient_evaluation_results.jsonl"        # Path to save evaluation results

MAX_WORKERS = 10  # Number of concurrent threads

# Initialize the LLM client
llm_client = OpenAI(api_key=YOUR_API_KEY, base_url=BASE_URL)

# =======================================================
# 2. Utility Classes & Functions
# =======================================================
class S2RateLimiter:
    """Thread-safe rate limiter for Semantic Scholar API"""
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.last_used = {k: 0.0 for k in api_keys}
        self.lock = threading.Lock()

    def get_key_and_wait(self):
        if not self.api_keys: return None
        with self.lock:
            oldest_key = min(self.last_used, key=self.last_used.get)
            now = time.time()
            elapsed = now - self.last_used[oldest_key]
            # Strictly enforce a 2.1-second cooldown per key
            wait_time = max(0, 2.1 - elapsed)
            self.last_used[oldest_key] = now + wait_time
        if wait_time > 0:
            time.sleep(wait_time)
        return oldest_key

s2_limiter = S2RateLimiter(S2_KEYS)

def normalize_text(t):
    """Text normalization for robust matching operations."""
    if not t: return ""
    return re.sub(r'\s+', '', str(t).lower())

def fetch_s2_paper_info(title):
    """Call S2 API to fetch only the paper's TLDR."""
    api_key = s2_limiter.get_key_and_wait()
    headers = {"x-api-key": api_key} if api_key else {}
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={title}&limit=1&fields=title,tldr"
    
    for attempt in range(3):
        try:
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code == 200:
                data = res.json().get('data', [])
                if data:
                    paper = data[0]
                    tldr = paper.get('tldr')
                    tldr_text = tldr.get('text') if tldr else ""
                    return f"TLDR: {tldr_text}" if tldr_text else "No TLDR found."
                return "No TLDR found."
            elif res.status_code >= 500:
                time.sleep(3) # Retry on S2 server errors
                continue
            else:
                break
        except Exception:
            time.sleep(2)
    return "API Request Failed or Timeout."

def llm_judge_appropriateness(paragraph, anchor, pred_title, paper_info):
    """Core: Uses an LLM as a judge to evaluate if a citation is appropriate (Outputs 1 or 0)."""
    prompt = f"""
You are an expert academic peer reviewer. Your task is to evaluate if a cited paper is academically appropriate for a specific location in a paragraph.

[Context Paragraph]
{paragraph}

[Citation Anchor (The text immediately preceding the citation)]
"{anchor}"

[Predicted Paper to be Inserted Here]
Title: {pred_title}
Details: {paper_info}

[Task]
Determine if this paper provides appropriate evidence, background, or methodology support for the claim made right before the anchor.
Respond in strict JSON format with exactly two keys:
- "judgment": strictly output the integer 1 (if appropriate) or 0 (if inappropriate).
- "reason": a 1-sentence explanation of why.
"""
    # Robust JSON parsing and error handling mechanism
    for attempt in range(3):
        try:
            response = llm_client.chat.completions.create(
                model="mimo-v2.5-pro", # Ensure this matches your specific model name
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            raw_content = response.choices[0].message.content.strip()
            
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:]
            elif raw_content.startswith("```"):
                raw_content = raw_content[3:]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]
            
            raw_content = raw_content.strip()
            res_json = json.loads(raw_content)
            
            is_appropriate = int(res_json.get("judgment", 0)) == 1
            return is_appropriate, res_json.get("reason", "")
            
        except json.JSONDecodeError:
            if attempt == 2:
                return False, "JSON Parse Failed"
            time.sleep(1)
        except Exception as e:
            time.sleep(2)
            
    return False, "Max Retries Exceeded"

# =======================================================
# 3. Core Evaluation Logic
# =======================================================
def evaluate_single_doc(doc_id, text, preds, gts):
    gt_anchors_norm = [normalize_text(gt.get('anchor', '')) for gt in gts]
    gt_titles_norm = [normalize_text(gt.get('title', '')) for gt in gts]
    
    # ---------------- A. Position-Only Metrics ----------------
    pos_tp = 0
    for p in preds:
        pred_anchor_norm = normalize_text(p.get('anchor_prefix', ''))
        is_pos_correct = any(pred_anchor_norm in g_a or g_a in pred_anchor_norm for g_a in gt_anchors_norm if g_a)
        if is_pos_correct:
            pos_tp += 1
            
    pos_fp = len(preds) - pos_tp
    pos_fn = max(0, len(gts) - pos_tp)

    # ---------------- B. Strict LLM Judgment Metrics ----------------
    strict_tp = 0
    llm_judgments = []
    
    for p in preds:
        pred_anchor = p.get('anchor_prefix', '')
        pred_title = p.get('bib_title', '')
        pred_anchor_norm = normalize_text(pred_anchor)
        pred_title_norm = normalize_text(pred_title)
        
        # Step 1: Check position. If incorrect, fail immediately without API call.
        is_pos_correct = any(pred_anchor_norm in g_a or g_a in pred_anchor_norm for g_a in gt_anchors_norm if g_a)
        
        if not is_pos_correct:
            llm_judgments.append({"title": pred_title, "judgment": 0, "reason": "Position incorrect"})
            continue
            
        # Step 2: Hard match bypass. If both position and title match perfectly, pass immediately.
        is_hard_match = False
        for gt in gts:
            g_a_norm = normalize_text(gt.get('anchor', ''))
            g_t_norm = normalize_text(gt.get('title', ''))
            if (pred_anchor_norm in g_a_norm or g_a_norm in pred_anchor_norm) and \
               (pred_title_norm in g_t_norm or g_t_norm in pred_title_norm) and pred_title_norm:
                is_hard_match = True
                break
                
        if is_hard_match:
            strict_tp += 1
            llm_judgments.append({"title": pred_title, "judgment": 1, "reason": "Position correct and hard match passed"})
            continue
            
        # Step 3: Title extraction failure. Fail immediately.
        if not pred_title or "Not Extracted" in pred_title:
            llm_judgments.append({"title": pred_title, "judgment": 0, "reason": "Position correct but title extraction failed"})
            continue
            
        # Step 4: Final LLM Judgment (API is called only if reaching this step).
        paper_info = fetch_s2_paper_info(pred_title)
        is_appropriate, reason = llm_judge_appropriateness(text, pred_anchor, pred_title, paper_info)
        
        if is_appropriate:
            strict_tp += 1
            llm_judgments.append({"title": pred_title, "judgment": 1, "reason": f"LLM Judgment: {reason}"})
        else:
            llm_judgments.append({"title": pred_title, "judgment": 0, "reason": f"LLM Judgment: {reason}"})

    strict_fp = len(preds) - strict_tp
    strict_fn = max(0, len(gts) - strict_tp)

    return {
        "id": doc_id,
        "pos_tp": pos_tp, "pos_fp": pos_fp, "pos_fn": pos_fn,
        "content_tp": strict_tp, "content_fp": strict_fp, "content_fn": strict_fn,
        "judgments": llm_judgments
    }

# =======================================================
# 4. Main Execution Flow
# =======================================================
def main():
    print("[*] Loading dataset...")
    
    texts_dict = {}
    with open(FILE_TEST, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            texts_dict[data['id']] = data.get('text', '')
            
    eval_tasks = []
    with open(FILE_PRED, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            doc_id = data['id']
            preds = data.get('agent_predictions', [])
            gts = data.get('ground_truth', [])
            text = texts_dict.get(doc_id, "")
            eval_tasks.append((doc_id, text, preds, gts))

    print(f"[*] Loaded {len(eval_tasks)} records. Initializing LLM judge engine...")
    
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(evaluate_single_doc, *task) for task in eval_tasks]
        
        for future in tqdm(as_completed(futures), total=len(eval_tasks), desc="Evaluation Progress", dynamic_ncols=True):
            results.append(future.result())
            
    total_pos_tp = sum(r['pos_tp'] for r in results)
    total_pos_fp = sum(r['pos_fp'] for r in results)
    total_pos_fn = sum(r['pos_fn'] for r in results)
    
    total_cnt_tp = sum(r['content_tp'] for r in results)
    total_cnt_fp = sum(r['content_fp'] for r in results)
    total_cnt_fn = sum(r['content_fn'] for r in results)

    def calc_metrics(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        return p, r, f1

    pos_p, pos_r, pos_f1 = calc_metrics(total_pos_tp, total_pos_fp, total_pos_fn)
    cnt_p, cnt_r, cnt_f1 = calc_metrics(total_cnt_tp, total_cnt_fp, total_cnt_fn)

    print("\n" + "="*40)
    print("Final Evaluation Report")
    print("="*40)
    print("[Position-Only Metrics]")
    print(f"Precision: {pos_p:.4f} | Recall: {pos_r:.4f} | F1: {pos_f1:.4f}")
    print("-" * 40)
    print("[Strict End-to-End Metrics]")
    print(f"Precision: {cnt_p:.4f} | Recall: {cnt_r:.4f} | F1: {cnt_f1:.4f}")
    print("="*40)

    with open(FILE_OUTPUT, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[*] Detailed judgment logs saved to: {FILE_OUTPUT}")

if __name__ == "__main__":
    main()
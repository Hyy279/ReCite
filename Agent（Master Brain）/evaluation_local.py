import json
import time
import re
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from agent import CiteAgent

# =======================================================
# Configuration Section
# =======================================================
TEST_FILE = "test_200.jsonl" 
RESULT_FILE = "evaluation_results.jsonl"

# Insert your Semantic Scholar API Keys here
S2_KEYS = [
    "YOUR_API_KEY_1",  
    "YOUR_API_KEY_2"
]

MAX_WORKERS = 15

# Initialize the Agent with anonymized model paths
agent = CiteAgent(
    Master_Brain_path="/path/to/Master_Brain",
    CiteLocator_path="/path/to/CiteLocato",
    QueryPlanner_path="/path/to/QueryPlanner",
    s2_api_keys=S2_KEYS
)
# =======================================================

def normalize_string(s):
    """
    Text normalization: converts to lowercase and removes all punctuation and spaces 
    to ensure highly robust matching against formatting variations.
    """
    if not s: return ""
    return re.sub(r'[^a-z0-9]', '', s.lower())

def extract_citations_from_output(final_article, global_bibliography):
    """Extracts numerical citation markers and aligns them with generated bibliography metadata."""
    extracted_citations = []
    
    # 1. Scan the main text for numerical insertion points (e.g., [1])
    matches = list(re.finditer(r'\[(\d+)\]', final_article))
    for m in matches:
        ref_num = m.group(1)
        raw_prefix = final_article[max(0, m.start() - 30):m.start()]
        clean_prefix = re.sub(r'\[\d+\]', '', raw_prefix).strip()
        
        extracted_citations.append({
            "ref_num": ref_num,
            "anchor_prefix": clean_prefix,
            "bib_author": "", 
            "bib_title": "" 
        })
        
    # 2. Scan the global bibliography to supplement title and author information
    for bib_str in global_bibliography:
        num_match = re.search(r'\[(\d+)\]', bib_str)
        if not num_match: continue
        ref_num = num_match.group(1)
        
        title_match = re.search(r'\btitle\s*=\s*[\{"](.*?)(?<!\\)[\}"]', bib_str, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else ""
        
        author_match = re.search(r'author\s*=\s*[\{"](.*?)(?<!\\)[\}"]', bib_str, re.IGNORECASE | re.DOTALL)
        author = author_match.group(1).strip() if author_match else ""

        if not title and not author:
            fallback_match = re.search(r'(.*?),\s*"(.*?)"', bib_str)
            if fallback_match:
                author = fallback_match.group(1).strip()
                title = fallback_match.group(2).strip()
            else:
                title = "Title Not Extracted"
                author = "Author Not Extracted"
        
        for cite in extracted_citations:
            if cite["ref_num"] == ref_num:
                cite["bib_title"] = title
                cite["bib_author"] = author
                
    return extracted_citations

def calculate_metrics(gt_list, pred_list):
    """Calculates the 5 core evaluation metrics: JPPA, RE, PE, FN, FP."""
    stats = {"JPPA": 0, "RE": 0, "PE": 0, "FN": 0, "FP": 0}
    
    matched_gt = set()
    matched_pred = set()

    for p_idx, p_cite in enumerate(pred_list):
        p_anchor = normalize_string(p_cite["anchor_prefix"])
        p_title = normalize_string(p_cite["bib_title"])
        p_author = normalize_string(p_cite["bib_author"]) 
        
        for g_idx, g_cite in enumerate(gt_list):
            if g_idx in matched_gt: continue
            
            g_anchor = normalize_string(g_cite.get("anchor", ""))
            g_title = normalize_string(g_cite.get("title", ""))
            g_author = normalize_string(g_cite.get("author", "")) 
            
            pos_match = (g_anchor in p_anchor) or (p_anchor in g_anchor)
            
            article_match = False
            if bool(p_title) and (p_title == g_title):
                article_match = True
            elif bool(p_author) and bool(g_author):
                if (g_author in p_author) or (p_author in g_author):
                    article_match = True

            if pos_match and article_match:
                stats["JPPA"] += 1     
                matched_gt.add(g_idx)
                matched_pred.add(p_idx)
                break
            elif pos_match and not article_match:
                stats["RE"] += 1       
                matched_gt.add(g_idx)
                matched_pred.add(p_idx)
                break
            elif not pos_match and article_match:
                stats["PE"] += 1       
                matched_gt.add(g_idx)
                matched_pred.add(p_idx)
                break

    stats["FN"] = len(gt_list) - len(matched_gt)     
    stats["FP"] = len(pred_list) - len(matched_pred) 
    
    return stats

def print_academic_metrics(total_stats):
    """Formats and prints the aggregated academic evaluation metrics."""
    jppa = total_stats["JPPA"]
    re_stat = total_stats["RE"]
    pe = total_stats["PE"]
    fn = total_stats["FN"]
    fp = total_stats["FP"]
    
    gt_total = jppa + re_stat + pe + fn
    pred_total = jppa + re_stat + pe + fp
    
    precision = jppa / pred_total if pred_total > 0 else 0
    recall = jppa / gt_total if gt_total > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*40)
    print("Academic Evaluation Metrics:")
    print(f"Total Ground Truth : {gt_total}")
    print(f"Total Predictions  : {pred_total}")
    print("-" * 40)
    print(f"Precision          : {precision * 100:.2f}%")
    print(f"Recall             : {recall * 100:.2f}%")
    print(f"F1-Score           : {f1 * 100:.2f}%")
    print("-" * 40)
    print("Error Analysis:")
    print(f"Reference Error Rate (RE) : {(re_stat / gt_total * 100) if gt_total > 0 else 0:.2f}%")
    print(f"Placement Error Rate (PE) : {(pe / gt_total * 100) if gt_total > 0 else 0:.2f}%")
    print(f"Miss Rate (FN)            : {(fn / gt_total * 100) if gt_total > 0 else 0:.2f}%")
    print(f"Hallucination Rate (FP)   : {(fp / pred_total * 100) if pred_total > 0 else 0:.2f}%")
    print("="*40 + "\n")

# =======================================================
# Concurrency & Main Execution Engine
# =======================================================
def process_single_worker(item):
    """Worker function to process a single document."""
    doc_id = item.get("id", "Unknown")
    start_time = time.time()
    
    # Execute vLLM Agent reasoning
    final_article, global_bibliography = agent.process_document(item.get("text", ""))
    cost_time = round(time.time() - start_time, 2)
    
    # Evaluate predictions
    pred_citations = extract_citations_from_output(final_article, global_bibliography)
    metrics = calculate_metrics(item.get("gt_citations", []), pred_citations)
    
    return {
        "id": doc_id,
        "cost_time_seconds": cost_time,
        "metrics": metrics,
        "agent_predictions": pred_citations,
        "ground_truth": item.get("gt_citations", [])
    }

def main():
    print(f"[*] Starting End-to-End Evaluation Engine (Concurrent vLLM Mode | Max Workers: {MAX_WORKERS})")
    
    total_stats = {"JPPA": 0, "RE": 0, "PE": 0, "FN": 0, "FP": 0}
    processed_results = {}  
    
    # 1. Resume mechanism: Parse existing results to skip processed items
    if os.path.exists(RESULT_FILE):
        with open(RESULT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    doc_id = str(data.get("id"))
                    processed_results[doc_id] = data
                except Exception as e:
                    print(f"  [Warning] Failed to parse historical record, skipping line: {e}")
                    
        # Clean historical file by overwriting with deduplicated data
        with open(RESULT_FILE, 'w', encoding='utf-8') as f:
            for data in processed_results.values():
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                
        # Accumulate metrics from existing valid records
        for data in processed_results.values():
            for k in total_stats:
                total_stats[k] += data["metrics"].get(k, 0)
                
        print(f"[*] Successfully loaded and cleaned {RESULT_FILE}. Found {len(processed_results)} unique completed records.")
    else:
        print("[*] No historical results found. Initiating full evaluation sequence.")

    # 2. Load the test dataset
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        all_test_data = [json.loads(line) for line in f][:200]
        
    # 3. Filter out already processed documents
    remaining_data = [item for item in all_test_data if str(item.get("id")) not in processed_results]
    
    if not remaining_data:
        print("[*] All 200 data points have been processed. Outputting final statistics.")
        print_academic_metrics(total_stats)
        return

    print(f"[*] Processing the remaining {len(remaining_data)} data points in parallel...")
    
    # Mutex lock for thread-safe file writing
    write_lock = threading.Lock()

    # 4. Dispatch tasks via ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_worker, item) for item in remaining_data]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            try:
                res = future.result()
                
                # Thread-safe metric accumulation and file I/O
                with write_lock:
                    for k in total_stats:
                        total_stats[k] += res["metrics"][k]
                    
                    with open(RESULT_FILE, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(res, ensure_ascii=False) + "\n")
                
                tqdm.write(f"  -> [ID: {res['id']}] Completed | Time: {res['cost_time_seconds']}s | Metrics: {res['metrics']}")
            except Exception as e:
                tqdm.write(f"  [Error] Execution failed for an item (skipped safely): {str(e)}")

    print("\n" + "="*40)
    print("[*] Evaluation Completed. Global Metrics Summary:")
    print(json.dumps(total_stats, indent=4))
    print(f"[*] Detailed results and traces saved to: {RESULT_FILE}")
    print("="*40)
    
    # Final global evaluation output
    print_academic_metrics(total_stats)

if __name__ == "__main__":
    main()
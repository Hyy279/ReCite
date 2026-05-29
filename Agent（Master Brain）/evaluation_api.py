import json
import re
import requests
import time
import os
import threading
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# =======================================================
# 🚀 新增：S2 API 智能排队限流器
# =======================================================
class S2RateLimiter:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        # 记录每个 key 上次使用（或预计将可用）的时间戳
        self.last_used = {k: 0.0 for k in api_keys}
        self.lock = threading.Lock()

    def get_key_and_wait(self):
        """线程安全的获取空闲 Key，如果不满足 1.2s 冷却则原地排队等待"""
        if not self.api_keys: return None
        
        with self.lock:
            # 找到最早可用的那个 Key
            oldest_key = min(self.last_used, key=self.last_used.get)
            now = time.time()
            elapsed = now - self.last_used[oldest_key]
            wait_time = max(0, 1.2 - elapsed)
            
            # 将这个 Key 的“下次可用时间”投射到未来
            self.last_used[oldest_key] = now + wait_time
            
        # 在锁外睡眠，不阻塞其他线程的排队分发
        if wait_time > 0:
            time.sleep(wait_time)
            
        return oldest_key

# =======================================================
# Baseline Agent 类 (合并到同一文件，方便管理并发锁)
# =======================================================
class BaselineCiteAgent:
    def __init__(self, provider="deepseek", llm_api_key=None, s2_api_keys=None):
        # 🚀 替换为排队限流器
        self.s2_limiter = S2RateLimiter(s2_api_keys or [])
        self.search_cache = {}
        self.provider = provider
        
        print(f"\n[🧠] 正在唤醒 Baseline LLM (引擎: {provider.upper()})...")
        
        if provider == "deepseek":
            self.client = OpenAI(api_key=llm_api_key, base_url="https://api.deepseek.com")
            self.model_name = "deepseek-v4-flash"
        elif provider == "qwen":
            self.client = OpenAI(api_key=llm_api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
            self.model_name = "qwen3.6-27b"
        elif provider == "qwen3.6-plus":
            self.client = OpenAI(api_key=llm_api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
            self.model_name = "qwen3.6-plus"
        elif provider == "zhipu":
            self.client = OpenAI(api_key=llm_api_key, base_url="https://open.bigmodel.cn/api/paas/v4/")
            self.model_name = "glm-4-plus"
        elif provider == "mimo":
            self.client = OpenAI(api_key=llm_api_key, base_url="https://token-plan-cn.xiaomimimo.com/v1")
            self.model_name = "mimo-v2.5-pro"
        elif provider == "kimi":
            self.client = OpenAI(api_key=llm_api_key, base_url="https://api.moonshot.cn/v1")
            self.model_name = "kimi-k2-0905-preview"
        elif provider == "gpt-5.1-chat":
            self.client = OpenAI(api_key=llm_api_key, base_url="https://api.gptsapi.net/v1")
            self.model_name = "gpt-5.1-chat"
        else:
            raise ValueError("Provider 必须是支持的模型")

    def _call_llm(self, system_prompt, user_prompt):
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1, 
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # print(f"      ❌ API 调用失败: {e}") # 多线程下关闭报错打印防刷屏
            time.sleep(3) 
            return ""

    def _call_semantic_scholar_api(self, query, max_retries=3):
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {"query": query, "limit": 12, "fields": "paperId,title,authors,tldr,citationCount,citationStyles"}

        for attempt in range(max_retries):
            try:
                # 🚀 绝杀：从排队锁里拿 Key，自动处理 1.2s 的冷却等待
                api_key = self.s2_limiter.get_key_and_wait()
                headers = {"x-api-key": api_key.strip()} if api_key else {}
                
                response = requests.get(url, params=params, headers=headers, timeout=15)
                if response.status_code == 200:
                    return response.json().get('data', [])
                elif response.status_code == 429:
                    time.sleep(2 ** (attempt + 1))
                    continue
                else: break
            except Exception:
                time.sleep(2)
        return []

    def _step1_mark_citations(self, text):
        system_prompt = "You are a professional academic copy-editor."
        user_prompt = f"Task: Act as a professional academic copy-editor. Insert the specified marker [#CITE#] at places that require citations, such as when mentioning prior works, existing methods, algorithms, models, datasets or other places where citations are needed. Rules: Do not alter any original content. Do not add extra preamble. Output only the processed text.\n\nInput Text:\n{text}"
        marked_text = self._call_llm(system_prompt, user_prompt)
        return marked_text.replace("```text", "").replace("```", "").strip()

    def _step2_analyze_intents(self, marked_text):
        system_prompt = (
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
        user_prompt = f"Please analyze this marked text:\n{marked_text}"
        response = self._call_llm(system_prompt, user_prompt)
        
        intents = []
        blocks = re.split(r'\[Index \d+\]', response)[1:] 
        for i, block in enumerate(blocks):
            reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', block, re.DOTALL)
            keywords_match = re.search(r'<keywords>(.*?)</keywords>', block, re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
            keywords_str = keywords_match.group(1).strip() if keywords_match else "[]"
            try:
                keywords = json.loads(keywords_str)
            except:
                keywords = re.findall(r'"(.*?)"', keywords_str)
            intents.append({"index": i, "reasoning": reasoning, "keywords": keywords})
        return intents

    def _step3_select_paper(self, index, reasoning, keywords, marked_text):
        if not keywords: return None
        search_query = " ".join(keywords[:2])
        papers = self._call_semantic_scholar_api(search_query)
        if not papers: return None
        
        context = ""
        for p in papers:
            self.search_cache[p['paperId']] = p
            authors_str = ", ".join([a.get('name', 'Unknown') for a in p.get('authors', [])[:3]])
            tldr_data = p.get('tldr')
            tldr_text = tldr_data.get('text', 'No TLDR available.') if tldr_data else 'No TLDR available.'
            context += f"- ID: {p['paperId']}\n  Title: {p.get('title')}\n  Authors: {authors_str}\n  Citations: {p.get('citationCount', 0)}\n  TLDR: {tldr_text}\n\n"

        system_prompt = "You are an academic paper selector. Your goal is to match the exact foundational paper needed for the specific citation context."
        select_prompt = f"You need to select the best citation for the marker [Index {index}] (which corresponds to the {index+1}-th [#CITE#] marker in the text).\n\n=== ORIGINAL CONTEXT ===\n{marked_text}\n========================\n\nYour reasoning for this citation was: {reasoning}\n\nHere are the candidate papers from the database:\n{context.strip()}\n\nEvaluate carefully based on the original context. If you find the correct foundational paper, reply with ONLY its ID. If NONE of these papers match the context, reply with exactly \"NONE\"."
        
        selection = self._call_llm(system_prompt, select_prompt).strip()
        if selection != "NONE" and any(p['paperId'] == selection for p in papers):
            return selection
        return None

    def process_document(self, long_text):
        paragraphs = [p.strip() for p in long_text.split('\n') if p.strip()]
        global_cite_counter = 1
        final_merged_paragraphs = []
        global_bibliography = []
        
        for i, para in enumerate(paragraphs):
            marked_text = self._step1_mark_citations(para)
            cite_count = marked_text.count("[#CITE#]")
            if cite_count == 0:
                final_merged_paragraphs.append(marked_text)
                continue
            
            intents = self._step2_analyze_intents(marked_text)
            local_citations = {}
            for intent in intents:
                idx = intent['index']
                selected_id = self._step3_select_paper(idx, intent['reasoning'], intent['keywords'], marked_text)
                
                if selected_id:
                    matched_paper = self.search_cache.get(selected_id)
                    bibtex = matched_paper.get('citationStyles', {}).get('bibtex')
                    if bibtex:
                        local_citations[idx] = bibtex.strip()
                    else:
                        title = matched_paper.get('title', 'Unknown Title')
                        authors = matched_paper.get('authors', [])
                        author_str = authors[0].get('name') if authors else "Unknown"
                        local_citations[idx] = f"{author_str} et al., \"{title}\""
                else:
                    local_citations[idx] = "Unknown Source (Baseline LLM failed to resolve)"
            
            for local_idx in range(cite_count):
                marked_text = marked_text.replace("[#CITE#]", f"[{global_cite_counter}]", 1)
                cite_str = local_citations.get(local_idx, "Unknown Source")
                global_bibliography.append(f"[{global_cite_counter}]\n{cite_str}\n")
                global_cite_counter += 1
                
            final_merged_paragraphs.append(marked_text)
            
        final_article = "\n\n".join(final_merged_paragraphs)
        return final_article, global_bibliography

# ================= 配置区 =================
# 🚀 修改：读取 200 条的测试集
TEST_FILE = "retest/test_200.jsonl"
RESULT_FILE = "retest/eval_results_v4flash_limit12.jsonl"

# 🚀 填入你的两个 S2 API Key
S2_KEYS = [
    "9BRUHRuJCc8pwaxHOFjGX7kQVNQl8VhE34WXESnP",  
    "s2k-yMRZe7qC0WJmsNRsKNUkWZe3UaqN6YIF6clPZPPW"  # <--- 请替换成你的真实Key！
]

PROVIDER = "deepseek"
API_KEY = "sk-d34e28b618e445b594ee509dd1523faf"

# Kimi 并发 3，其他模型并发 5
MAX_WORKERS = 3 if PROVIDER == "kimi" else 15

agent = BaselineCiteAgent(
    provider=PROVIDER, 
    llm_api_key=API_KEY,
    s2_api_keys=S2_KEYS
)

# ================= 辅助函数与评测 (原封不动) =================
def normalize_string(s):
    if not s: return ""
    return re.sub(r'[^a-z0-9]', '', s.lower())

def extract_citations_from_output(final_article, global_bibliography):
    extracted_citations = []
    matches = list(re.finditer(r'\[(\d+)\]', final_article))
    for m in matches:
        ref_num = m.group(1)
        raw_prefix = final_article[max(0, m.start() - 30):m.start()]
        clean_prefix = re.sub(r'\[\d+\]', '', raw_prefix).strip()
        extracted_citations.append({"ref_num": ref_num, "anchor_prefix": clean_prefix, "bib_author": "", "bib_title": "" })
        
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
                stats["JPPA"] += 1; matched_gt.add(g_idx); matched_pred.add(p_idx); break
            elif pos_match and not article_match:
                stats["RE"] += 1; matched_gt.add(g_idx); matched_pred.add(p_idx); break
            elif not pos_match and article_match:
                stats["PE"] += 1; matched_gt.add(g_idx); matched_pred.add(p_idx); break

    stats["FN"] = len(gt_list) - len(matched_gt)     
    stats["FP"] = len(pred_list) - len(matched_pred) 
    return stats

def print_academic_metrics(total_stats):
    jppa, re_stat, pe, fn, fp = total_stats["JPPA"], total_stats["RE"], total_stats["PE"], total_stats["FN"], total_stats["FP"]
    gt_total = jppa + re_stat + pe + fn
    pred_total = jppa + re_stat + pe + fp
    precision = jppa / pred_total if pred_total > 0 else 0
    recall = jppa / gt_total if gt_total > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "📊"*15)
    print("📉 Baseline 学术论文标准评测指标 (Academic Metrics):")
    print(f"Total Ground Truth : {gt_total}")
    print(f"Total Predictions  : {pred_total}")
    print("-" * 30)
    print(f"Precision (精确率) : {precision * 100:.2f}%")
    print(f"Recall    (召回率) : {recall * 100:.2f}%")
    print(f"F1-Score  (F1值)   : {f1 * 100:.2f}%")
    print("-" * 30)
    print("🔬 细分错误率 (Error Analysis):")
    print(f"Reference Error Rate (RE) : {(re_stat / gt_total * 100) if gt_total > 0 else 0:.2f}%")
    print(f"Placement Error Rate (PE) : {(pe / gt_total * 100) if gt_total > 0 else 0:.2f}%")
    print(f"Miss Rate (FN)            : {(fn / gt_total * 100) if gt_total > 0 else 0:.2f}%")
    print(f"Hallucination Rate (FP)   : {(fp / pred_total * 100) if pred_total > 0 else 0:.2f}%")
    print("📊"*15 + "\n")

# ================= 并发调度主控 =================
def process_single_worker(item):
    doc_id = item.get("id", "Unknown")
    start_time = time.time()
    
    final_article, global_bibliography = agent.process_document(item.get("text", ""))
    cost_time = round(time.time() - start_time, 2)
    
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
    print(f"🚀 启动高并发多线程 Baseline 测试 (Max Workers: {MAX_WORKERS})")
    
    # 1. 恢复之前跑完的数据统计
    total_stats = {"JPPA": 0, "RE": 0, "PE": 0, "FN": 0, "FP": 0}
    existing_count = 0
    if os.path.exists(RESULT_FILE):
        with open(RESULT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                existing_count += 1
                for k in total_stats:
                    total_stats[k] += data["metrics"][k]
        print(f"✅ 成功读取 {RESULT_FILE}：已完成 {existing_count} 条数据的评测，直接接力！")
    else:
        print("🆕 未发现历史评测结果，将从第 1 条开始。")

    # 2. 读取完整的 200 条测试集，并切片剩下的数据
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        all_test_data = [json.loads(line) for line in f][:200]
        
    remaining_data = all_test_data[existing_count:]
    if not remaining_data:
        print("🎉 所有 200 条数据已测试完毕，直接输出全局统计！")
        print_academic_metrics(total_stats)
        return

    print(f"⏳ 即将并行处理剩下的 {len(remaining_data)} 条数据...")
    
    write_lock = threading.Lock()

    # 3. 启动线程池并发跑图
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_worker, item) for item in remaining_data]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            try:
                res = future.result()
                
                # 线程安全地更新统计指标和写入文件
                with write_lock:
                    for k in total_stats:
                        total_stats[k] += res["metrics"][k]
                    
                    with open(RESULT_FILE, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(res, ensure_ascii=False) + "\n")
                
                tqdm.write(f"🟢 [ID: {res['id']}] 完成 | 耗时: {res['cost_time_seconds']}s | 成绩: {res['metrics']}")
            except Exception as e:
                tqdm.write(f"❌ 运行报错: {str(e)}")

    print("\n" + "🌟"*20)
    print("🏆 全部测试完毕！200条全局指标汇总：")
    print(json.dumps(total_stats, indent=4))
    print(f"📁 详细结果已增量保存至: {RESULT_FILE}")
    print("🌟"*20)
    
    # 打印融合了前 50 条和新增跑完结果的 全局最终指标
    print_academic_metrics(total_stats)

if __name__ == "__main__":
    main()
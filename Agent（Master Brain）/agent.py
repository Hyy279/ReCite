import json
import re
import requests
import time
import threading
from tools import CiteModelWorkers

# =======================================================
# S2 API Rate Limiter (Thread-safe)
# =======================================================
class S2RateLimiter:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        # Record the last used (or expected available) timestamp for each key
        self.last_used = {k: 0.0 for k in api_keys}
        self.lock = threading.Lock()

    def get_key_and_wait(self):
        """Thread-safe acquisition of an idle Key. Waits in queue if the 1.2s cooldown is not met"""
        if not self.api_keys: return None
        
        with self.lock:
            # Find the earliest available Key
            oldest_key = min(self.last_used, key=self.last_used.get)
            now = time.time()
            elapsed = now - self.last_used[oldest_key]
            wait_time = max(0, 2 - elapsed)
            
            # Project the next available time for this Key into the future
            self.last_used[oldest_key] = now + wait_time
            
        # Sleep outside the lock to avoid blocking other threads
        if wait_time > 0:
            time.sleep(wait_time)
            
        return oldest_key


class CiteAgent:
    # s2_api_keys accepts a list of keys
    def __init__(self, agent_model_path=None, tool1_model_path=None, tool2_model_path=None, device=None, s2_api_keys=None):
        self.s2_limiter = S2RateLimiter(s2_api_keys or [])
        self.search_cache = {}  

        print("\n[*] Connecting to Tool 1 (SFT Marking Engine)...")
        self.locator_worker = CiteModelWorkers(port=8001, model_name="tool1")
        
        print("\n[*] Connecting to Tool 2 (GRPO Intent Engine)...")
        self.intent_worker = CiteModelWorkers(port=8002, model_name="tool2")
        
        print("\n[🧠] Connecting to Brain (Qwen Instruct)...")
        self.brain_worker = CiteModelWorkers(port=8003, model_name="brain")

        # =======================================================
        # 1. Strict Tool JSON Schema
        # =======================================================
        self.tools_schema = [
            {
                "name": "mark_citations",
                "description": "Step 1: Scan the raw text and insert [#CITE#] markers where academic citations are needed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The raw academic text"}
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "analyze_intents",
                "description": "Step 2: Analyze the text with [#CITE#] markers to extract citation intents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "marked_text": {"type": "string", "description": "The text containing [#CITE#] markers"}
                    },
                    "required": ["marked_text"]
                }
            },
            {
                "name": "search_db_single",
                "description": "Step 3 (Loop): Search Semantic Scholar for ONE specific citation index.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer", "description": "The current marker index being processed"},
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Array of keywords. Put the MOST critical entity or dataset name FIRST. We will only use the first 1-2 terms to prevent over-specification!"
                        }
                    },
                    "required": ["index", "keywords"]
                }
            },
            {
                "name": "evaluate_and_submit_single",
                "description": "Step 4 (Loop): Submit the chosen paper ID for the CURRENT index.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer", "description": "The current marker index"},
                        "selected_id": {"type": "string", "description": "The best paper ID (or null if none match)"}
                    },
                    "required": ["index", "selected_id"]
                }
            },
            {
                "name": "finalize_chunk",
                "description": "Step 5: Call this ONLY when ALL indices extracted by analyze_intents have been successfully submitted.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        ]

    # =======================================================
    # 2. Independent Tool Implementation (Semantic Scholar + Rate Limiting)
    # =======================================================
    def _call_semantic_scholar_api(self, query, max_retries=3):
        """Call S2 API to get metadata including TLDR, citation count, and BibTeX"""
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": 8, 
            "fields": "paperId,title,authors,tldr,citationCount,citationStyles"
        }

        for attempt in range(max_retries):
            try:
                # Fetch key from queue lock, handling the cooldown automatically
                api_key = self.s2_limiter.get_key_and_wait()
                headers = {"x-api-key": api_key.strip()} if api_key else {}
                
                # Replaced fixed sleep with lock-managed time control
                response = requests.get(url, params=params, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    return response.json().get('data', [])
                elif response.status_code == 429:
                    wait_time = 2 ** (attempt + 1)
                    print(f"  ⚠️ API Rate Limit (429) triggered, retrying in {wait_time}s [{attempt+1}/{max_retries}]...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Log invalid Key and error message
                    safe_key = f"{api_key[:6]}***" if api_key else "None"
                    print(f"\n  [S2 API Error] HTTP {response.status_code} | Key: {safe_key} | Response: {response.text}")
                    break
            except Exception as e:
                print(f"  Network Error: {e}, retrying...")
                time.sleep(2)
                
        return []

    def _tool_mark_citations(self, text):
        res = self.locator_worker.locate_citations(text)
        return res.replace("【Marked Text】:\n", "").strip()

    def _tool_analyze_intents(self, marked_text):
        res = self.intent_worker.analyze_intents(marked_text)
        try:
            return json.dumps(res, ensure_ascii=False)
        except:
            return str(res)

    def _tool_search_db_single(self, index, keywords):
        if not keywords:
            return f"Error: No keywords provided for index {index}."
            
        core_keywords = keywords[:1]
        search_query = " ".join(core_keywords)
        
        # print(f"      [Search Opt] Original keywords: {keywords}")
        # print(f"      [Actual Request] Sending to API: '{search_query}'")
        
        papers = self._call_semantic_scholar_api(search_query)
        
        result_context = f"\n--- Candidate Papers for [Index {index}] ---\n"
        if not papers:
            result_context += "No candidates found on Semantic Scholar. (Consider refining keywords if needed)\n"
        else:
            for p in papers:
                p_id = p.get('paperId', 'Unknown')
                title = p.get('title', 'No Title')
                
                citations = p.get('citationCount', 0)
                
                tldr_data = p.get('tldr')
                tldr_text = tldr_data.get('text') if tldr_data else "No TLDR available."
                
                authors_list = p.get('authors', [])
                author_names = [a.get('name', 'Unknown') for a in authors_list[:3]]
                author_str = ", ".join(author_names) + (" et al." if len(authors_list) > 3 else "")
                
                self.search_cache[p_id] = p
                
                result_context += f"- ID: {p_id}\n  Title: {title}\n  Authors: {author_str}\n  Citations: {citations}\n  TLDR: {tldr_text}\n\n"
                
        return result_context.strip()

    def _format_author_name(self, author_str):
        if not author_str: return "Unknown"
        parts = author_str.strip().split()
        if len(parts) == 1: return parts[0]
        return f"{parts[-1]}, {parts[0][0].upper()}."

    # =======================================================
    # 3. Brain System Prompt 
    # =======================================================
    def _get_system_prompt(self):
        return f"""You are an Autonomous Academic Agent capable of reflection and self-correction. You process text by calling functions.

You have access to the following functions:
{json.dumps(self.tools_schema, indent=2)}

CRITICAL RULES:
1. Sequential Workflow: mark_citations -> analyze_intents -> (Loop starts) -> search_db_single (Index 0) -> evaluate_and_submit_single (Index 0) -> search_db_single (Index 1) ... -> finalize_chunk.
2. You MUST process each index one by one. Do NOT search for Index 1 until you have submitted Index 0.
3. Once all indices reported by `analyze_intents` are submitted, call `finalize_chunk`.

OUTPUT FORMAT:
Every response MUST consist of two parts:
1. A reasoning process wrapped in <think>...</think> tags.
2. A function call wrapped in <act>...</act> tags.
Example:
<think>
[Your reasoning here]
</think>
<act>
{{"name": "...", "arguments": {{...}}}}
</act>
"""

    # =======================================================
    # 4. Function Call Core State Machine Loop
    # =======================================================
    def _run_single_chunk(self, chunk_text, max_steps=40):
        # print(f"\n  [Agent Activated] Brain has taken control, starting autonomous planning...")
        
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": f"Please process this text:\n{chunk_text}"}
        ]
        
        current_marked_text = chunk_text 
        citations_dict = {}
        total_indices_expected = 0
        indices_submitted = 0

        for step in range(1, max_steps + 1):
            # print(f"   [Round {step}/{max_steps}] Brain is thinking and deciding...")
            
            prompt = ""
            for msg in messages:
                prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n<think>\n" 
            
            raw_response = self.brain_worker._generate(prompt, max_tokens=1024)
            response = "<think>\n" + raw_response 
            
            # Disabled large outputs during multi-threading to prevent console spam. Uncomment to debug.
            # print("\n" + "✨ " + "━"*20 + " [Brain Thought & Action] " + "━"*20 + " ✨")
            
            think_match = re.search(r'<think>(.*?)(?:</think>|$)', response, re.DOTALL)
            # if think_match:
            #     content = think_match.group(1).strip()
            #     tag = "[THINK]" if "{" not in content else " [THINK anomaly contains JSON]"
            #     print(f"{tag}\n{content}\n" + "━"*55)
            
            act_match = re.search(r'<act>(.*?)(?:</act>|$)', response, re.DOTALL)
            if act_match:
                pass # print(f"[ACT]\n{act_match.group(1).strip()}\n")
            elif "{" in response and "<act>" not in response:
                print(f"[Missing ACT tag, raw output]:\n{response.split('</think>')[-1].strip()}\n")
            
            # print("✨ " + "━"*56 + " ✨\n")
            # ---------------------------------------------
            messages.append({"role": "assistant", "content": response})
            
            act_match = re.search(r'<act>\s*({.*?})\s*</act>', response, re.DOTALL)
            
            if act_match:
                try:
                    call_str = act_match.group(1).strip()
                    call_str = re.sub(r'```json|```', '', call_str).strip()
                    call_data = json.loads(call_str)
                    func_name = call_data.get("name")
                    args = call_data.get("arguments", {})
                    
                    # print(f"    Brain decides to call API: [{func_name}]")
                    
                    if func_name == "mark_citations":
                        obs = self._tool_mark_citations(args.get("text", chunk_text))
                        current_marked_text = obs
                        # print(f"       Success: SFT marking completed.")
                        if "[#CITE#]" not in obs:
                            # print("      ℹ️ Info: No citation required, terminating early.")
                            return {"marked_text": chunk_text, "citations": {}}
                        messages.append({"role": "user", "content": f"Result from mark_citations:\n{obs}"})
                        
                    elif func_name == "analyze_intents":
                        obs_json_str = self._tool_analyze_intents(args.get("marked_text", current_marked_text))
                        try:
                            intents_list = json.loads(obs_json_str)
                            total_indices_expected = len(intents_list)
                            # print(f"       Success: Found {total_indices_expected} citation intents.")
                        except: pass
                        messages.append({"role": "user", "content": f"Result from analyze_intents:\n{obs_json_str}"})
                        
                    elif func_name == "search_db_single":
                        idx = args.get("index")
                        kws = args.get("keywords", [])
                        obs = self._tool_search_db_single(idx, kws)
                        # print(f"       Success: Database returned candidates.")
                        messages.append({"role": "user", "content": f"Result from search_db_single:\n{obs}"})
                        
                    elif func_name == "evaluate_and_submit_single":
                        idx = args.get("index")
                        p_id = args.get("selected_id")
                        # print(f"       Submitted Index {idx} choice: ID {p_id}")
                        
                        if p_id is not None and str(p_id).lower() != "null":
                            matched_paper = self.search_cache.get(str(p_id))
                            if matched_paper:
                                title = matched_paper.get('title', 'Unknown Title')
                                bibtex_data = matched_paper.get('citationStyles', {})
                                bibtex = bibtex_data.get('bibtex')
                                
                                if bibtex:
                                    citations_dict[idx] = bibtex.strip()
                                    # print(f"       [Index {idx}] Success: Extracted native BibTeX")
                                else:
                                    authors_list = matched_paper.get('authors', [])
                                    author_str = authors_list[0].get('name') if authors_list else "Unknown Author"
                                    citations_dict[idx] = f"{author_str} et al., \"{title}\" (BibTeX Not Available)"
                                    # print(f"       [Index {idx}] Success: {title[:30]}...")
                            else:
                                citations_dict[idx] = f"Unknown Source (ID {p_id} not in cache)"
                        else:
                            citations_dict[idx] = "Unknown Source (Agent skipped or passed null)"
                            
                        indices_submitted += 1
                        obs = f"Submission for index {idx} recorded. {total_indices_expected - indices_submitted} indices remaining."
                        messages.append({"role": "user", "content": f"Result from evaluate_and_submit_single:\n{obs}"})
                        
                    elif func_name == "finalize_chunk":
                        # print("       Brain announced current Chunk is processed!")
                        return {"marked_text": current_marked_text, "citations": citations_dict}
                        
                    else:
                        print(f"   Hallucination call: {func_name}")
                        messages.append({"role": "user", "content": f"Result from {func_name}:\nSystem Error: Function '{func_name}' does not exist."})
                
                except Exception as e:
                    print(f"   JSON Parsing Error: {e}")
                    messages.append({"role": "user", "content": f"JSON Parse Error: {e}. You MUST output valid JSON inside <act>."})
            else:
                print("   Brain output missing <act>, correcting...")
                messages.append({"role": "user", "content": "Your response must end with an <act> tag containing a JSON function call. Please continue."})

        return {"marked_text": current_marked_text, "citations": citations_dict}

    # =======================================================
    # 5. Global Document Scheduler 
    # =======================================================
    def process_document(self, long_text):
        # Disabled large outputs during multi-threading to prevent console spam
        # print("\n" + "="*60)
        # print(" Started Global Document Processing Engine (Semantic Scholar API)")
        # print("="*60)
        
        paragraphs = [p.strip() for p in long_text.split('\n') if p.strip()]
        # print(f"[*] Document split into {len(paragraphs)} Chunks.")
        
        global_cite_counter = 1
        final_merged_paragraphs = []
        global_bibliography = []
        
        for i, para in enumerate(paragraphs):
            # print(f"\n" + "-"*50)
            # print(f" Processing Chunk [{i+1}/{len(paragraphs)}]")
            # print("-" * 50)
            
            result = self._run_single_chunk(para)
            marked_text = result["marked_text"]
            local_citations = result["citations"]
            
            cite_count = marked_text.count("[#CITE#]")
            for local_idx in range(cite_count):
                marked_text = marked_text.replace("[#CITE#]", f"[{global_cite_counter}]", 1)
                cite_str = local_citations.get(local_idx, "Unknown Source (System failed to resolve)")
                global_bibliography.append(f"[{global_cite_counter}]\n{cite_str}\n")
                global_cite_counter += 1
                
            final_merged_paragraphs.append(marked_text)
            
        final_article = "\n\n".join(final_merged_paragraphs)
        
        # print("\n" + ""*30)
        # print(" Document processed! Final merged output:")
        # print(""*30 + "\n")
        # print("[Main Text]")
        # print(final_article)
        # print("\n[References (BibTeX)]")
        # for bib in global_bibliography:
        #     print(bib)
            
        return final_article, global_bibliography
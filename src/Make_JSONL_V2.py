"""Generate RAFT-style instruction-tuning data from the champion corpus via Gemini."""
import os
import json
import time
import random
import requests
from tqdm import tqdm

API_KEY = "YOUR_GEMINI_API_KEY"
INPUT_FILE = "lol_knowledge_corpus.txt"
OUTPUT_FILE = "lora_training_data_v2.jsonl"
MODEL_NAME = "gemini-2.5-pro"

CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200
MAX_CHUNKS_PER_CHAMP = 3
QA_PER_CHUNK = 4

RAFT_RATIO = 0.6
N_DISTRACTORS = 3
ORACLE_KEEP_P = 0.8
REQUEST_DELAY = 0.5


def parse_corpus(filepath):
    champions, cur_title, cur_text = [], None, []
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("CHAMPION TITLE:"):
                if cur_title and cur_text:
                    champions.append({"title": cur_title, "text": "".join(cur_text).strip()})
                cur_title, cur_text = line.replace("CHAMPION TITLE:", "").strip(), []
            elif not line.startswith("==="):
                cur_text.append(line)
    if cur_title and cur_text:
        champions.append({"title": cur_title, "text": "".join(cur_text).strip()})
    return champions


def chunk_text(text, size, overlap):
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start:start + size].strip()
        if len(chunk) > 300:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def call_gemini(prompt_text):
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{MODEL_NAME}:generateContent?key={API_KEY}")
    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {"response_mime_type": "application/json", "temperature": 0.7},
    }
    for _ in range(3):
        try:
            res = requests.post(url, json=payload, timeout=120).json()
            if "error" in res:
                if res["error"].get("code") in (429, 503):
                    time.sleep(5)
                    continue
                return []
            txt = res["candidates"][0]["content"]["parts"][0]["text"].strip()
            if txt.startswith("```json"):
                txt = txt[7:]
            elif txt.startswith("```"):
                txt = txt[3:]
            if txt.endswith("```"):
                txt = txt[:-3]
            return json.loads(txt.strip())
        except Exception:
            time.sleep(2)
    return []


def generate_qa(champion, chunk):
    prompt = f"""You are a League of Legends expert creating training data.
Read the CONTENT about champion "{champion}" and write {QA_PER_CHUNK} question-answer pairs in English.

Rules:
- Focus on concrete mechanics: ability effects, cooldowns, scaling, power spikes, item synergies, role, matchups.
- Each "output" must briefly reason from the content, then state the conclusion.
- Use only facts present in the CONTENT.

Return only a JSON array:
[{{"instruction": "...", "input": "", "output": "..."}}]

CONTENT ({champion}):
{chunk}
"""
    out = call_gemini(prompt)
    return [q for q in out if isinstance(q, dict) and q.get("instruction") and q.get("output")]


def build_raft_input(oracle_chunk, chunk_pool):
    pool = [c for c in chunk_pool if c is not oracle_chunk]
    docs = random.sample(pool, k=min(N_DISTRACTORS, len(pool)))
    if random.random() < ORACLE_KEEP_P:
        docs.append(oracle_chunk)
    if not docs:
        docs = [oracle_chunk]
    random.shuffle(docs)
    return "\n\n".join(f"[Document {i + 1}]\n{d}" for i, d in enumerate(docs))


if __name__ == "__main__":
    champs = parse_corpus(INPUT_FILE)
    if not champs:
        raise SystemExit(f"No data in {INPUT_FILE}")

    champ_chunks, all_chunks = {}, []
    for c in champs:
        ch = chunk_text(c["text"], CHUNK_SIZE, CHUNK_OVERLAP)[:MAX_CHUNKS_PER_CHAMP]
        if ch:
            champ_chunks[c["title"]] = ch
            all_chunks.extend(ch)

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    total = 0
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
        for champ, chunks in tqdm(champ_chunks.items()):
            for chunk in chunks:
                for qa in generate_qa(champ, chunk):
                    qa["input"] = build_raft_input(chunk, all_chunks) if random.random() < RAFT_RATIO else ""
                    f_out.write(json.dumps(qa, ensure_ascii=False) + "\n")
                    total += 1
                f_out.flush()
                time.sleep(REQUEST_DELAY)

    print(f"{total} examples written to {OUTPUT_FILE}")

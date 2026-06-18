"""
make_jsonl_v2.py  (TP2 업그레이드판)
========================================================================
기존 make_jsonl.py 의 두 가지 약점을 고친다.

  (1) 얕음:   위키 앞 3000자만 쓰고 챔피언당 QA 3쌍 -> 풀 스킬셋/아이템/매치업 누락
              => 페이지 '전체'를 청크로 쪼개고, 청크마다 QA를 여러 개 생성한다.
              => 프롬프트에서 스킬 메커니즘/파워스파이크/아이템/카운터를 '명시적으로' 요구
                 (TP1 Table 2 의 Fiora 스킬 환각, Jax 아이템 오류를 직접 겨냥)

  (2) input 비어있음:  모델이 '검색된 문서를 보고 답하는 법'을 학습한 적이 없음
              => RAFT(arXiv:2403.10131) 방식. 학습 예시의 input 필드에
                 [정답이 들어있는 oracle 청크 + 다른 챔피언의 distractor 청크들]을 넣고,
                 답은 '근거를 짚고(CoT) 결론'을 내도록 만든다.
              => 일부 예시는 일부러 oracle 을 빼서(distractor만) 검색이 부실할 때도
                 견디게 한다. (RAFT 의 (1-P) 비율)

출력 스키마는 기존과 동일: {"instruction": ..., "input": ..., "output": ...}
=> 기존 train 노트북의 formatting_prompts_func 를 '그대로' 쓸 수 있다.
   (alpaca_prompt 가 input 을 "### Input:" 로 렌더링하므로 RAFT 컨텍스트가 자동으로 들어감)

실행 위치: 네 PC 또는 Colab 아무데서나 (인터넷 + Gemini API 키만 있으면 됨).
입력 파일:  lol_knowledge_corpus.txt  (inven_scrapper.py 가 만드는 그 파일)
출력 파일:  lora_training_data_v2.jsonl
========================================================================
"""

import os
import json
import time
import random
import requests
from tqdm import tqdm

# ====================== 설정 (여기만 만지면 됨) ======================
API_KEY      = "my_api_key"                      # <- 네 Gemini 키
INPUT_FILE   = "lol_knowledge_corpus.txt"        # <- 스크래퍼가 만든 코퍼스 (그대로 재사용)
OUTPUT_FILE  = "lora_training_data_v2.jsonl"
MODEL_NAME   = "gemini-2.5-pro"                  # 느리면 "gemini-2.5-flash" 로 (싸고 빠름)

# 데이터 양/깊이 조절 (크게 하면 데이터 많아지지만 API 호출·시간·비용 증가)
CHUNK_SIZE          = 1800   # 한 청크 글자 수
CHUNK_OVERLAP       = 200    # 청크 간 겹침 (문맥 끊김 방지)
MAX_CHUNKS_PER_CHAMP = 3     # 챔피언당 사용할 청크 수 (3이면 앞~중반 커버, 5로 올리면 더 깊게)
QA_PER_CHUNK        = 4       # 청크 1개당 생성할 QA 수

# RAFT 설정
RAFT_RATIO     = 0.6   # 전체 QA 중 'input 에 문서를 넣는' RAFT 예시 비율 (나머지는 input="")
N_DISTRACTORS  = 3     # RAFT 예시에 섞을 '다른 챔피언' 방해 문서 개수
ORACLE_KEEP_P  = 0.8   # RAFT 예시 중 정답 문서를 실제로 포함시킬 비율 (RAFT 의 P)

REQUEST_DELAY  = 0.5   # 호출 간 대기 (초). 429/503 뜨면 늘려라.
# ===================================================================


def parse_corpus(filepath):
    """CHAMPION TITLE: ... 로 구분된 코퍼스를 {title, text} 리스트로."""
    champions, cur_title, cur_text = [], None, []
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("CHAMPION TITLE:"):
                if cur_title and cur_text:
                    champions.append({"title": cur_title, "text": "".join(cur_text).strip()})
                cur_title = line.replace("CHAMPION TITLE:", "").strip()
                cur_text = []
            elif not line.startswith("==="):
                cur_text.append(line)
    if cur_title and cur_text:
        champions.append({"title": cur_title, "text": "".join(cur_text).strip()})
    return champions


def chunk_text(text, size, overlap):
    """글자 기준 슬라이딩 윈도우 청킹. 너무 짧은 꼬리는 버린다."""
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start:start + size].strip()
        if len(chunk) > 300:           # 의미 없는 짧은 조각 제외
            chunks.append(chunk)
        start += size - overlap
    return chunks


def call_gemini(prompt_text):
    """Gemini 호출 (마크다운 찌꺼기 제거 + 재시도). 실패 시 [] 반환."""
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{MODEL_NAME}:generateContent?key={API_KEY}")
    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {"response_mime_type": "application/json", "temperature": 0.7},
    }
    for attempt in range(3):
        try:
            r = requests.post(url, json=payload, timeout=120)
            res = r.json()
            if "error" in res:
                if res["error"].get("code") in (429, 503):
                    time.sleep(5); continue
                print(f"\n[API error] {res['error'].get('message')}"); return []
            txt = res["candidates"][0]["content"]["parts"][0]["text"].strip()
            if txt.startswith("```json"): txt = txt[7:]
            elif txt.startswith("```"):   txt = txt[3:]
            if txt.endswith("```"):       txt = txt[:-3]
            return json.loads(txt.strip())
        except json.JSONDecodeError:
            time.sleep(2); continue
        except Exception as e:
            print(f"\n[sys error] {e}"); time.sleep(2); continue
    return []


def generate_qa(champion, chunk):
    """청크 1개에서 QA_PER_CHUNK 개의, 근거를 짚는(CoT) QA 생성."""
    prompt = f"""You are a League of Legends expert creating training data.
Read the CONTENT about champion "{champion}" and write {QA_PER_CHUNK} high-quality
question-answer pairs IN ENGLISH.

REQUIREMENTS:
- Prefer questions about concrete mechanics: ability effects, cooldowns, scaling,
  level-up timings / power spikes, item synergies, role/class, and matchups/counters.
  (Do NOT ask vague lore questions.)
- Each "output" MUST first briefly reason from the content, then state the conclusion.
  Example style: "Because <fact from content>, <conclusion>."
- Only use facts present in the CONTENT. Do not invent skill levels or items.

Return ONLY a JSON array, no extra text:
[{{"instruction": "...", "input": "", "output": "..."}}]

CONTENT ({champion}):
{chunk}
"""
    out = call_gemini(prompt)
    # 형식 방어: dict/list 가 아니면 버림
    return [q for q in out if isinstance(q, dict) and q.get("instruction") and q.get("output")]


def build_raft_input(oracle_chunk, chunk_pool):
    """RAFT용 input 문자열: oracle(정답) 청크 + 다른 챔피언 distractor 청크들, 순서 셔플."""
    pool = [c for c in chunk_pool if c is not oracle_chunk]
    distractors = random.sample(pool, k=min(N_DISTRACTORS, len(pool)))
    include_oracle = random.random() < ORACLE_KEEP_P
    docs = distractors + ([oracle_chunk] if include_oracle else [])
    if not docs:                       # 안전장치
        docs = [oracle_chunk]
    random.shuffle(docs)
    return "\n\n".join(f"[Document {i+1}]\n{d}" for i, d in enumerate(docs))


if __name__ == "__main__":
    champs = parse_corpus(INPUT_FILE)
    if not champs:
        print(f"❌ '{INPUT_FILE}' 가 없거나 비어있음. 먼저 inven_scrapper.py 를 돌려 코퍼스를 만들어라.")
        raise SystemExit

    # 1) 모든 챔피언을 청크로 쪼개고, distractor 풀(전체 청크)도 만든다.
    champ_chunks = {}
    all_chunks = []
    for c in champs:
        ch = chunk_text(c["text"], CHUNK_SIZE, CHUNK_OVERLAP)[:MAX_CHUNKS_PER_CHAMP]
        if ch:
            champ_chunks[c["title"]] = ch
            all_chunks.extend(ch)
    print(f"🚀 {len(champ_chunks)} champions / {len(all_chunks)} chunks -> generating QA "
          f"(model: {MODEL_NAME})")

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    total = 0
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
        for champ, chunks in tqdm(champ_chunks.items()):
            for chunk in chunks:
                for qa in generate_qa(champ, chunk):
                    # RAFT 비율만큼 input(문서)을 채운다. 나머지는 input="".
                    if random.random() < RAFT_RATIO:
                        qa["input"] = build_raft_input(chunk, all_chunks)
                    else:
                        qa["input"] = ""
                    f_out.write(json.dumps(qa, ensure_ascii=False) + "\n")
                    total += 1
                f_out.flush()
                time.sleep(REQUEST_DELAY)

    print(f"\n🎉 done. {total} QA pairs -> '{OUTPUT_FILE}'")
    print(f"   (RAFT 예시 비율 ~{int(RAFT_RATIO*100)}%, distractor {N_DISTRACTORS}개)")

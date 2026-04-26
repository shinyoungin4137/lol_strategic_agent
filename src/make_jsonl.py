import os
import json
import time
import requests
from tqdm import tqdm

# ================== Settings (Pro Version) ==================
API_KEY = "my_api_key"
INPUT_FILE = "lol_knowledge_corpus.txt"
OUTPUT_FILE = "lora_training_data_gemini_pro_en.jsonl"
MODEL_NAME = "gemini-2.5-pro"


# ============================================================

def parse_corpus(filepath):
    champions_data = []
    current_title = None
    current_text = []
    if not os.path.exists(filepath):
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("CHAMPION TITLE:"):
            if current_title and current_text:
                champions_data.append({"title": current_title, "text": "".join(current_text).strip()})
            current_title = line.replace("CHAMPION TITLE:", "").strip()
            current_text = []
        elif not line.startswith("==="):
            current_text.append(line)

    if current_title and current_text:
        champions_data.append({"title": current_title, "text": "".join(current_text).strip()})

    return champions_data


def generate_qa_pairs(champion_name, context_text):
    truncated_context = context_text[:3000]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
    prompt_text = f"""
    You are a League of Legends expert. 
    Based on the following content, generate 3 pairs of high-quality player Questions and Answers (QA) in English.
    Answer strictly in a JSON array format. Do not provide any additional explanation.

    [
        {{"instruction": "Question", "input": "", "output": "Answer"}}
    ]

    Content: {champion_name}
    {truncated_context}
    """

    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.7
        }
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload)
            result = response.json()

            if 'error' in result:
                if result['error'].get('code') in [429, 503]:
                    time.sleep(5)
                    continue
                print(f"\n❌ API Error for {champion_name}: {result['error'].get('message')}")
                return []

            answer_text = result['candidates'][0]['content']['parts'][0]['text']

            # 마크다운 찌꺼기 완벽 제거
            answer_text = answer_text.strip()
            if answer_text.startswith("```json"):
                answer_text = answer_text[7:]
            elif answer_text.startswith("```"):
                answer_text = answer_text[3:]
            if answer_text.endswith("```"):
                answer_text = answer_text[:-3]
            answer_text = answer_text.strip()

            return json.loads(answer_text)

        except json.JSONDecodeError as e:
            print(f"\n⚠️ JSON Parsing Failed for {champion_name}: {e}")
            time.sleep(2)
            continue
        except Exception as e:
            print(f"\n⚠️ Unexpected System Error for {champion_name}: {e}")
            time.sleep(2)
            continue

    print(f"\n❌ Failed to generate valid data for {champion_name} after {max_retries} attempts.")
    return []


if __name__ == "__main__":
    champion_data = parse_corpus(INPUT_FILE)
    if not champion_data:
        print(f"❌ No data found in {INPUT_FILE}")
        exit()

    print(f"🚀 Starting High-Speed Data Generation (Model: {MODEL_NAME})")

    # 기존에 실패해서 만들어진 빈 파일이 있다면 삭제
    if os.path.exists(OUTPUT_FILE):
        print(f"🧹 Removing existing '{OUTPUT_FILE}' to start fresh...")
        os.remove(OUTPUT_FILE)

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
        for data in tqdm(champion_data):
            qa_pairs = generate_qa_pairs(data['title'], data['text'])
            if qa_pairs:
                for qa in qa_pairs:
                    f_out.write(json.dumps(qa, ensure_ascii=False) + "\n")
                f_out.flush()

                # Pro 요금제 전용 초고속 대기 시간
            time.sleep(0.5)

    print(f"\n🎉 Finished! The dataset has been successfully saved to '{OUTPUT_FILE}'.")
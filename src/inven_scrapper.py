import requests
from bs4 import BeautifulSoup
import time
import os
from tqdm import tqdm

# ================== 설정 ==================
BASE_URL = "https://wiki.leagueoflegends.com/en-us/"  # 한국어 위키 쓰고 싶으면 "https://wiki.leagueoflegends.com/ko-kr/" 로 변경
CATEGORY = "Category:League_of_Legends_champion"  # 영어 위키 기준 전체 챔피언 카테고리 (172개 정도)
OUTPUT_DIR = "lol_wiki_champions"
USER_AGENT = "MyLoLScraper/1.0 (your.email@example.com)"  # Fandom 규정상 User-Agent 필수
DELAY = 1.0  # 초 (너무 빠르면 차단될 수 있음)


# =========================================

def get_category_members(category):
    """카테고리 내 모든 페이지 제목 가져오기"""
    api_url = BASE_URL + "api.php"
    members = []
    cmcontinue = None

    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": 500,
            "cmprop": "title",
            "format": "json",
            "cmcontinue": cmcontinue
        }
        headers = {"User-Agent": USER_AGENT}

        resp = requests.get(api_url, params=params, headers=headers)
        data = resp.json()

        members.extend([member["title"] for member in data["query"]["categorymembers"]])

        if "continue" in data and "cmcontinue" in data["continue"]:
            cmcontinue = data["continue"]["cmcontinue"]
        else:
            break

    return members


def scrape_page_text(title):
    """한 페이지의 본문 텍스트만 추출 (parse API 사용)"""
    api_url = BASE_URL + "api.php"
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "format": "json",
        "disableeditsection": True,
        "disabletoc": True
    }
    headers = {"User-Agent": USER_AGENT}

    resp = requests.get(api_url, params=params, headers=headers)
    data = resp.json()

    if "parse" not in data or "text" not in data["parse"]:
        return f"[ERROR] {title} - 파싱 실패"

    html = data["parse"]["text"]["*"]
    soup = BeautifulSoup(html, "html.parser")

    # 불필요한 요소 제거
    for tag in soup.find_all(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    # 메인 콘텐츠만 추출
    content = soup.find("div", class_="mw-parser-output")
    if content:
        # 여러 줄 텍스트로 정리
        text = content.get_text(separator="\n", strip=True)
    else:
        text = soup.get_text(separator="\n", strip=True)

    # 너무 짧은 페이지 필터 (예: 리다이렉트)
    if len(text.strip()) < 100:
        return f"[SKIP] {title} - 내용 부족"

    return text


# ================== 실행 ==================
# ================== 실행 ==================
if __name__ == "__main__":
    # 개별 파일 대신, 이 하나의 파일에 모든 데이터를 밀어 넣습니다.
    COMBINED_FILE = "lol_knowledge_corpus.txt"

    print("🔍 [Step 1] Fetching category members...")
    pages = get_category_members(CATEGORY)
    print(f"✅ Found {len(pages)} pages.\n")

    # 파일을 'w' 모드로 열어서 기존 내용이 있다면 초기화하고 시작합니다.
    with open(COMBINED_FILE, "w", encoding="utf-8") as f:
        f.write("=== League of Legends Knowledge Base ===\n\n")

    for title in tqdm(pages, desc="Scraping progress"):
        print(f"   -> {title}")
        text = scrape_page_text(title)

        # 건너뛴 페이지나 에러가 아닌 실제 데이터만 저장합니다.
        if not text.startswith("[SKIP]") and not text.startswith("[ERROR]"):
            # 'a' (append) 모드로 파일을 열어서 기존 내용 뒤에 계속 이어 붙입니다.
            with open(COMBINED_FILE, "a", encoding="utf-8") as f:
                # 챔피언 간의 구분을 위해 구분선을 넣어줍니다.
                f.write(f"\n\n{'=' * 50}\n")
                f.write(f"CHAMPION TITLE: {title}\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(f"{text}\n")

        time.sleep(DELAY)

    print(f"\n🎉 Success! All data has been successfully merged into '{COMBINED_FILE}'.")
import requests
from bs4 import BeautifulSoup

def get_naver_product_keyword(search_keyword):
    url = f"https://search.shopping.naver.com/search/all?query={search_keyword}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    response + requests.get(url, headers+headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    #검색 결과에서 상품 제목을 가져옵니다
    item_titles = soup.select(basicList_title__3P9Q7)



    # 제목에서 텍스트만 추출하여 리스트에 담습니다
    keywords = [title.text.strip() for title in item_titles]
    
    return keywords

# 검색할 키워드 설정
search_keyword = "파이썬 책"

# 네이버에서 상품 검색 결과에서 키워드 수집
keywords = get_naver_product_keywords(search_keyword)

# 결과 출력
print("검색 키워드 관련 상품 제목:")
for keyword in keywords:
    print(keyword)


 import requests
 
from bs4 import BeautifulSoup

def get_naver_product_keywords(search_keyword):
    url = f"https://search.shopping.naver.com/search/all?query={search_keyword}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # 검색 결과에서 상품 제목을 가져옵니다
    item_titles = soup.select('.basicList_title__3P9Q7')
    
    # 제목에서 텍스트만 추출하여 리스트에 담습니다
    keywords = [title.text.strip() for title in item_titles]
    
    return keywords

def save_keywords_to_file(keywords, filename='keywords.txt'):
    with open(filename, 'w', encoding='utf-8') as f:
        for keyword in keywords:
            f.write(keyword + '\n')

# 검색할 키워드 설정
search_keyword = "파이썬 책"

# 네이버에서 상품 검색 결과에서 키워드 수집
keywords = get_naver_product_keywords(search_keyword)

# 결과를 파일에 저장
save_keywords_to_file(keywords, 'naver_keywords.txt')

print(f"{len(keywords)}개의 키워드를 파일에 저장했습니다.")

def analyze_keywords(keywords):
    keyword_counts = {}
    
    for keyword in keywords:
        if keyword in keyword_counts:
            keyword_counts[keyword] += 1
        else:
            keyword_counts[keyword] = 1
    
    sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_keywords

# 분석할 키워드 파일 읽어오기
def read_keywords_from_file(filename='naver_keywords.txt'):
    with open(filename, 'r', encoding='utf-8') as f:
        keywords = [line.strip() for line in f.readlines()]
    return keywords

# 파일에서 키워드 읽어오기
keywords = read_keywords_from_file('naver_keywords.txt')

# 키워드 분석
keyword_analysis = analyze_keywords(keywords)

# 결과 출력
print("키워드 분석 결과:")
for keyword, count in keyword_analysis:
    print(f"{keyword}: {count} 건")





                              
                              

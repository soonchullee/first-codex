import streamlit as st
import pandas as pd
import requests
import json
import plotly.graph_objects as go
from collections import Counter
import re
import datetime
import hashlib
import hmac
import base64
import time

# 여기에 새로 발급받은 네이버 검색 API용 Client ID와 Client Secret을 입력하세요.
naver_search_client_id = 'LsIevNgInDgJxnW7EEUC'
naver_search_client_secret = 'SqAhgUjGOG'

# 여기에 네이버 광고 API용 Access License, Secret Key, Customer ID를 입력하세요.
# https://manage.searchad.naver.com/
naver_ad_access_license = '0100000000b023976c548606bfd54a31b499d1066cd59544d9ace90d87de8da28d33426930'
naver_ad_secret_key = 'AQAAAACwI5dsVIYGv9VKMbSZ0QZs646RMlg7Og3b+eFMDgNt3A=='
naver_ad_customer_id = '748791'


def get_related_keywords_from_api(keyword):
    """
    네이버 검색 API를 사용하여 실제 연관 검색어를 가져오고 빈도수 및 시뮬레이션 검색수를 계산합니다.
    """
    url = 'https://openapi.naver.com/v1/search/shop.json'
    headers = {
        'X-Naver-Client-Id': naver_search_client_id,
        'X-Naver-Client-Secret': naver_search_client_secret
    }
    params = {'query': keyword, 'display': 50} # 최대 50개의 상품 검색

    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            items = data.get('items', [])
            
            # 상품 제목에서 중복 키워드 빈도수 계산
            all_keywords = []
            for item in items:
                title = item.get('title', '').replace('<b>', '').replace('</b>', '')
                
                # 불필요한 문자 제거 및 공백으로 분리
                cleaned_title = re.sub(r'\[.*?\]|\(.*?\)|\<.*?\>|\s+', ' ', title).strip()
                words = cleaned_title.split()

                for word in words:
                    # 너무 짧거나 불필요한 단어는 제외
                    if len(word) > 1 and word.lower() != keyword.lower():
                        all_keywords.append(word)

            # 키워드별 빈도수 계산
            keyword_counts = Counter(all_keywords)
            
            # DataFrame으로 변환 및 빈도수 기준 정렬
            df = pd.DataFrame(keyword_counts.items(), columns=["연관 검색어", "중복 횟수"])
            df = df.sort_values(by="중복 횟수", ascending=False)
            
            # 중복 횟수가 같으면 같은 순위를 부여
            df['순위'] = df['중복 횟수'].rank(method='min', ascending=False).astype(int)

            # '중복 횟수'를 기반으로 시뮬레이션 검색수 열 추가 (실제 데이터 아님)
            df['시뮬레이션 검색수'] = df['중복 횟수'] * 100 + (df['중복 횟수'] * 20)
            df['시뮬레이션 검색수'] = df['시뮬레이션 검색수'].astype(int)
                
            return df
        else:
            st.error(f"검색 API 호출 실패: {response.status_code} - {response.text}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"검색 API 호출 중 오류 발생: {e}")
        return pd.DataFrame()


def get_search_volume_from_api(keywords, customer_id, access_license, secret_key):
    """
    네이버 광고 API를 호출하여 키워드별 검색량을 가져옵니다.
    """
    url = "https://api.naver.com/keywordstool"
    
    timestamp = str(int(time.time() * 1000))
    signature = hmac.new(
        key=secret_key.encode('utf-8'),
        msg=f"{timestamp}.GET./keywordstool".encode('utf-8'),
        digestmod=hashlib.sha256
    ).digest()
    
    headers = {
        'Content-Type': 'application/json;charset=UTF-8',
        'X-API-KEY': access_license,
        'X-Customer': str(customer_id),
        'X-Timestamp': timestamp,
        'X-Signature': base64.b64encode(signature).decode('utf-8')
    }
    
    params = {
        'hintKeywords': ','.join(keywords),
        'showDetail': '1' # 상세 정보 포함
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"광고 API 호출 실패: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"광고 API 호출 중 오류 발생: {e}")
        return None

# Streamlit 웹페이지 레이아웃 설정
st.title("네이버 쇼핑 키워드 분석")
st.markdown("관심 있는 쇼핑 키워드의 검색량 트렌드를 확인해 보세요! 📈")

# 네이버 쇼핑 주요 카테고리 목록
categories = {
    "전체": "all",
    "패션의류": "50000000",
    "패션잡화": "50000001",
    "화장품/미용": "50000002",
    "디지털/가전": "50000003",
    "가구/인테리어": "50000004",
    "출산/육아": "50000005",
    "식품": "50000006",
    "스포츠/레저": "50000007",
    "생활/건강": "50000008",
    "여행/문화": "50000009",
    "반려동물용품": "50000010"
}

# 사용자 입력 받기
selected_category = st.selectbox('카테고리를 선택하세요', list(categories.keys()))
keyword = st.text_input('키워드를 입력하세요', '반팔티')
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input('시작일', datetime.date.today() - datetime.timedelta(days=90))
with col2:
    end_date = st.date_input('종료일', datetime.date.today())

# 기간 단위 선택
time_units = {
    "일별": "date",
    "주별": "week",
    "월별": "month"
}
selected_time_unit = st.selectbox('기간 단위를 선택하세요', list(time_units.keys()))

# '분석 시작' 버튼
if st.button('분석 시작'):
    if keyword:
        with st.spinner('데이터를 불러오는 중...'):
            # 카테고리 선택에 따라 분석할 카테고리 목록 결정
            if selected_category == "전체":
                categories_to_analyze = {k: v for k, v in categories.items() if v != 'all'}
            else:
                categories_to_analyze = {selected_category: categories[selected_category]}

            # 연관 검색어 데이터 (네이버 검색 API)
            st.markdown("---")
            st.subheader("연관 검색어 및 월간 검색수 (네이버 검색/광고 API)")
            related_df = get_related_keywords_from_api(keyword)
            if not related_df.empty:
                # 광고 API를 통해 실제 검색량 가져오기 (5개씩 나누어 요청)
                related_keywords = related_df['연관 검색어'].tolist()
                
                # 키워드 리스트를 5개씩 분할
                keyword_batches = [related_keywords[i:i + 5] for i in range(0, len(related_keywords), 5)]
                
                all_volume_data = []
                for batch in keyword_batches:
                    search_volume_data = get_search_volume_from_api(
                        batch,
                        naver_ad_customer_id,
                        naver_ad_access_license,
                        naver_ad_secret_key
                    )
                    if search_volume_data and 'keywordList' in search_volume_data:
                        all_volume_data.extend(search_volume_data['keywordList'])
                    time.sleep(1) # API 호출 사이에 1초 지연
                
                if all_volume_data:
                    # 검색량 데이터를 DataFrame으로 변환
                    volume_df = pd.DataFrame(all_volume_data)
                    
                    # 연관 검색어 DataFrame과 검색량 데이터 합치기
                    merged_df = pd.merge(related_df, volume_df, left_on='연관 검색어', right_on='relKeyword', how='left')
                    merged_df.rename(columns={'monthlyPcQcCnt': '월간 PC 검색수', 'monthlyMobileQcCnt': '월간 모바일 검색수'}, inplace=True)
                    merged_df = merged_df[['순위', '연관 검색어', '중복 횟수', '월간 PC 검색수', '월간 모바일 검색수']].sort_values(by='중복 횟수', ascending=False)
                    st.dataframe(merged_df, hide_index=True)
                else:
                    st.dataframe(related_df, hide_index=True)
                    st.warning("네이버 광고 API에서 검색량 데이터를 불러오지 못했습니다.")
            else:
                st.warning("네이버 검색 API에서 연관 검색어를 불러오지 못했습니다.")

    else:
        st.error("키워드를 입력해 주세요.")

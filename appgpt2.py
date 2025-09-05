import streamlit as st
import pandas as pd
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import re
import datetime
import hashlib
import hmac
import base64
import time
from urllib.parse import quote
import os
from typing import Dict, List, Optional, Tuple
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경변수에서 API 키 가져오기 (보안 강화)
NAVER_SEARCH_CLIENT_ID = os.getenv('NAVER_SEARCH_CLIENT_ID', 'LsIevNgInDgJxnW7EEUC')
NAVER_SEARCH_CLIENT_SECRET = os.getenv('NAVER_SEARCH_CLIENT_SECRET', 'SqAhgUjGOG')
NAVER_AD_ACCESS_LICENSE = os.getenv('NAVER_AD_ACCESS_LICENSE', '0100000000859694c937668dd701fcb3aeff2c3ab8391a20d5d5cec6ca29aade23fddb6975')
NAVER_AD_SECRET_KEY = os.getenv('NAVER_AD_SECRET_KEY', 'AQAAAAAXL8UcMJRZ+X6l38kLGJAQYW5sMh90+zSzf3umtq10cQ==')
NAVER_AD_CUSTOMER_ID = os.getenv('NAVER_AD_CUSTOMER_ID', '748791')

class NaverKeywordAnalyzer:
    """네이버 쇼핑 키워드 분석 클래스"""
    
    def __init__(self):
        self.search_headers = {
            'X-Naver-Client-Id': NAVER_SEARCH_CLIENT_ID,
            'X-Naver-Client-Secret': NAVER_SEARCH_CLIENT_SECRET
        }
        
    @st.cache_data(ttl=3600)
    def get_product_count_for_keywords(_self, keywords: List[str], batch_size: int = 5) -> Dict[str, int]:
        """키워드별 네이버 쇼핑 상품수를 조회합니다."""
        url = 'https://openapi.naver.com/v1/search/shop.json'
        product_counts = {}
        
        for i in range(0, len(keywords), batch_size):
            batch = keywords[i:i + batch_size]
            
            for keyword in batch:
                try:
                    params = {'query': keyword, 'display': 1}
                    response = requests.get(
                        url, 
                        headers=_self.search_headers, 
                        params=params, 
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        total_count = data.get('total', 0)
                        product_counts[keyword] = total_count
                    else:
                        logger.warning(f"Product count API failed for {keyword}: {response.status_code}")
                        product_counts[keyword] = 0
                        
                    time.sleep(0.1)  # API 호출 간격
                    
                except Exception as e:
                    logger.error(f"Error getting product count for {keyword}: {str(e)}")
                    product_counts[keyword] = 0
        
        return product_counts

    def clean_title(self, title: str) -> str:
        """상품 제목에서 불필요한 요소 제거"""
        # HTML 태그 제거
        title = re.sub(r'<[^>]+>', '', title)
        # 괄호 안 내용 제거
        title = re.sub(r'[\[\](){}〈〉《》「」『』【】\<\>].*?[\[\](){}〈〉《》「」『』【】\<\>]', '', title)
        # 특수문자 제거
        title = re.sub(r'[^\w\s가-힣]', ' ', title)
        # 연속된 공백 정리
        title = re.sub(r'\s+', ' ', title).strip()
        return title

    def is_valid_keyword(self, word: str, original_keyword: str) -> bool:
        """유효한 키워드인지 판단"""
        # 제외할 키워드 리스트 확장
        excluded_words = {
            '상품', '무료배송', '당일배송', '택배', '네이버', '쿠팡', '11번가',
            '배송', '무료', '할인', '세트', '개입', '개들이', '포함', '증정',
            '이벤트', '특가', '한정', '선물', '사은품', 'PC', 'TV', 'DVD'
        }
        
        return (
            len(word) >= 2 and 
            word.lower() != original_keyword.lower() and 
            not word.isdigit() and
            word not in excluded_words and
            not re.match(r'^\d+[가-힣]*$', word)  # 숫자+한글 패턴 제외 (예: 1개, 2번째 등)
        )

    def get_related_keywords(self, keyword: str, max_items: int = 100) -> pd.DataFrame:
        """네이버 검색 API를 사용하여 연관 검색어를 가져옵니다."""
        url = 'https://openapi.naver.com/v1/search/shop.json'
        params = {
            'query': keyword, 
            'display': max_items,
            'sort': 'sim'
        }

        try:
            response = requests.get(url, headers=self.search_headers, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('items', [])
                
                if not items:
                    st.warning(f"'{keyword}'에 대한 검색 결과가 없습니다.")
                    return pd.DataFrame()
                
                # 키워드 추출 및 정리
                all_keywords = []
                for item in items:
                    title = item.get('title', '')
                    cleaned_title = self.clean_title(title)
                    words = cleaned_title.split()

                    for word in words:
                        if self.is_valid_keyword(word, keyword):
                            all_keywords.append(word)

                # 키워드별 빈도수 계산
                keyword_counts = Counter(all_keywords)
                
                if not keyword_counts:
                    st.warning("유의미한 연관 검색어를 찾을 수 없습니다.")
                    return pd.DataFrame()
                
                # DataFrame으로 변환
                df = pd.DataFrame(keyword_counts.items(), columns=["연관 검색어", "언급 횟수"])
                
                # 데이터 타입 안전하게 변환
                df['언급 횟수'] = pd.to_numeric(df['언급 횟수'], errors='coerce').fillna(1)
                df['언급 횟수'] = df['언급 횟수'].astype(int)
                
                # 정렬 및 순위 추가
                df = df.sort_values(by="언급 횟수", ascending=False).head(40)
                df = df[df['언급 횟수'] > 0].reset_index(drop=True)  # 0보다 큰 값만 필터링
                df['순위'] = range(1, len(df) + 1)
                
                return df[['순위', '연관 검색어', '언급 횟수']]
                
            elif response.status_code == 429:
                st.error("API 호출 한도를 초과했습니다. 잠시 후 다시 시도해주세요.")
            else:
                st.error(f"검색 API 호출 실패: {response.status_code}")
                
        except requests.exceptions.Timeout:
            st.error("API 호출 시간이 초과되었습니다.")
        except Exception as e:
            st.error(f"검색 API 호출 중 오류 발생: {str(e)}")
            logger.error(f"API Error Details: {str(e)}")
            
        return pd.DataFrame()

    def create_signature(self, timestamp: str, method: str, uri: str) -> str:
        """네이버 광고 API 시그니처 생성"""
        signature_string = f"{timestamp}.{method}.{uri}"
        signature = hmac.new(
            key=NAVER_AD_SECRET_KEY.encode('utf-8'),
            msg=signature_string.encode('utf-8'),
            digestmod=hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode('utf-8')

    def get_search_volume(self, keywords: str) -> Optional[Dict]:
        """네이버 광고 API를 호출하여 검색량을 가져옵니다."""
        # API 권한 확인을 위한 체크
        if not all([NAVER_AD_ACCESS_LICENSE, NAVER_AD_SECRET_KEY, NAVER_AD_CUSTOMER_ID]):
            st.warning("네이버 광고 API 설정이 완료되지 않았습니다. 검색량 데이터는 제외하고 분석을 진행합니다.")
            return None
            
        base_url = "https://api.searchad.naver.com"
        uri = "/keywordstool"
        method = "GET"
        
        timestamp = str(int(time.time() * 1000))
        signature = self.create_signature(timestamp, method, uri)
        
        headers = {
            'Content-Type': 'application/json; charset=UTF-8',
            'X-API-KEY': NAVER_AD_ACCESS_LICENSE,
            'X-Customer': str(NAVER_AD_CUSTOMER_ID),
            'X-Timestamp': timestamp,
            'X-Signature': signature
        }
        
        params = {
            'hintKeywords': keywords,
            'showDetail': '1'
        }
        
        try:
            full_url = base_url + uri
            response = requests.get(full_url, headers=headers, params=params, timeout=20)
            
            if response.status_code == 200:
                return response.json()
            else:
                self._handle_api_error(response)
                # 403 에러 시에는 None 반환하여 검색량 없이 진행
                if response.status_code == 403:
                    st.info("광고 API 권한 문제로 검색량 데이터 없이 분석을 계속 진행합니다.")
                    return None
                
        except requests.exceptions.Timeout:
            st.error("API 호출 시간 초과 (20초)")
        except Exception as e:
            st.error(f"예상치 못한 오류: {str(e)}")
            
        return None

    def _handle_api_error(self, response):
        """API 에러 처리"""
        # 403 에러는 조용히 처리 (중복 메시지 방지)
        if response.status_code == 403:
            # 세션 상태로 한 번만 표시
            if 'api_error_shown' not in st.session_state:
                st.session_state.api_error_shown = True
                # 조용히 로그에만 기록
                logger.warning("Naver Ad API 403 permission error")
            return
            
        error_messages = {
            401: "인증 실패: API 키나 시크릿 키를 확인해주세요.",
            429: "호출 한도 초과: 잠시 후 다시 시도해주세요.",
            400: "잘못된 요청: 파라미터를 확인해주세요."
        }
        
        st.error(f"광고 API 호출 실패 (상태 코드: {response.status_code})")
        if response.status_code in error_messages:
            st.error(error_messages[response.status_code])
        else:
            st.error(f"응답 내용: {response.text[:200]}")

def create_enhanced_chart(df: pd.DataFrame) -> go.Figure:
    """향상된 연관 검색어 차트 생성"""
    if df.empty:
        return None
        
    top_15 = df.head(15)
    
    # 색상 그라데이션 생성
    if len(top_15) > len(px.colors.sequential.Blues_r):
        colors = px.colors.sequential.Blues_r
    else:
        colors = px.colors.sequential.Blues_r[:len(top_15)]
    
    fig = go.Figure(data=[
        go.Bar(
            x=top_15['언급 횟수'],
            y=top_15['연관 검색어'],
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(50, 50, 50, 0.8)', width=1)
            ),
            text=top_15['언급 횟수'],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>언급 횟수: %{x}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': '연관 검색어 언급 빈도 (Top 15)',
            'x': 0.5,
            'font': {'size': 18}
        },
        xaxis_title='언급 횟수',
        yaxis_title='연관 검색어',
        height=500,
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial", size=12),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_comparison_chart(df: pd.DataFrame) -> go.Figure:
    """검색량과 언급횟수 비교 차트"""
    if df.empty or '총 월간검색수' not in df.columns:
        return None
    
    # 안전한 데이터 필터링
    try:
        # 숫자형 데이터로 변환하여 비교
        df_clean = df.copy()
        df_clean['총 월간검색수'] = pd.to_numeric(df_clean['총 월간검색수'], errors='coerce').fillna(0)
        df_clean['언급 횟수'] = pd.to_numeric(df_clean['언급 횟수'], errors='coerce').fillna(0)
        
        valid_data = df_clean[(df_clean['총 월간검색수'] > 0) & (df_clean['언급 횟수'] > 0)].head(10)
        
        if valid_data.empty:
            return None
        
        fig = go.Figure()
        
        # 언급 횟수
        fig.add_trace(go.Scatter(
            x=valid_data['연관 검색어'],
            y=valid_data['언급 횟수'],
            mode='markers+lines',
            name='언급 횟수',
            yaxis='y',
            marker=dict(size=10, color='blue'),
            line=dict(color='blue', width=2)
        ))
        
        # 검색량
        fig.add_trace(go.Scatter(
            x=valid_data['연관 검색어'],
            y=valid_data['총 월간검색수'],
            mode='markers+lines',
            name='월간 검색수',
            yaxis='y2',
            marker=dict(size=10, color='red'),
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title='언급 횟수 vs 월간 검색수 비교',
            xaxis_title='연관 검색어',
            yaxis=dict(title='언급 횟수', side='left', color='blue'),
            yaxis2=dict(title='월간 검색수', side='right', overlaying='y', color='red'),
            height=400,
            hovermode='x unified',
            legend=dict(x=0.01, y=0.99)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Chart creation error: {str(e)}")
        return None

def safe_format_number(x):
    """안전한 숫자 포맷팅"""
    try:
        if pd.isna(x) or x is None or x == "" or x == 0:
            return "-"
        
        # 문자열로 변환 후 숫자인지 확인
        x_str = str(x)
        if x_str.replace('.', '').replace('-', '').isdigit():
            num_val = float(x)
            if num_val > 0:
                return f"{int(num_val):,}"
        return "-"
    except (ValueError, TypeError, OverflowError):
        return "-"

def _merge_and_clean_data(related_df, volume_data, product_counts):
    """데이터 병합 및 정리"""
    try:
        # 검색량 데이터 변환
        if volume_data:
            volume_df = pd.DataFrame(volume_data)
            merged_df = pd.merge(related_df, volume_df, 
                               left_on='연관 검색어', right_on='relKeyword', how='left')
        else:
            merged_df = related_df.copy()
        
        # 상품수 데이터 병합
        if product_counts:
            product_count_df = pd.DataFrame(list(product_counts.items()), 
                                          columns=['연관 검색어', '상품수'])
            merged_df = pd.merge(merged_df, product_count_df, on='연관 검색어', how='left')
        
        # 컬럼명 정리
        column_mapping = {
            'monthlyPcQcCnt': 'PC 월간검색수',
            'monthlyMobileQcCnt': '모바일 월간검색수',
            'compIdx': '경쟁강도'
        }
        merged_df.rename(columns=column_mapping, inplace=True)
        
        # 숫자 컬럼 안전하게 변환
        numeric_columns = ['PC 월간검색수', '모바일 월간검색수', '상품수', '경쟁강도']
        for col in numeric_columns:
            if col in merged_df.columns:
                # 문자열을 숫자로 안전하게 변환
                merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(0)
                if col != '경쟁강도':  # 경쟁강도는 소수점 유지
                    merged_df[col] = merged_df[col].astype(int)
        
        # 기본 컬럼들도 안전하게 변환
        if '언급 횟수' in merged_df.columns:
            merged_df['언급 횟수'] = pd.to_numeric(merged_df['언급 횟수'], errors='coerce').fillna(1).astype(int)
        if '순위' in merged_df.columns:
            merged_df['순위'] = pd.to_numeric(merged_df['순위'], errors='coerce').fillna(0).astype(int)
        
        # 총 검색수 계산
        if 'PC 월간검색수' in merged_df.columns and '모바일 월간검색수' in merged_df.columns:
            merged_df['총 월간검색수'] = merged_df['PC 월간검색수'] + merged_df['모바일 월간검색수']
        
        return merged_df
        
    except Exception as e:
        logger.error(f"Data merge error: {str(e)}")
        # 에러 발생 시 기본 데이터프레임 반환
        return related_df.copy()

def _display_card_view(df, columns):
    """카드형 보기 - 스크롤 없이 보기 편한 형태"""
    # 중복 제거 및 정렬
    df_clean = df.drop_duplicates(subset=['연관 검색어'], keep='first').copy()
    df_clean = df_clean.sort_values('언급 횟수', ascending=False).reset_index(drop=True)
    df_clean['순위'] = range(1, len(df_clean) + 1)
    
    # 페이지네이션 설정 - 20개씩 표시
    items_per_page = 20
    total_pages = max(1, (len(df_clean) - 1) // items_per_page + 1)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # 페이지 번호를 1부터 시작하도록 수정
        if 'card_current_page' not in st.session_state:
            st.session_state.card_current_page = 1
            
        current_page = st.selectbox(
            "페이지 선택",
            range(1, total_pages + 1),
            index=st.session_state.card_current_page - 1,
            format_func=lambda x: f"페이지 {x} ({(x-1)*items_per_page + 1}-{min(x*items_per_page, len(df_clean))})",
            key="card_page_selector"
        )
        
        # 페이지 변경 시 세션 상태 업데이트
        st.session_state.card_current_page = current_page
    
    # 현재 페이지 데이터
    start_idx = (current_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(df_clean))
    page_data = df_clean.iloc[start_idx:end_idx]
    
    # 카드 형태로 표시 - 2열 레이아웃으로 조정
    for i in range(0, len(page_data), 2):
        cols = st.columns(2)
        
        for j, col in enumerate(cols):
            if i + j < len(page_data):
                row = page_data.iloc[i + j]
                
                with col:
                    # 컴팩트한 카드 디자인
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 0.8rem; border-radius: 0.4rem; margin-bottom: 0.5rem; border: 1px solid #e9ecef;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <h5 style="margin: 0; color: #1f77b4; font-size: 1.1rem;">#{row['순위']} {row['연관 검색어']}</h5>
                                <p style="margin: 0.3rem 0 0 0; font-size: 0.9rem; color: #666;">
                                    언급 <strong>{row['언급 횟수']}</strong>회
                                </p>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # 추가 지표들을 작은 텍스트로 표시
                    metrics_text = []
                    if '총 월간검색수' in row and pd.notna(row['총 월간검색수']) and row['총 월간검색수'] > 0:
                        metrics_text.append(f"검색량: {int(row['총 월간검색수']):,}")
                    if '상품수' in row and pd.notna(row['상품수']) and row['상품수'] > 0:
                        metrics_text.append(f"상품: {int(row['상품수']):,}개")
                    if '경쟁강도' in row and pd.notna(row['경쟁강도']):
                        competition_level = "높음" if row['경쟁강도'] > 70 else "보통" if row['경쟁강도'] > 40 else "낮음"
                        metrics_text.append(f"경쟁: {competition_level}")
                    
                    if metrics_text:
                        st.markdown(f"""
                        <div style="font-size: 0.8rem; color: #888; margin-top: 0.3rem;">
                            {' | '.join(metrics_text)}
                        </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("</div>", unsafe_allow_html=True)
    
    # 페이지 정보 표시
    st.info(f"총 {len(df_clean)}개 중 {start_idx + 1}-{end_idx}번째 표시 | 페이지 {current_page}/{total_pages}")

def _display_table_view(df, columns):
    """개선된 테이블 보기 - 필터링 및 정렬 기능 포함"""
    try:
        # 중복 제거 및 정렬
        df_clean = df.drop_duplicates(subset=['연관 검색어'], keep='first').copy()
        
        # 데이터 타입 안전성 확보
        if '언급 횟수' in df_clean.columns:
            df_clean['언급 횟수'] = pd.to_numeric(df_clean['언급 횟수'], errors='coerce').fillna(1).astype(int)
        
        # 정렬 시 데이터 타입 확보
        df_clean = df_clean.sort_values('언급 횟수', ascending=False).reset_index(drop=True)
        df_clean['순위'] = range(1, len(df_clean) + 1)
        
        st.markdown("#### 필터 및 정렬 옵션")
        
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            # 언급횟수 필터 - 안전한 min/max 값 계산
            try:
                min_val = max(1, int(df_clean['언급 횟수'].min())) if len(df_clean) > 0 else 1
                max_val = max(1, int(df_clean['언급 횟수'].max())) if len(df_clean) > 0 else 1
                
                # min_val과 max_val이 같은 경우 처리
                if min_val == max_val:
                    min_mentions = min_val
                    st.write(f"언급 횟수: {min_val}")
                else:
                    min_mentions = st.slider(
                        "최소 언급 횟수",
                        min_value=min_val,
                        max_value=max_val,
                        value=min_val,
                        key="mentions_filter"
                    )
            except Exception as e:
                logger.error(f"Slider error: {str(e)}")
                min_mentions = 1
                st.write("언급 횟수 필터를 설정할 수 없습니다.")
        
        with filter_col2:
            # 검색어 필터
            search_term = st.text_input("검색어 필터", placeholder="특정 키워드 검색...", key="search_filter")
        
        with filter_col3:
            # 정렬 옵션
            sort_options = {
                '언급 횟수 (내림차순)': ('언급 횟수', False),
                '언급 횟수 (오름차순)': ('언급 횟수', True),
                '순위 (오름차순)': ('순위', True),
                '가나다순': ('연관 검색어', True)
            }
            
            if '총 월간검색수' in df_clean.columns:
                sort_options.update({
                    '검색수 (내림차순)': ('총 월간검색수', False),
                    '검색수 (오름차순)': ('총 월간검색수', True)
                })
            
            sort_by = st.selectbox("정렬 기준", list(sort_options.keys()), key="sort_option")
        
        # 필터 적용 - 안전한 비교
        try:
            filtered_df = df_clean[df_clean['언급 횟수'] >= min_mentions].copy()
        except Exception as e:
            logger.error(f"Filter error: {str(e)}")
            filtered_df = df_clean.copy()
        
        if search_term:
            try:
                filtered_df = filtered_df[
                    filtered_df['연관 검색어'].str.contains(search_term, case=False, na=False)
                ]
            except Exception as e:
                logger.error(f"Search filter error: {str(e)}")
        
        # 정렬 적용 - 안전한 정렬
        try:
            sort_column, ascending = sort_options[sort_by]
            if sort_column in filtered_df.columns:
                # 정렬 컬럼 타입 확보
                if sort_column in ['언급 횟수', '순위', '총 월간검색수']:
                    filtered_df[sort_column] = pd.to_numeric(filtered_df[sort_column], errors='coerce').fillna(0)
                
                filtered_df = filtered_df.sort_values(sort_column, ascending=ascending).reset_index(drop=True)
                # 정렬 후 순위 재조정
                if sort_column == '언급 횟수' and not ascending:
                    filtered_df['순위'] = range(1, len(filtered_df) + 1)
        except Exception as e:
            logger.error(f"Sort error: {str(e)}")
            pass

        # 페이지네이션 - 20개씩 표시
        items_per_page = 20
        
        if len(filtered_df) == 0:
            st.warning("필터 조건에 맞는 데이터가 없습니다.")
            return
        
        total_pages = max(1, (len(filtered_df) - 1) // items_per_page + 1)
        
        if total_pages > 1:
            if 'table_current_page' not in st.session_state:
                st.session_state.table_current_page = 1
                
            current_page = st.selectbox(
                f"페이지 선택 (총 {total_pages}페이지)",
                range(1, total_pages + 1),
                index=st.session_state.table_current_page - 1,
                key="table_page_selector"
            )
            st.session_state.table_current_page = current_page
            
            start_idx = (current_page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, len(filtered_df))
            page_data = filtered_df.iloc[start_idx:end_idx].copy()
        else:
            page_data = filtered_df.copy()
            start_idx, end_idx = 0, len(filtered_df)
        
        st.markdown(f"#### 결과: {len(filtered_df)}개 키워드 발견")
        styled_df = page_data[columns].copy()

        def fmt_num(x):
            try:
                v = float(x)
                return f"{int(v):,}" if v > 0 else "-"
            except Exception:
                return "-"

        for col in ['총 월간검색수', 'PC 월간검색수', '모바일 월간검색수', '상품수']:
            if col in styled_df.columns:
                styled_df[col] = styled_df[col].apply(fmt_num)
        
        if '경쟁강도' in styled_df.columns:
            def format_competition(x):
                try:
                    if pd.isna(x) or float(x) == 0:
                        return "-"
                    comp_value = float(x)
                    if comp_value >= 80:
                        return f"매우높음({comp_value:.0f})"
                    elif comp_value >= 60:
                        return f"높음({comp_value:.0f})"
                    elif comp_value >= 40:
                        return f"보통({comp_value:.0f})"
                    elif comp_value >= 20:
                        return f"낮음({comp_value:.0f})"
                    else:
                        return f"매우낮음({comp_value:.0f})"
                except Exception:
                    return "-"
            styled_df['경쟁강도'] = styled_df['경쟁강도'].apply(format_competition)
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=600)
        
        if total_pages > 1:
            st.info(f"{start_idx + 1}-{end_idx}번째 / 총 {len(filtered_df)}개")
            
        # CSV 다운로드
        try:
            csv = filtered_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="CSV 다운로드",
                data=csv,
                file_name=f"naver_keyword_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"CSV 생성 중 오류: {str(e)}")
    except Exception as e:
        logger.error(f"Table view display error: {str(e)}")
        st.error(f"테이블 뷰를 표시하는 중 오류가 발생했습니다: {str(e)}")


def _display_final_results(df):
    """최종 결과 표시 (개선된 버전)"""
    st.subheader("상세 분석 결과")
    
    display_columns = ['순위', '연관 검색어', '언급 횟수']
    if '총 월간검색수' in df.columns:
        display_columns.extend(['PC 월간검색수', '모바일 월간검색수', '총 월간검색수'])
    if '상품수' in df.columns:
        display_columns.append('상품수')
    if '경쟁강도' in df.columns:
        display_columns.append('경쟁강도')
    
    available_columns = [col for col in display_columns if col in df.columns]
    
    tab1, tab2 = st.tabs(["카드형 보기", "테이블 보기"])
    with tab1:
        _display_card_view(df, available_columns)
    with tab2:
        _display_table_view(df, available_columns)

def main():
    # Streamlit 페이지 설정
    st.set_page_config(
        page_title="CHURIXX Title Maker Analyze",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 커스텀 CSS
    st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 2rem; cursor: pointer; text-decoration: none; }
    .main-header:hover { color: #0d5aa7; text-decoration: none; }
    .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
    </style>
    """, unsafe_allow_html=True)

    # 홈 버튼 (상태 초기화)
    if st.button("", key="home_button_hidden", help="홈으로 돌아가기"):
        for key in list(st.session_state.keys()):
            if key.startswith(('card_', 'table_', 'last_keyword', 'api_')):
                del st.session_state[key]
        st.rerun()

    st.markdown("""
    <div onclick="document.querySelector('[data-testid=\\'home_button_hidden\\']').click()" 
         class="main-header">
        CHURIXX Title Maker Analyze
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 네이버 쇼핑 키워드 연관검색어와 검색량을 분석하는 도구입니다!")

    analyzer = NaverKeywordAnalyzer()

    # 사이드바
    with st.sidebar:
        st.header("분석 설정")

        # 데이터 소스 토글
        source = st.radio(
            "데이터 소스",
            ["실시간 1페이지(권장)", "Open API 근사치"],
            index=0,
            help="실시간: 실제 네이버쇼핑 1페이지 1~40위 / 근사치: Open API 유사도 정렬"
        )
        
        keyword = st.text_input(
            '키워드 입력', 
            value='',
            help="분석하고 싶은 상품 키워드를 입력하세요",
            placeholder="예: 반팔티, 노트북, 운동화"
        )
        
        st.subheader("분석 옵션")
        show_basic_chart = st.checkbox("기본 차트 표시", value=True)
        show_comparison_chart = st.checkbox("비교 차트 표시", value=False)
        show_search_volume = st.checkbox("검색량 데이터 포함", value=True)
        
        max_keywords_for_volume = st.slider(
            "검색량 조회할 키워드 수",
            min_value=5, max_value=20, value=10,
            help="더 많은 키워드를 선택하면 시간이 더 걸릴 수 있습니다."
        )
        
        analyze_button = st.button('분석 시작', type="primary", use_container_width=True)

    if 'last_keyword' not in st.session_state:
        st.session_state.last_keyword = ''
    
    keyword_changed = (keyword != st.session_state.last_keyword and keyword.strip() != '')
    should_analyze = analyze_button or keyword_changed
    
    if keyword:
        st.session_state.last_keyword = keyword

    # 메인 분석 로직
    if should_analyze and keyword:
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        try:
            status_text.text('연관 검색어 분석 중...')
            progress_bar.progress(20)

            if source.startswith("실시간"):
                # This part is missing in the provided appcud.py, assuming it's meant to be here
                # related_df = analyzer.get_related_keywords_realtime(keyword) 
                # For now, using the get_related_keywords from the class
                related_df = analyzer.get_related_keywords(keyword) 
                time.sleep(random.uniform(0.6, 1.2))  # Add random sleep if real-time scraping is involved
            else:
                related_df = analyzer.get_related_keywords(keyword)

            if not related_df.empty:
                related_df = related_df.copy()
                # Ensure numeric conversion for '순위' and '언급 횟수'
                related_df['순위'] = pd.to_numeric(related_df['순위'], errors='coerce').fillna(0).astype(int)
                related_df['언급 횟수'] = pd.to_numeric(related_df['언급 횟수'], errors='coerce').fillna(1).astype(int)
                related_df['연관 검색어'] = related_df['연관 검색어'].astype(str)
                related_df = related_df[related_df['언급 횟수'] > 0].reset_index(drop=True)
                related_df['순위'] = range(1, len(related_df) + 1)
            
            if not related_df.empty:
                progress_bar.progress(40)
                
                results_container = st.container()
                with results_container:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("발견된 연관 검색어", f"{len(related_df)}개")
                    with col2:
                        st.metric("평균 언급 횟수", f"{related_df['언급 횟수'].mean():.1f}")
                    with col3:
                        st.metric("최고 언급 횟수", f"{related_df['언급 횟수'].max()}")
                
                chart_container = st.container()
                with chart_container:
                    if show_basic_chart:
                        basic_chart = create_enhanced_chart(related_df)
                        if basic_chart:
                            st.plotly_chart(basic_chart, use_container_width=True)
                
                if show_search_volume:
                    status_text.text('검색량 및 상품수 데이터 수집 중...')
                    progress_bar.progress(60)
                    
                    related_keywords = related_df['연관 검색어'].head(max_keywords_for_volume).tolist()
                    product_counts = analyzer.get_product_count_for_keywords(related_keywords)
                    progress_bar.progress(70)
                    
                    all_volume_data = []
                    keyword_batches = [related_keywords[i:i + 5] for i in range(0, len(related_keywords), 5)]
                    for i, batch in enumerate(keyword_batches):
                        keywords_str = ','.join(batch)
                        search_volume_data = analyzer.get_search_volume(keywords_str)
                        if search_volume_data and 'keywordList' in search_volume_data:
                            all_volume_data.extend(search_volume_data['keywordList'])
                        progress_bar.progress(70 + (i + 1) * 20 // len(keyword_batches))
                        if i < len(keyword_batches) - 1:
                            time.sleep(1)
                    
                    final_df = _merge_and_clean_data(related_df, all_volume_data, product_counts)

                    if show_comparison_chart:
                        comparison_chart = create_comparison_chart(final_df)
                        if comparison_chart:
                            st.plotly_chart(comparison_chart, use_container_width=True)
                    
                    _display_final_results(final_df)
                
                if not show_search_volume:
                    related_keywords = related_df['연관 검색어'].tolist()
                    product_counts = analyzer.get_product_count_for_keywords(related_keywords)
                    if product_counts:
                        product_count_df = pd.DataFrame(list(product_counts.items()), columns=['연관 검색어', '상품수'])
                        final_df = pd.merge(related_df, product_count_df, on='연관 검색어', how='left')
                        final_df['상품수'] = final_df['상품수'].fillna(0).astype(int)
                        st.subheader("연관 검색어 + 상품수 분석 결과")
                        st.dataframe(final_df, use_container_width=True, hide_index=True)
                    else:
                        st.subheader("연관 검색어 분석 결과")
                        st.dataframe(related_df, use_container_width=True, hide_index=True)
                
                progress_bar.progress(100)
                status_text.text('분석 완료!')
                time.sleep(0.6)
                progress_container.empty()
            else:
                progress_container.empty()
                st.error("연관 검색어를 찾을 수 없습니다. 다른 키워드로 시도해보세요.")
                
        except Exception as e:
            progress_container.empty()
            st.error(f"분석 중 오류가 발생했습니다: {str(e)}")
            logger.error(f"Analysis error: {str(e)}")
            
    elif should_analyze and not keyword:
        st.error("키워드를 입력해주세요!")
    else:
        st.info("사이드바에서 키워드를 입력하고 '분석 시작' 버튼을 클릭하세요!")
        st.subheader("인기 키워드 제안")
        sample_keywords = ['반팔티', '노트북', '운동화', '화장품', '핸드폰케이스', '텀블러', '백팩']
        cols = st.columns(len(sample_keywords))
        for i, sample in enumerate(sample_keywords):
            with cols[i]:
                if st.button(sample, key=f"sample_{i}"):
                    st.session_state.last_keyword = sample
                    st.rerun()

    with st.expander("사용법 및 도움말"):
        st.markdown("""
        ### 사용법
        1. **키워드 입력**: 사이드바에서 분석하고 싶은 상품명이나 카테고리를 입력하세요
        2. **데이터 소스 선택**: 실시간 1페이지(권장) 또는 Open API 근사치
        3. **분석 실행**: '분석 시작' 버튼 클릭
        4. **결과 확인**: 카드형/테이블 탭으로 세부 확인 및 CSV 다운로드
        """)

if __name__ == "__main__":
    main()
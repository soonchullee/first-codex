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
NAVER_AD_ACCESS_LICENSE = os.getenv('NAVER_AD_ACCESS_LICENSE', '0100000000b023976c548606bfd54a31b499d1066cd59544d9ace90d87de8da28d33426930')
NAVER_AD_SECRET_KEY = os.getenv('NAVER_AD_SECRET_KEY', 'AQAAAACwI5dsVIYGv9VKMbSZ0QZs246RMlg7Og3b+eFMDgNt3A==')
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
                df = df.sort_values(by="언급 횟수", ascending=False).head(40)
                
                # 중복 제거 및 순위 추가
                df = df.drop_duplicates(subset=['연관 검색어'], keep='first')
                df.reset_index(drop=True, inplace=True)
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
                
        except requests.exceptions.Timeout:
            st.error("⏰ API 호출 시간 초과 (20초)")
        except Exception as e:
            st.error(f"❌ 예상치 못한 오류: {str(e)}")
            
        return None

    def _handle_api_error(self, response):
        """API 에러 처리"""
        error_messages = {
            401: "🔐 인증 실패: API 키, 시크릿 키, 고객 ID를 확인해주세요.",
            403: "🚫 권한 없음: 해당 고객 ID에 대한 접근 권한이 없습니다.",
            429: "⏰ 호출 한도 초과: 잠시 후 다시 시도해주세요.",
            400: "📝 잘못된 요청: 파라미터를 확인해주세요."
        }
        
        st.error(f"❌ 광고 API 호출 실패 (상태 코드: {response.status_code})")
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
            'text': '📊 연관 검색어 언급 빈도 (Top 15)',
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
    
    # 데이터가 있는 항목만 필터링
    valid_data = df[(df['총 월간검색수'] > 0) & (df['언급 횟수'] > 0)].head(10)
    
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
        title='📈 언급 횟수 vs 월간 검색수 비교',
        xaxis_title='연관 검색어',
        yaxis=dict(title='언급 횟수', side='left', color='blue'),
        yaxis2=dict(title='월간 검색수', side='right', overlaying='y', color='red'),
        height=400,
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig

def main():
    # Streamlit 페이지 설정
    st.set_page_config(
        page_title="네이버 쇼핑 키워드 분석",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 커스텀 CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🛒 네이버 쇼핑 키워드 분석</h1>', unsafe_allow_html=True)
    st.markdown("### 관심 있는 쇼핑 키워드의 연관 검색어와 검색량을 분석해보세요!")

    # 분석기 인스턴스 생성
    analyzer = NaverKeywordAnalyzer()

    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 분석 설정")
        
        keyword = st.text_input(
            '🔍 키워드 입력', 
            value='',
            help="분석하고 싶은 상품 키워드를 입력하세요",
            placeholder="예: 반팔티, 노트북, 운동화"
        )
        
        st.subheader("📊 분석 옵션")
        show_basic_chart = st.checkbox("기본 차트 표시", value=True)
        show_comparison_chart = st.checkbox("비교 차트 표시", value=False)
        show_search_volume = st.checkbox("검색량 데이터 포함", value=True)
        
        max_keywords_for_volume = st.slider(
            "검색량 조회할 키워드 수",
            min_value=5,
            max_value=20,
            value=10,
            help="더 많은 키워드를 선택하면 시간이 더 걸릴 수 있습니다."
        )
        
        analyze_button = st.button('🚀 분석 시작', type="primary", use_container_width=True)

    # 세션 상태 관리
    if 'last_keyword' not in st.session_state:
        st.session_state.last_keyword = ''
    
    keyword_changed = (keyword != st.session_state.last_keyword and keyword.strip() != '')
    should_analyze = analyze_button or keyword_changed
    
    if keyword:
        st.session_state.last_keyword = keyword

    # 메인 분석 로직
    if should_analyze and keyword:
        # 진행상황 표시
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        try:
            # 1단계: 연관 검색어 분석
            status_text.text('🔍 연관 검색어 분석 중...')
            progress_bar.progress(20)
            
            related_df = analyzer.get_related_keywords(keyword)
            
            if not related_df.empty:
                progress_bar.progress(40)
                
                # 결과 표시용 컨테이너
                results_container = st.container()
                
                with results_container:
                    # 성공 메시지와 기본 통계
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📝 발견된 연관 검색어", f"{len(related_df)}개")
                    with col2:
                        st.metric("🔢 평균 언급 횟수", f"{related_df['언급 횟수'].mean():.1f}")
                    with col3:
                        st.metric("🏆 최고 언급 횟수", f"{related_df['언급 횟수'].max()}")
                
                # 차트 표시
                chart_container = st.container()
                with chart_container:
                    if show_basic_chart:
                        basic_chart = create_enhanced_chart(related_df)
                        if basic_chart:
                            st.plotly_chart(basic_chart, use_container_width=True)
                
                # 검색량 데이터 수집
                if show_search_volume:
                    status_text.text('📊 검색량 및 상품수 데이터 수집 중...')
                    progress_bar.progress(60)
                    
                    # 상품수 조회
                    related_keywords = related_df['연관 검색어'].head(max_keywords_for_volume).tolist()
                    product_counts = analyzer.get_product_count_for_keywords(related_keywords)
                    progress_bar.progress(70)
                    
                    # 검색량 조회 (배치 처리)
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
                    
                    # 데이터 병합 및 정리
                    final_df = self._merge_and_clean_data(analyzer, related_df, all_volume_data, product_counts)
                    
                    # 비교 차트 표시
                    if show_comparison_chart:
                        comparison_chart = create_comparison_chart(final_df)
                        if comparison_chart:
                            st.plotly_chart(comparison_chart, use_container_width=True)
                    
                    # 최종 결과 표시
                    self._display_final_results(analyzer, final_df)
                else:
                    # 검색량 없이 상품수만
                    related_keywords = related_df['연관 검색어'].tolist()
                    product_counts = analyzer.get_product_count_for_keywords(related_keywords)
                    
                    if product_counts:
                        product_count_df = pd.DataFrame(list(product_counts.items()), 
                                                      columns=['연관 검색어', '상품수'])
                        final_df = pd.merge(related_df, product_count_df, on='연관 검색어', how='left')
                        final_df['상품수'] = final_df['상품수'].fillna(0).astype(int)
                        
                        st.subheader("📋 연관 검색어 + 상품수 분석 결과")
                        st.dataframe(final_df, use_container_width=True, hide_index=True)
                    else:
                        st.subheader("📋 연관 검색어 분석 결과")
                        st.dataframe(related_df, use_container_width=True, hide_index=True)
                
                progress_bar.progress(100)
                status_text.text('✅ 분석 완료!')
                time.sleep(1)
                progress_container.empty()
                
            else:
                progress_container.empty()
                st.error("❌ 연관 검색어를 찾을 수 없습니다. 다른 키워드로 시도해보세요.")
                
        except Exception as e:
            progress_container.empty()
            st.error(f"❌ 분석 중 오류가 발생했습니다: {str(e)}")
            logger.error(f"Analysis error: {str(e)}")
            
    elif should_analyze and not keyword:
        st.error("⚠️ 키워드를 입력해주세요!")
    else:
        # 시작 화면
        st.info("💡 사이드바에서 키워드를 입력하고 '분석 시작' 버튼을 클릭하세요!")
        
        # 샘플 키워드 제안
        st.subheader("💭 인기 키워드 제안")
        sample_keywords = ['반팔티', '노트북', '운동화', '화장품', '핸드폰케이스', '텀블러', '백팩']
        
        cols = st.columns(len(sample_keywords))
        for i, sample in enumerate(sample_keywords):
            with cols[i]:
                if st.button(sample, key=f"sample_{i}"):
                    st.session_state['keyword_input'] = sample
                    st.rerun()

    def _merge_and_clean_data(analyzer, related_df, volume_data, product_counts):
        """데이터 병합 및 정리"""
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
        
        # 숫자 컬럼 변환
        numeric_columns = ['PC 월간검색수', '모바일 월간검색수', '상품수']
        for col in numeric_columns:
            if col in merged_df.columns:
                merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(0).astype(int)
        
        # 총 검색수 계산
        if 'PC 월간검색수' in merged_df.columns and '모바일 월간검색수' in merged_df.columns:
            merged_df['총 월간검색수'] = merged_df['PC 월간검색수'] + merged_df['모바일 월간검색수']
        
        return merged_df

    def _display_final_results(analyzer, df):
        """최종 결과 표시"""
        st.subheader("📋 상세 분석 결과")
        
        # 탭으로 결과 분리
        tab1, tab2, tab3 = st.tabs(["📊 전체 결과", "🔍 검색량 있는 키워드", "📈 상위 키워드 요약"])
        
        with tab1:
            display_columns = ['순위', '연관 검색어', '언급 횟수']
            if '총 월간검색수' in df.columns:
                display_columns.extend(['PC 월간검색수', '모바일 월간검색수', '총 월간검색수'])
            if '상품수' in df.columns:
                display_columns.append('상품수')
            if '경쟁강도' in df.columns:
                display_columns.append('경쟁강도')
            
            available_columns = [col for col in display_columns if col in df.columns]
            st.dataframe(df[available_columns], use_container_width=True, hide_index=True)
        
        with tab2:
            if '총 월간검색수' in df.columns:
                with_volume = df[df['총 월간검색수'] > 0]
                if not with_volume.empty:
                    st.dataframe(with_volume[available_columns], use_container_width=True, hide_index=True)
                else:
                    st.info("검색량 데이터가 있는 키워드가 없습니다.")
            else:
                st.info("검색량 데이터가 포함되지 않았습니다.")
        
        with tab3:
            # 상위 5개 키워드 요약
            top_5 = df.head(5)
            for idx, row in top_5.iterrows():
                with st.expander(f"#{row['순위']} {row['연관 검색어']} (언급 {row['언급 횟수']}회)"):
                    col1, col2 = st.columns(2)
                    with col1:
                        if '총 월간검색수' in row:
                            st.write(f"**월간 검색수:** {row['총 월간검색수']:,}")
                        if '상품수' in row:
                            st.write(f"**상품수:** {row['상품수']:,}")
                    with col2:
                        if '경쟁강도' in row:
                            st.write(f"**경쟁강도:** {row['경쟁강도']}")

    # 사용법 안내
    with st.expander("ℹ️ 사용법 및 도움말"):
        st.markdown("""
    # 사용법 안내
    with st.expander("ℹ️ 사용법 및 도움말"):
        st.markdown("""
        ### 📝 사용법
        1. **키워드 입력**: 사이드바에서 분석하고 싶은 상품명이나 카테고리를 입력하세요
           - 예시: "반팔티", "노트북", "운동화", "화장품" 등
           - 너무 구체적이거나 브랜드명보다는 일반적인 상품 카테고리가 좋습니다
        
        2. **분석 옵션 선택**: 
           - **기본 차트**: 연관 검색어의 언급 빈도를 막대 차트로 표시
           - **비교 차트**: 언급 횟수와 월간 검색수를 함께 비교
           - **검색량 데이터**: 네이버 광고 API를 통한 실제 검색량 데이터 포함
           - **검색량 조회 키워드 수**: API 한도를 고려하여 조절 가능 (5-20개)
        
        3. **분석 실행**: '🚀 분석 시작' 버튼을 클릭하여 분석을 시작하세요
        
        4. **결과 확인**: 4가지 탭으로 구분된 결과를 확인하세요
           - **카드형 보기**: 스크롤 없이 페이지별로 보기 편한 형태
           - **테이블 보기**: 필터링과 정렬이 가능한 표 형태
           - **검색량 분석**: 검색량 구간별 상세 분석
           - **종합 리포트**: 마케팅 인사이트와 추천사항
        
        ### 📊 결과 해석 가이드
        
        #### 주요 지표 설명:
        - **언급 횟수**: 네이버 쇼핑 검색결과 상품명에서 해당 키워드가 언급된 횟수
        - **월간검색수**: 네이버에서 실제로 검색되는 월평균 횟수 (PC + 모바일)
        - **상품수**: 해당 키워드로 검색했을 때 나오는 네이버 쇼핑 상품의 총 개수
        - **경쟁강도**: 광고 경쟁 정도 (0-100, 높을수록 경쟁 치열)
        
        #### 마케팅 활용법:
        - **언급 횟수가 높은 키워드**: 실제 판매자들이 많이 사용하는 핫한 키워드
        - **검색량이 높은 키워드**: 고객들이 실제로 많이 검색하는 키워드 - 광고 키워드로 활용
        - **경쟁강도가 낮은 키워드**: 진입하기 쉬운 틈새 키워드 - SEO 전략에 활용
        - **상품수가 적은 키워드**: 경쟁이 적어 상위 노출 기회가 높은 키워드
        
        ### ⚠️ 주의사항 및 제한사항
        
        #### 데이터 특성:
        - 연관 검색어는 최대 40개까지 표시됩니다
        - 검색량 데이터는 선택한 개수만큼만 조회됩니다 (API 한도 고려)
        - 일부 키워드는 네이버 광고에서 지원하지 않아 검색량 데이터가 없을 수 있습니다
        - 모든 데이터는 월간 평균 기준이며 실시간 데이터가 아닙니다
        
        #### 최적 사용 팁:
        - 처음에는 검색량 조회 키워드 수를 적게(5-10개) 설정하여 빠르게 테스트해보세요
        - 여러 비슷한 키워드로 분석하여 트렌드를 파악해보세요
        - 정기적으로 분석하여 키워드 트렌드 변화를 모니터링하세요
        
        #### API 관련:
        - 네이버 검색 API와 광고 API를 사용하므로 호출 한도가 있습니다
        - 연속적인 분석 시 잠시 기다린 후 다시 시도해주세요
        - API 오류 발생 시 키워드를 바꾸어 다시 시도해보세요
        
        ### 💡 활용 시나리오
        
        #### 쇼핑몰 운영자:
        - 신상품 출시 전 관련 키워드 트렌드 파악
        - 상품명 최적화를 위한 인기 키워드 발굴
        - 광고 키워드 선정 및 예산 배분 참고자료
        
        #### 마케터:
        - 콘텐츠 마케팅 키워드 발굴
        - SEO 전략 수립을 위한 키워드 리서치
        - 경쟁사 분석 및 시장 트렌드 파악
        
        #### 개인 판매자:
        - 판매할 상품의 시장성 검증
        - 상품 설명 작성 시 포함할 키워드 선정
        - 네이버 쇼핑 최적화 전략 수립
        """)


if __name__ == "__main__":
    main()
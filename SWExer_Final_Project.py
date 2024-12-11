import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import heapq
import re

# 데이터 로드 함수 (캐싱 적용)
@st.cache_data
def load_data():
    # GitHub 파일 URL
    subway_url = "https://github.com/zzunni/SWExer_Final_Project/blob/main/subway.csv"
    subway_location_url = "https://github.com/zzunni/SWExer_Final_Project/blob/main/subwayLocation.csv"
    real_estate_url = "https://github.com/zzunni/SWExer_Final_Project/blob/main/%EC%84%9C%EC%9A%B8%EC%8B%9C%20%EB%B6%80%EB%8F%99%EC%82%B0%20%EC%A0%84%EC%9B%94%EC%84%B8%EA%B0%80%20%EC%A0%95%EB%B3%B4.csv"

    # CSV 파일 읽기
    subway_data = pd.read_csv(subway_url)
    subway_location_data = pd.read_csv(subway_location_url)
    real_estate_data = pd.read_csv(real_estate_url)
    
    # 원하는 열만 선택
    selected_columns = [
        "법정동명", "전월세 구분", 
        "보증금(만원)", "임대료(만원)", "임대면적(㎡)", 
        "건물명", "건축년도", "위도", "경도"
    ]
    real_estate_data = real_estate_data[selected_columns]
    return subway_data, subway_location_data, real_estate_data
  
# 다익스트라 알고리즘 함수
def dijkstra(subway_data, start_station, end_station):
    # 지하철역 데이터를 이용해 그래프를 생성
    graph = {}
    for _, row in subway_data.iterrows():
        if row['출발역'] not in graph:
            graph[row['출발역']] = []
        graph[row['출발역']].append((row['도착역'], row['소요시간']))
        
        if row['도착역'] not in graph:
            graph[row['도착역']] = []
        graph[row['도착역']].append((row['출발역'], row['소요시간']))

    # 다익스트라 알고리즘
    def shortest_path(graph, start, end):
        pq = [(0, start)]  # 우선순위 큐, (누적시간, 역)
        dist = {station: float('inf') for station in graph}
        dist[start] = 0
        prev = {station: None for station in graph}
        
        while pq:
            current_dist, current_station = heapq.heappop(pq)
            
            if current_station == end:
                break

            for neighbor, time in graph[current_station]:
                new_dist = current_dist + time
                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    prev[neighbor] = current_station
                    heapq.heappush(pq, (new_dist, neighbor))

        # 최단 경로 반환
        return dist[end]

    # 최단 시간 계산
    shortest_time = shortest_path(graph, start_station, end_station)
    return shortest_time

# 출발역과 도착역 이름에서 괄호와 숫자 부분만 제거
def remove_parentheses(station_name):
    return re.sub(r'\(.*\)', '', station_name).strip()
  
# 지도 생성 함수
def create_map(filtered_properties, subway_location_data, start_station, end_station):
    # NaN이 포함된 위치 제거
    filtered_properties = filtered_properties.dropna(subset=["위도", "경도"])

    # 기본 지도 생성
    m = folium.Map(location=[37.5665, 126.9780], zoom_start=12)

    # 지하철 위치 마커 추가
    for _, row in subway_location_data.iterrows():
        folium.Marker(
            location=[row['위도'], row['경도']],
            popup=f"{row['출발역']} (지하철)",
            icon=folium.Icon(color="blue", icon="subway", prefix="fa")
        ).add_to(m)

    # 마커 클러스터링 추가
    marker_cluster = MarkerCluster().add_to(m)

    # 출발역과 도착역 데이터를 가져오는 부분 수정
    start_station_cleaned = remove_parentheses(start_station)
    end_station_cleaned = remove_parentheses(end_station)
    
      # 출발역과 도착역을 구별할 수 있도록 마커 추가
    start_station_data = subway_location_data[subway_location_data['출발역'] == start_station_cleaned].iloc[0]
    end_station_data = subway_location_data[subway_location_data['출발역'] == end_station_cleaned].iloc[0]
    
    # 출발역 마커 (빨간색, 크고 눈에 띄게)
    folium.Marker(
        location=[start_station_data['위도'], start_station_data['경도']],
        popup=f"출발역: {start_station}",
        icon=folium.Icon(color="red", icon="circle", prefix="fa", icon_size=(30, 30))
    ).add_to(m)

    # 도착역 마커 (빨간색, 크고 눈에 띄게)
    folium.Marker(
        location=[end_station_data['위도'], end_station_data['경도']],
        popup=f"도착역: {end_station}",
        icon=folium.Icon(color="darkred", icon="circle", prefix="fa", icon_size=(30, 30))
    ).add_to(m)
    
    # 부동산 매물 마커 추가
    for _, row in filtered_properties.iterrows():
        # 팝업 내용을 HTML로 구성
        popup_content = f"""
        <div style="font-family: Arial, sans-serif; font-size: 14px; line-height: 1.6;">
            <b style="color: #2A81CB;">건물명:</b> {row['건물명'] if pd.notna(row['건물명']) else "정보 없음"}<br>
            <b style="color: #2A81CB;">법정동:</b> {row['법정동명']}<br>
            <b style="color: #2A81CB;">전월세:</b> {row['전월세 구분']}<br>
            <b style="color: #2A81CB;">보증금:</b> {row['보증금(만원)']}만원<br>
            <b style="color: #2A81CB;">임대료:</b> {row['임대료(만원)']}만원<br>
            <b style="color: #2A81CB;">면적:</b> {row['임대면적(㎡)']}㎡<br>
        </div>
        """
        popup = folium.Popup(popup_content, max_width=300)  # 팝업 생성

        # 클러스터에 마커 추가
        folium.Marker(
            location=[row["위도"], row["경도"]],
            popup=popup,
            icon=folium.Icon(color="green", icon="home", prefix="fa"),
        ).add_to(marker_cluster)

    return m

# 부동산 매물 필터링 함수
def filter_properties(real_estate_data, deal_type, max_budget, max_age, min_area, max_area, deposit=None, rent=None, district=None):
    filtered_data = real_estate_data.copy()

    # 법정동 필터링
    if district:
        filtered_data = filtered_data[filtered_data['법정동명'] == district]

    # 전월세 구분 필터링
    filtered_data = filtered_data[filtered_data['전월세 구분'] == deal_type]

    # 예산 필터링
    if deal_type == '전세':
        filtered_data = filtered_data[filtered_data['보증금(만원)'] <= max_budget]
    elif deal_type == '월세':
        if deposit is not None and rent is not None:
            filtered_data = filtered_data[
                (filtered_data['보증금(만원)'] <= deposit) & (filtered_data['임대료(만원)'] <= rent)
            ]

    # 건물 연식 필터
    current_year = pd.to_datetime('today').year
    filtered_data = filtered_data[filtered_data['건축년도'] >= current_year - max_age]

    # 면적 필터
    filtered_data = filtered_data[(filtered_data['임대면적(㎡)'] >= min_area) & (filtered_data['임대면적(㎡)'] <= max_area)]
    return filtered_data

# Streamlit 앱
def main():
    st.title("부동산 검색 프로그램")
    st.sidebar.header("검색 조건")
    
    # 데이터 로드
    subway_data, subway_location_data, real_estate_data = load_data()
    
    # 사용자 입력
    district = st.sidebar.text_input("법정동명을 입력하세요:", "강남구")
    deal_type = st.sidebar.selectbox("부동산 유형", ["전세", "월세"])
    
    
    # 예산 입력
    max_budget = None
    deposit = None
    rent = None

    if deal_type == "전세":
        max_budget = st.sidebar.number_input("최대 보증금 (만원)", min_value=0, value=10000)
    elif deal_type == "월세":
        deposit = st.sidebar.number_input("최대 보증금 (만원)", min_value=0, value=10000)
        rent = st.sidebar.number_input("최대 월세 (만원)", min_value=0, value=1000)
    
    # 건물 연식
    max_age = st.sidebar.slider("건물 연식 (최대 연도)", min_value=0, max_value=100, value=30)
    
    # 면적 필터
    min_area = st.sidebar.number_input("최소 면적 (㎡)", min_value=0, value=20)
    max_area = st.sidebar.number_input("최대 면적 (㎡)", min_value=0, value=100)
    
    # 출발역과 도착역 입력
    start_station = st.text_input("출발역을 입력하세요:")
    end_station = st.text_input("도착역을 입력하세요:")

    # 최단 경로 예상 소요시간 출력
    if start_station and end_station:
        shortest_time = dijkstra(subway_data, start_station, end_station)
        st.write(f"최단 시간: {shortest_time} 분")

    # 검색 버튼 추가
    if st.sidebar.button("검색"):
        # 부동산 매물 필터링
        filtered_properties = filter_properties(
            real_estate_data=real_estate_data,
            deal_type=deal_type,
            max_budget=max_budget,
            max_age=max_age,
            min_area=min_area,
            max_area=max_area,
            deposit=deposit,
            rent=rent,
            district=district
        )
        # 검색 결과 저장
        st.session_state['filtered_properties'] = filtered_properties

    # 검색 결과 표시
    if 'filtered_properties' in st.session_state:
        filtered_properties = st.session_state['filtered_properties']

        # 지도 생성
        st.subheader("지하철 및 부동산 위치 지도")
        st.write(f"현재 법정동: {district} / 부동산 유형: {deal_type}")
        property_map = create_map(filtered_properties, subway_location_data, start_station, end_station)
        st_folium(property_map, width=700, height=500)

        # 필터링된 부동산 매물 출력
        st.subheader("검색된 부동산 매물")
        st.write(f"검색된 매물 수: {len(filtered_properties)}개")
        st.write(filtered_properties)

# 실행
if __name__ == "__main__":
    main()

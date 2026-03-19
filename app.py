import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np

st.set_page_config(page_title="Airline L/F Predictor", layout="wide")
st.title("✈️ 노선별 L/F 미래 예측 시뮬레이터")

uploaded_file = st.file_uploader("1년치 실적 엑셀 파일을 업로드하세요 (Route, Date, LF 컬럼 포함)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # [추가된 부분] LF 데이터를 숫자로 변환하고, 숫자가 아닌 것은 비워둠(NaN)
    df['LF'] = pd.to_numeric(df['LF'], errors='coerce')
    
    # 노선 선택 메뉴
    routes = df['Route'].unique()
    selected_route = st.sidebar.selectbox("분석할 노선을 선택하세요", routes)
    
    # 선택한 노선 데이터 추출 및 정렬
    route_df = df[df['Route'] == selected_route].sort_values('Date').copy()
    
    # [추가된 부분] 빈 데이터가 있으면 앞의 값으로 채워줌 (계산 오류 방지)
    route_df['LF'] = route_df['LF'].ffill().bfill()
    
    if len(route_df) > 14:
        st.subheader(f"📊 {selected_route} 노선 분석 결과")
        
        try:
            # 예측 모델 실행 (데이터를 부동소수점으로 변환하여 계산 안정성 확보)
            lf_values = route_df['LF'].astype(float).values
            model = ExponentialSmoothing(lf_values, seasonal='add', seasonal_periods=7).fit()
            forecast = model.forecast(60)
            
            # 미래 날짜 생성
            last_date = route_df['Date'].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=60)
            forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_LF': forecast})

            # 그래프 시각화
            fig = px.line(route_df, x='Date', y='LF', title=f"{selected_route} L/F 추이 및 향후 60일 예측")
            fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Predicted_LF'], name="미래 예측치", mode='lines+markers')
            st.plotly_chart(fig, use_container_width=True)
            
            # 예측 데이터 표 노출
            st.write("📅 향후 60일 예측 데이터 (상위 5일)")
            st.dataframe(forecast_df.head())
            
        except Exception as e:
            st.error(f"예측 도중 오류가 발생했습니다: {e}")
    else:
        st.info("데이터가 부족합니다. 최소 14일 이상의 실적이 필요합니다.")

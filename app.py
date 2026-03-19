import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import holidays # [추가] 공휴일 라이브러리

st.set_page_config(page_title="Airline L/F Predictor", layout="wide")
st.title("✈️ 노선별 L/F 예측 (공휴일 반영 버전)")

# 한국 공휴일 정보 가져오기
kr_holidays = holidays.KR()

uploaded_file = st.file_uploader("엑셀 파일을 업로드하세요", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df['LF'] = pd.to_numeric(df['LF'], errors='coerce').ffill().bfill()
    
    selected_route = st.sidebar.selectbox("노선 선택", df['Route'].unique())
    route_df = df[df['Route'] == selected_route].sort_values('Date').copy()
    
    if len(route_df) > 20:
        forecast_days = 90 
        model = ExponentialSmoothing(route_df['LF'], seasonal='add', seasonal_periods=7).fit()
        forecast = model.forecast(forecast_days)
        
        future_dates = pd.date_range(start=route_df['Date'].max() + pd.Timedelta(days=1), periods=forecast_days)
        
        # [핵심 로직] 공휴일이면 예측치에 보정값(예: +15%) 추가
        adjusted_forecast = []
        for date, val in zip(future_dates, forecast):
            if date in kr_holidays:
                adjusted_forecast.append(min(100, val + 15)) # 공휴일은 15% 가산 (최대 100%)
            else:
                adjusted_forecast.append(val)
        
        forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_LF': adjusted_forecast})

        fig = px.line(route_df, x='Date', y='LF', title=f"{selected_route} 공휴일 보정 예측 (90일)")
        fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Predicted_LF'], name="미래 예측(공휴일반영)", mode='lines+markers')
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 Tip: 한국 공휴일(설, 추석 등)에는 예측치에 약 15%의 가산점이 자동으로 반영되었습니다.")

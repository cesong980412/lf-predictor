import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import holidays

st.set_page_config(page_title="Airline L/F Predictor", layout="wide")
st.title("✈️ 노선별 L/F 미래 예측 시뮬레이터")

kr_holidays = holidays.KR()

uploaded_file = st.file_uploader("엑셀 파일을 업로드하세요", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df['LF'] = pd.to_numeric(df['LF'], errors='coerce').ffill().bfill()
    
    selected_route = st.sidebar.selectbox("노선 선택", df['Route'].unique())
    route_df = df[df['Route'] == selected_route].sort_values('Date').copy()
    
    if len(route_df) > 14:
        forecast_days = 90 
        model = ExponentialSmoothing(route_df['LF'], seasonal='add', seasonal_periods=7).fit()
        forecast = model.forecast(forecast_days)
        
        future_dates = pd.date_range(start=route_df['Date'].max() + pd.Timedelta(days=1), periods=forecast_days)
        
        # [수정된 로직] 단순히 15를 더하는 게 아니라, 원래 값의 15%를 가산합니다.
        adjusted_forecast = []
        for date, val in zip(future_dates, forecast):
            if date in kr_holidays:
                # 공휴일이면 기존 예측값에 1.15를 곱함 (15% 할증)
                # 만약 데이터가 0~1 사이라면 최대 1.0으로 제한
                new_val = val * 1.15 
                if val <= 1.0: # 소수점 데이터인 경우
                    new_val = min(1.0, new_val)
                else: # 80 같은 정수 데이터인 경우
                    new_val = min(100, new_val)
                adjusted_forecast.append(new_val)
            else:
                adjusted_forecast.append(val)
        
        forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_LF': adjusted_forecast})

        fig = px.line(route_df, x='Date', y='LF', title=f"{selected_route} 공휴일 반영 예측 (90일)")
        fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Predicted_LF'], name="공휴일 보정치", mode='lines+markers')
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("✅ 공휴일 당일 데이터에 15% 가산율을 적용하여 보정했습니다.")

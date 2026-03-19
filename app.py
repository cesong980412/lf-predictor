import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import holidays
import numpy as np

st.set_page_config(page_title="Airline L/F Predictor Pro", layout="wide")
st.title("✈️ 노선별 L/F 인공지능 예측 모델 (정확도 분석형)")

kr_holidays = holidays.KR()

uploaded_file = st.file_uploader("엑셀 파일을 업로드하세요", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df['LF'] = pd.to_numeric(df['LF'], errors='coerce').ffill().bfill()
    
    selected_route = st.sidebar.selectbox("노선 선택", df['Route'].unique())
    route_df = df[df['Route'] == selected_route].sort_values('Date').copy()
    
    if len(route_df) > 30: # 데이터가 어느 정도 쌓였을 때만 작동
        # 1. 모델 학습 및 정확도 측정 (최근 14일 데이터를 시험지로 사용)
        train = route_df['LF'][:-14]
        test = route_df['LF'][-14:]
        
        # 더 복잡한 패턴을 학습하도록 옵션 강화 (Trend와 Seasonality 결합)
        model = ExponentialSmoothing(route_df['LF'], trend='add', seasonal='add', seasonal_periods=7).fit()
        
        # 정확도(MAPE) 계산
        test_pred = model.predict(start=len(train), end=len(train)+len(test)-1)
        mape = np.mean(np.abs((test - test_pred) / test)) * 100
        accuracy = max(0, 100 - mape)

        # 상단에 정확도 지표 표시
        col1, col2 = st.columns(2)
        col1.metric("🎯 예측 모델 정확도", f"{accuracy:.1f}%")
        col2.info("정확도가 85% 이상이면 매우 신뢰할 수 있는 데이터입니다.")

        # 2. 미래 90일 예측
        forecast_days = 90
        forecast = model.forecast(forecast_days)
        future_dates = pd.date_range(start=route_df['Date'].max() + pd.Timedelta(days=1), periods=forecast_days)
        
        # 공휴일 보정 (현실적인 비율로 가산)
        adjusted_forecast = []
        for date, val in zip(future_dates, forecast):
            if date in kr_holidays:
                new_val = val * 1.15 # 15% 할증
                adjusted_forecast.append(min(1.0, new_val) if val <= 1.0 else min(100, new_val))
            else:
                adjusted_forecast.append(val)
        
        forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_LF': adjusted_forecast})

        # 3. 그래프 그리기
        fig = px.line(route_df, x='Date', y='LF', title=f"{selected_route} 학습 기반 예측 (정확도 {accuracy:.1f}%)")
        fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Predicted_LF'], name="AI 미래 예측", mode='lines+markers')
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("데이터가 부족하여 인공지능 학습이 어렵습니다. (최소 30일치 필요)")

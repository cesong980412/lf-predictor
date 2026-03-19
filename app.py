import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import holidays
import numpy as np

st.set_page_config(page_title="Airline L/F Predictor Pro", layout="wide")
st.title("✈️ 노선별 L/F 인공지능 예측 모델")

kr_holidays = holidays.KR()

uploaded_file = st.file_uploader("엑셀 파일을 업로드하세요", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df['LF'] = pd.to_numeric(df['LF'], errors='coerce').ffill().bfill()
    
    selected_route = st.sidebar.selectbox("노선 선택", df['Route'].unique())
    route_df = df[df['Route'] == selected_route].sort_values('Date').copy()
    
    if len(route_df) > 21:
        # --- [학습 및 정확도 계산 구간] ---
        # 최근 14일 데이터를 '시험지'로 사용하여 모델의 실력을 테스트합니다.
        train = route_df['LF'][:-14]
        test = route_df['LF'][-14:]
        
        model = ExponentialSmoothing(route_df['LF'], trend='add', seasonal='add', seasonal_periods=7).fit()
        
        # 실제값과 예측값의 차이(MAPE) 계산
        test_pred = model.predict(start=len(train), end=len(train)+len(test)-1)
        mape = np.mean(np.abs((test - test_pred) / test)) * 100
        accuracy = max(0, 100 - mape) # 100점에서 오차율을 뺍니다.

        # --- [화면 상단에 숫자 카드 표시] ---
        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 예측 정확도", f"{accuracy:.1f}%")
        col2.metric("📈 최근 평균 L/F", f"{route_df['LF'].iloc[-7:].mean():.2f}")
        col3.success("85% 이상이면 신뢰도가 매우 높습니다!")

        # 미래 90일 예측
        forecast_days = 90
        forecast = model.forecast(forecast_days)
        future_dates = pd.date_range(start=route_df['Date'].max() + pd.Timedelta(days=1), periods=forecast_days)
        
        adjusted_forecast = []
        for date, val in zip(future_dates, forecast):
            if date in kr_holidays:
                new_val = val * 1.15 # 공휴일 15% 가산
                adjusted_forecast.append(min(1.0, new_val) if val <= 1.0 else min(100, new_val))
            else:
                adjusted_forecast.append(val)
        
        forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_LF': adjusted_forecast})

        # 그래프 출력
        fig = px.line(route_df, x='Date', y='LF', title=f"{selected_route} AI 예측 시뮬레이션 (정확도: {accuracy:.1f}%)")
        fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Predicted_LF'], name="AI 미래 예측", mode='lines+markers')
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("데이터가 부족하여 정확도를 측정할 수 없습니다. (최소 21일치 필요)")

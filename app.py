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
    
    # [수정] 데이터가 0~1 사이면 100을 곱해 퍼센트로 변환, 이미 100 단위면 그대로 유지
    if df['LF'].max() <= 1.0:
        df['LF'] = df['LF'] * 100
    
    df['LF'] = pd.to_numeric(df['LF'], errors='coerce').ffill().bfill()
    
    selected_route = st.sidebar.selectbox("노선 선택", df['Route'].unique())
    route_df = df[df['Route'] == selected_route].sort_values('Date').copy()
    
    if len(route_df) > 21:
        # 1. 모델 학습 및 정확도 계산
        train = route_df['LF'][:-14]
        test = route_df['LF'][-14:]
        
        model = ExponentialSmoothing(route_df['LF'], trend='add', seasonal='add', seasonal_periods=7).fit()
        
        # 실제값과 예측값 비교하여 정확도 측정
        test_pred = model.predict(start=len(train), end=len(train)+len(test)-1)
        mape = np.mean(np.abs((test - test_pred) / test)) * 100
        accuracy = max(0, 100 - mape)

        # 2. 상단에 정확도 지표(Metric) 표시
        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric(label="🎯 모델 예측 정확도", value=f"{accuracy:.1f}%")
        col2.metric(label="📊 최근 7일 평균 L/F", value=f"{route_df['LF'].iloc[-7:].mean():.1f}%")
        col3.success("85% 이상이면 보고서 신뢰도가 매우 높습니다!")
        st.divider()

        # 3. 미래 90일 예측
        forecast_days = 90
        forecast = model.forecast(forecast_days)
        future_dates = pd.date_range(start=route_df['Date'].max() + pd.Timedelta(days=1), periods=forecast_days)
        
        adjusted_forecast = []
        for date, val in zip(future_dates, forecast):
            if date in kr_holidays:
                new_val = val * 1.15 # 공휴일 15% 가산
                adjusted_forecast.append(min(100, new_val))
            else:
                adjusted_forecast.append(val)
        
        forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_LF': adjusted_forecast})

        # 4. 그래프 출력 (Y축을 % 단위로 설정)
        fig = px.line(route_df, x='Date', y='LF', title=f"{selected_route} AI 예측 시뮬레이션 (정확도: {accuracy:.1f}%)")
        fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Predicted_LF'], name="AI 미래 예측", mode='lines+markers')
        
        # Y축 범위를 0~100%로 고정하고 뒤에 % 기호 붙이기
        fig.update_layout(yaxis_tickformat='.0f', yaxis_title="Load Factor (%)", yaxis_range=[0, 105])
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("데이터가 부족합니다. (최소 21일치 실적 필요)")

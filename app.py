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
    
    if df['LF'].max() <= 1.0:
        df['LF'] = df['LF'] * 100
    
    selected_route = st.sidebar.selectbox("노선 선택", df['Route'].unique())
    route_df = df[df['Route'] == selected_route].sort_values('Date').reset_index(drop=True)
    
    if len(route_df) > 10:
        model = ExponentialSmoothing(route_df['LF'], trend='add', seasonal='add', seasonal_periods=7).fit()
        
        train = route_df['LF'][:-7]
        test = route_df['LF'][-7:]
        test_pred = model.predict(start=len(train), end=len(train)+len(test)-1)
        mape = np.mean(np.abs((test - test_pred) / np.where(test == 0, 1, test))) * 100
        accuracy = max(0, 100 - mape)

        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 예측 정확도", f"{accuracy:.1f}%")
        col2.metric("📊 데이터 최신 7일 평균", f"{route_df['LF'].iloc[-7:].mean():.1f}%")
        col3.success("바닥 탈출 성공! 실적+예측 구간 줌인 완료")
        st.divider()

        forecast_days = 60
        forecast = model.forecast(forecast_days)
        future_dates = pd.date_range(start=route_df['Date'].max() + pd.Timedelta(days=1), periods=forecast_days)
        
        adjusted_forecast = []
        for date, val in zip(future_dates, forecast):
            if date in kr_holidays:
                adjusted_forecast.append(min(100, val * 1.15))
            else:
                adjusted_forecast.append(val)
        
        forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_LF': adjusted_forecast})

        # 그래프 그리기
        fig = px.line(route_df, x='Date', y='LF', title=f"{selected_route} 실적 분석 및 향후 {forecast_days}일 예측")
        fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Predicted_LF'], name="AI 미래 예측", mode='lines+markers')
        
        # [핵심] 현재 날짜(2026년)를 무시하고 오직 데이터가 있는 시작과 끝만 범위로 설정
        view_start = route_df['Date'].min()
        view_end = forecast_df['Date'].max()

        fig.update_layout(
            yaxis_title="Load Factor (%)",
            yaxis_range=[0, 105],
            yaxis=dict(tickformat='.0f', ticksuffix="%"),
            # X축 강제 고정: 오늘 날짜가 2026년이라도 2025년 데이터 구간만 보여줘!
            xaxis=dict(range=[view_start, view_end], autorescale=False, type="date")
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("데이터가 부족합니다.")
